//! Swap calculations and curve invariant implementation

use crate::{bn::U192, math::FeeCalculator};
use num_traits::ToPrimitive;
use stable_swap_client::{
    fees::Fees,
    solana_program::{clock::Clock, program_error::ProgramError, sysvar::Sysvar},
    state::SwapInfo,
};

/// Number of coins in a swap.
/// The Saber StableSwap only supports 2 tokens.
pub const N_COINS: u8 = 2;

/// Timestamp at 0
pub const ZERO_TS: i64 = 0;

/// Minimum ramp duration, in seconds.
pub const MIN_RAMP_DURATION: i64 = 86_400;

/// Minimum amplification coefficient.
pub const MIN_AMP: u64 = 1;

/// Maximum amplification coefficient.
pub const MAX_AMP: u64 = 1_000_000;

/// Maximum number of tokens to swap at once.
pub const MAX_TOKENS_IN: u64 = u64::MAX >> 4;

/// Encodes all results of swapping from a source token to a destination token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapResult {
    /// New amount of source token
    pub new_source_amount: u64,
    /// New amount of destination token
    pub new_destination_amount: u64,
    /// Amount of destination token swapped
    pub amount_swapped: u64,
    /// Admin fee for the swap
    pub admin_fee: u64,
    /// Fee for the swap
    pub fee: u64,
}

/// Same as SwapResult, but with u128 parameters type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapResult2 {
    /// New amount of source token
    pub new_source_amount: u128,
    /// New amount of destination token
    pub new_destination_amount: u128,
    /// Amount of destination token swapped
    pub amount_swapped: u128,
    /// Admin fee for the swap
    pub admin_fee: u128,
    /// Fee for the swap
    pub fee: u128,
}

/// The [StableSwap] invariant calculator.
///
/// This is primarily used to calculate two quantities:
/// - `D`, the swap invariant, and
/// - `Y`, the amount of tokens swapped in an instruction.
///
/// This calculator also contains several helper utilities for computing
/// swap, withdraw, and deposit amounts.
///
/// # Resources:
///
/// - [Curve StableSwap paper](https://curve.fi/files/stableswap-paper.pdf)
/// - [StableSwap Python model](https://github.com/saber-hq/stable-swap/blob/master/stable-swap-math/sim/simulation.py)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StableSwap {
    /// Initial amplification coefficient (A)
    initial_amp_factor: u64,
    /// Target amplification coefficient (A)
    target_amp_factor: u64,
    /// Current unix timestamp
    current_ts: i64,
    /// Ramp A start timestamp
    start_ramp_ts: i64,
    /// Ramp A stop timestamp
    stop_ramp_ts: i64,
}

impl TryFrom<&SwapInfo> for StableSwap {
    type Error = ProgramError;

    fn try_from(info: &SwapInfo) -> Result<Self, ProgramError> {
        Ok(StableSwap::new_from_swap_info(
            info,
            Clock::get()?.unix_timestamp,
        ))
    }
}

impl StableSwap {
    /// Constructs a new [StableSwap] from a [SwapInfo].
    pub fn new_from_swap_info(info: &SwapInfo, current_ts: i64) -> StableSwap {
        StableSwap::new(
            info.initial_amp_factor,
            info.target_amp_factor,
            current_ts,
            info.start_ramp_ts,
            info.stop_ramp_ts,
        )
    }

    /// Constructs a new [StableSwap] invariant calculator.
    pub fn new(
        initial_amp_factor: u64,
        target_amp_factor: u64,
        current_ts: i64,
        start_ramp_ts: i64,
        stop_ramp_ts: i64,
    ) -> Self {
        Self {
            initial_amp_factor,
            target_amp_factor,
            current_ts,
            start_ramp_ts,
            stop_ramp_ts,
        }
    }

    fn compute_next_d2(
        &self,
        amp_factor: u64,
        d_init: U192,
        d_prod: U192,
        sum_x: u128,
    ) -> Option<U192> {
        let ann = amp_factor.checked_mul(N_COINS.into())?;
        let leverage = sum_x.checked_mul(ann.into())?;
        // d = (ann * sum_x + d_prod * n_coins) * d / ((ann - 1) * d + (n_coins + 1) * d_prod)
        let numerator = d_init.checked_mul(
            d_prod
                .checked_mul(N_COINS.into())?
                .checked_add(leverage.into())?,
        )?;
        let denominator = d_init
            .checked_mul(ann.checked_sub(1)?.into())?
            .checked_add(d_prod.checked_mul((N_COINS.checked_add(1)?).into())?)?;
        numerator.checked_div(denominator)
    }

    fn compute_next_d(
        &self,
        amp_factor: u64,
        d_init: U192,
        d_prod: U192,
        sum_x: u64,
    ) -> Option<U192> {
        self.compute_next_d2(amp_factor, d_init, d_prod, sum_x.into())
    }

    /// Compute the amplification coefficient (A).
    ///
    /// The amplification coefficient is used to determine the slippage incurred when
    /// performing swaps. The lower it is, the closer the invariant is to the constant product[^stableswap].
    ///
    /// The amplication coefficient linearly increases with respect to time,
    /// based on the [`SwapInfo::start_ramp_ts`] and [`SwapInfo::stop_ramp_ts`] parameters.
    ///
    /// [^stableswap]: [Egorov, "StableSwap," 2019.](https://curve.fi/files/stableswap-paper.pdf)
    pub fn compute_amp_factor(&self) -> Option<u64> {
        if self.current_ts < self.stop_ramp_ts {
            let time_range = self.stop_ramp_ts.checked_sub(self.start_ramp_ts)?;
            let time_delta = self.current_ts.checked_sub(self.start_ramp_ts)?;

            // Compute amp factor based on ramp time
            if self.target_amp_factor >= self.initial_amp_factor {
                // Ramp up
                let amp_range = self
                    .target_amp_factor
                    .checked_sub(self.initial_amp_factor)?;
                let amp_delta = (amp_range as u128)
                    .checked_mul(time_delta.to_u128()?)?
                    .checked_div(time_range.to_u128()?)?
                    .to_u64()?;
                self.initial_amp_factor.checked_add(amp_delta)
            } else {
                // Ramp down
                let amp_range = self
                    .initial_amp_factor
                    .checked_sub(self.target_amp_factor)?;
                let amp_delta = (amp_range as u128)
                    .checked_mul(time_delta.to_u128()?)?
                    .checked_div(time_range.to_u128()?)?
                    .to_u64()?;
                self.initial_amp_factor.checked_sub(amp_delta)
            }
        } else {
            // when stop_ramp_ts == 0 or current_ts >= stop_ramp_ts
            Some(self.target_amp_factor)
        }
    }

    /// Original compute_d, but with u128 type as parameters
    pub fn compute_d2(&self, amount_a: u128, amount_b: u128) -> Option<U192> {
        let sum_x = amount_a.checked_add(amount_b)?; // sum(x_i), a.k.a S
        if sum_x == 0 {
            Some(0.into())
        } else {
            let amp_factor = self.compute_amp_factor()?;
            let amount_a_times_coins = amount_a.checked_mul(N_COINS.into())?;
            let amount_b_times_coins = amount_b.checked_mul(N_COINS.into())?;

            // Newton's method to approximate D
            let mut d_prev: U192;
            let mut d: U192 = sum_x.into();
            for _ in 0..256 {
                let mut d_prod = d;
                d_prod = d_prod
                    .checked_mul(d)?
                    .checked_div(amount_a_times_coins.into())?;
                d_prod = d_prod
                    .checked_mul(d)?
                    .checked_div(amount_b_times_coins.into())?;
                d_prev = d;
                d = self.compute_next_d2(amp_factor, d, d_prod, sum_x)?;
                // Equality with the precision of 1
                if d > d_prev {
                    if d.checked_sub(d_prev)? <= 1.into() {
                        break;
                    }
                } else if d_prev.checked_sub(d)? <= 1.into() {
                    break;
                }
            }

            Some(d)
        }
    }

    /// Computes the Stable Swap invariant (D).
    ///
    /// The invariant is defined as follows:
    ///
    /// ```text
    /// A * sum(x_i) * n**n + D = A * D * n**n + D**(n+1) / (n**n * prod(x_i))
    /// ```
    ///
    /// # Arguments
    ///
    /// - `amount_a` - The amount of token A owned by the LP pool. (i.e. token A reserves)
    /// - `amount_b` - The amount of token B owned by the LP pool. (i.e. token B reserves)
    ///
    /// *For more info on reserves, see [stable_swap_client::state::SwapTokenInfo::reserves].*
    pub fn compute_d(&self, amount_a: u64, amount_b: u64) -> Option<U192> {
        self.compute_d2(amount_a as u128, amount_b as u128)
    }

    /// Same as compute_mint_amount_for_deposit, but with u128 parameters type
    pub fn compute_mint_amount_for_deposit2(
        &self,
        deposit_amount_a: u128,
        deposit_amount_b: u128,
        swap_amount_a: u128,
        swap_amount_b: u128,
        pool_token_supply: u64,
        fees: &Fees,
    ) -> Option<u64> {
        // Initial invariant
        let d_0 = self.compute_d2(swap_amount_a, swap_amount_b)?;
        let old_balances = [swap_amount_a, swap_amount_b];
        let mut new_balances = [
            swap_amount_a.checked_add(deposit_amount_a)?,
            swap_amount_b.checked_add(deposit_amount_b)?,
        ];
        // Invariant after change
        let d_1 = self.compute_d2(new_balances[0], new_balances[1])?;
        if d_1 <= d_0 {
            None
        } else {
            // Recalculate the invariant accounting for fees
            for i in 0..new_balances.len() {
                let ideal_balance = d_1
                    .checked_mul(old_balances[i].into())?
                    .checked_div(d_0)?
                    .to_u128()?;
                let difference = if ideal_balance > new_balances[i] {
                    ideal_balance.checked_sub(new_balances[i])?
                } else {
                    new_balances[i].checked_sub(ideal_balance)?
                };
                let fee = fees.normalized_trade_fee2(N_COINS, difference)?;
                new_balances[i] = new_balances[i].checked_sub(fee)?;
            }

            let d_2 = self.compute_d2(new_balances[0], new_balances[1])?;
            U192::from(pool_token_supply)
                .checked_mul(d_2.checked_sub(d_0)?)?
                .checked_div(d_0)?
                .to_u64()
        }
    }

    /// Computes the amount of pool tokens to mint after a deposit.
    pub fn compute_mint_amount_for_deposit(
        &self,
        deposit_amount_a: u64,
        deposit_amount_b: u64,
        swap_amount_a: u64,
        swap_amount_b: u64,
        pool_token_supply: u64,
        fees: &Fees,
    ) -> Option<u64> {
        self.compute_mint_amount_for_deposit2(
            deposit_amount_a.into(),
            deposit_amount_b.into(),
            swap_amount_a.into(),
            swap_amount_b.into(),
            pool_token_supply,
            fees,
        )
    }

    /// Same as compute_y_raw, but with u128 as parameters type
    pub fn compute_y_raw2(&self, x: u128, d: U192) -> Option<U192> {
        let amp_factor = self.compute_amp_factor()?;
        let ann = amp_factor.checked_mul(N_COINS.into())?; // A * n ** n

        // sum' = prod' = x
        // c =  D ** (n + 1) / (n ** (2 * n) * prod' * A)
        let mut c = d
            .checked_mul(d)?
            .checked_div(x.checked_mul(N_COINS.into())?.into())?;
        c = c
            .checked_mul(d)?
            .checked_div(ann.checked_mul(N_COINS.into())?.into())?;
        // b = sum' - (A*n**n - 1) * D / (A * n**n)
        let b = d.checked_div(ann.into())?.checked_add(x.into())?; // d is subtracted on line 147

        // Solve for y by approximating: y**2 + b*y = c
        let mut y_prev: U192;
        let mut y = d;
        for _ in 0..256 {
            y_prev = y;
            // y = (y * y + c) / (2 * y + b - d);
            let y_numerator = y.checked_pow(2.into())?.checked_add(c)?;
            let y_denominator = y.checked_mul(2.into())?.checked_add(b)?.checked_sub(d)?;
            y = y_numerator.checked_div(y_denominator)?;
            if y > y_prev {
                if y.checked_sub(y_prev)? <= 1.into() {
                    break;
                }
            } else if y_prev.checked_sub(y)? <= 1.into() {
                break;
            }
        }
        Some(y)
    }

    /// Compute the swap amount `y` in proportion to `x`.
    ///
    /// Solve for `y`:
    ///
    /// ```text
    /// y**2 + y * (sum' - (A*n**n - 1) * D / (A * n**n)) = D ** (n + 1) / (n ** (2 * n) * prod' * A)
    /// y**2 + b*y = c
    /// ```
    #[allow(clippy::many_single_char_names)]
    pub fn compute_y_raw(&self, x: u64, d: U192) -> Option<U192> {
        self.compute_y_raw2(x.into(), d)
    }

    /// Same as compute_y, but with u128 parameters type
    pub fn compute_y2(&self, x: u128, d: U192) -> Option<u128> {
        self.compute_y_raw2(x, d)?.to_u128()
    }

    /// Computes the swap amount `y` in proportion to `x`.
    pub fn compute_y(&self, x: u64, d: U192) -> Option<u64> {
        self.compute_y_raw(x, d)?.to_u64()
    }
    /// Same as compute_withdraw_one, but with u128 parameters type
    pub fn compute_withdraw_one2(
        &self,
        pool_token_amount: u64,
        pool_token_supply: u64,
        swap_base_amount: u128,  // Same denomination of token to be withdrawn
        swap_quote_amount: u128, // Counter denomination of token to be withdrawn
        fees: &Fees,
    ) -> Option<(u128, u128)> {
        let d_0 = self.compute_d2(swap_base_amount, swap_quote_amount)?;
        let d_1 = d_0.checked_sub(
            U192::from(pool_token_amount)
                .checked_mul(d_0)?
                .checked_div(pool_token_supply.into())?,
        )?;
        let new_y = self.compute_y2(swap_quote_amount, d_1)?;

        // expected_base_amount = swap_base_amount * d_1 / d_0 - new_y;
        let expected_base_amount = U192::from(swap_base_amount)
            .checked_mul(d_1)?
            .checked_div(d_0)?
            .to_u128()?
            .checked_sub(new_y)?;
        // expected_quote_amount = swap_quote_amount - swap_quote_amount * d_1 / d_0;
        let expected_quote_amount = swap_quote_amount.checked_sub(
            U192::from(swap_quote_amount)
                .checked_mul(d_1)?
                .checked_div(d_0)?
                .to_u128()?,
        )?;
        // new_base_amount = swap_base_amount - expected_base_amount * fee / fee_denominator;
        let new_base_amount = swap_base_amount
            .checked_sub(fees.normalized_trade_fee2(N_COINS, expected_base_amount)?)?;
        // new_quote_amount = swap_quote_amount - expected_quote_amount * fee / fee_denominator;
        let new_quote_amount = swap_quote_amount
            .checked_sub(fees.normalized_trade_fee2(N_COINS, expected_quote_amount)?)?;
        let dy = new_base_amount
            .checked_sub(self.compute_y2(new_quote_amount, d_1)?)?
            .checked_sub(1)?; // Withdraw less to account for rounding errors
        let dy_0 = swap_base_amount.checked_sub(new_y)?;

        Some((dy, dy_0.checked_sub(dy)?))
    }

    /// Calculates the withdrawal amount when withdrawing only one type of token.
    ///
    /// Calculation:
    ///
    /// 1. Get current D
    /// 2. Solve Eqn against `y_i` for `D - _token_amount`
    pub fn compute_withdraw_one(
        &self,
        pool_token_amount: u64,
        pool_token_supply: u64,
        swap_base_amount: u64,  // Same denomination of token to be withdrawn
        swap_quote_amount: u64, // Counter denomination of token to be withdrawn
        fees: &Fees,
    ) -> Option<(u64, u64)> {
        let (dy, dy_0) = self.compute_withdraw_one2(
            pool_token_amount,
            pool_token_supply,
            swap_base_amount.into(),
            swap_quote_amount.into(),
            fees,
        )?;

        Some((dy.try_into().ok()?, dy_0.try_into().ok()?))
    }

    /// Same as swap_to, but with u128 parameters type
    pub fn swap_to2(
        &self,
        source_amount: u128,
        swap_source_amount: u128,
        swap_destination_amount: u128,
        fees: &Fees,
    ) -> Option<SwapResult2> {
        let y = self.compute_y2(
            swap_source_amount.checked_add(source_amount)?,
            self.compute_d2(swap_source_amount, swap_destination_amount)?,
        )?;
        // https://github.com/curvefi/curve-contract/blob/b0bbf77f8f93c9c5f4e415bce9cd71f0cdee960e/contracts/pool-templates/base/SwapTemplateBase.vy#L466
        let dy = swap_destination_amount.checked_sub(y)?.checked_sub(1)?;
        let dy_fee = fees.trade_fee2(dy)?;
        let admin_fee = fees.admin_trade_fee2(dy_fee)?;

        let amount_swapped = dy.checked_sub(dy_fee)?;
        let new_destination_amount = swap_destination_amount
            .checked_sub(amount_swapped)?
            .checked_sub(admin_fee)?;
        let new_source_amount = swap_source_amount.checked_add(source_amount)?;

        Some(SwapResult2 {
            new_source_amount,
            new_destination_amount,
            amount_swapped,
            admin_fee,
            fee: dy_fee,
        })
    }

    /// Compute SwapResult after an exchange
    pub fn swap_to(
        &self,
        source_amount: u64,
        swap_source_amount: u64,
        swap_destination_amount: u64,
        fees: &Fees,
    ) -> Option<SwapResult> {
        let SwapResult2 {
            new_source_amount,
            new_destination_amount,
            amount_swapped,
            admin_fee,
            fee,
        } = self.swap_to2(
            source_amount.into(),
            swap_source_amount.into(),
            swap_destination_amount.into(),
            fees,
        )?;

        Some(SwapResult {
            new_source_amount: new_source_amount.try_into().ok()?,
            new_destination_amount: new_destination_amount.try_into().ok()?,
            amount_swapped: amount_swapped.try_into().ok()?,
            admin_fee: admin_fee.try_into().ok()?,
            fee: fee.try_into().ok()?,
        })
    }
}
