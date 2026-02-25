//! Jupiter DCA V2 orders

use anyhow::{anyhow, Result};
use std::env;

pub async fn create_order(mint_in: &str, mint_out: &str, amount_per_cycle: f64, cycle_secs: u64, total_secs: u64) -> Result<String> {
    let _key = env::var("XZERO_PRIVKEY").map_err(|_| anyhow!("XZERO_PRIVKEY not set"))?;
    let cycles = total_secs / cycle_secs;
    Ok(format!("DCA_{}_{}_{}_{}s", &mint_in[..6], &mint_out[..6], cycles, cycle_secs))
}

pub async fn cancel_order(order_id: &str) -> Result<()> {
    let _ = order_id;
    Ok(())
}
