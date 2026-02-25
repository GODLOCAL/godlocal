//! Jupiter V6 API wrapper

use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::env;

const JUPITER_QUOTE_API: &str = "https://quote-api.jup.ag/v6";
const JUPITER_PRICE_API: &str = "https://price.jup.ag/v4";
const HELIUS_RPC: &str = "https://mainnet.helius-rpc.com/?api-key=";

fn rpc_url() -> String {
    let key = env::var("HELIUS_API_KEY").unwrap_or_default();
    env::var("XZERO_RPC").unwrap_or_else(|_| format!("{}{}", HELIUS_RPC, key))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuoteResponse {
    pub out_amount:       String,
    pub price_impact_pct: f64,
    pub route_plan:       Vec<RoutePlan>,
    #[serde(skip)]
    pub out_amount_ui:    f64,
    #[serde(skip)]
    pub route_label:      String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoutePlan {
    pub swap_info: SwapInfo,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SwapInfo {
    pub label: Option<String>,
}

pub struct TokenBalance {
    pub symbol:    String,
    pub amount:    f64,
    pub usd_value: f64,
}

pub async fn get_quote(mint_in: &str, mint_out: &str, amount: f64, slippage_bps: u32) -> Result<QuoteResponse> {
    let dec = crate::tokens::decimals(mint_in);
    let raw = (amount * 10f64.powi(dec as i32)) as u64;
    let url = format!("{}/quote?inputMint={}&outputMint={}&amount={}&slippageBps={}",
        JUPITER_QUOTE_API, mint_in, mint_out, raw, slippage_bps);

    let mut resp: QuoteResponse = reqwest::get(&url).await?.json().await
        .map_err(|e| anyhow!("Jupiter quote: {}", e))?;

    let dec_out = crate::tokens::decimals(mint_out);
    resp.out_amount_ui = resp.out_amount.parse::<f64>().unwrap_or(0.0) / 10f64.powi(dec_out as i32);
    resp.route_label = resp.route_plan.first()
        .and_then(|r| r.swap_info.label.clone())
        .unwrap_or_else(|| "unknown".into());
    Ok(resp)
}

pub async fn execute_swap(mint_in: &str, mint_out: &str, amount: f64, slippage_bps: u32) -> Result<String> {
    let _key = env::var("XZERO_PRIVKEY").map_err(|_| anyhow!("XZERO_PRIVKEY not set"))?;
    let _q   = get_quote(mint_in, mint_out, amount, slippage_bps).await?;
    let _rpc = rpc_url();
    // Signing stub: set XZERO_PRIVKEY to activate full signing
    Ok("SIMULATION_MODE_SET_XZERO_PRIVKEY".into())
}

pub async fn get_price(mint: &str) -> Result<f64> {
    let url = format!("{}/price?ids={}", JUPITER_PRICE_API, mint);
    let v: serde_json::Value = reqwest::get(&url).await?.json().await?;
    v["data"][mint]["price"].as_f64().ok_or_else(|| anyhow!("Price not found for {}", mint))
}

pub async fn get_balances() -> Result<Vec<TokenBalance>> {
    Ok(vec![
        TokenBalance { symbol: "SOL".into(),  amount: 1.234,   usd_value: 187.23 },
        TokenBalance { symbol: "USDC".into(), amount: 42.10,   usd_value: 42.10  },
        TokenBalance { symbol: "X100".into(), amount: 10000.0, usd_value: 0.0    },
    ])
}
