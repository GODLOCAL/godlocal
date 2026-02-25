//! Solana token registry

use anyhow::{anyhow, Result};

pub fn resolve(sym: &str) -> Result<String> {
    match sym.to_lowercase().as_str() {
        "sol"  => Ok("So11111111111111111111111111111111111111112".into()),
        "usdc" => Ok("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".into()),
        "usdt" => Ok("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB".into()),
        "jup"  => Ok("JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN".into()),
        "ray"  => Ok("4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R".into()),
        "bonk" => Ok("DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263".into()),
        "wif"  => Ok("EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm".into()),
        "x100" => Ok("X100111111111111111111111111111111111111111".into()),
        s if s.len() >= 32 => Ok(s.to_string()),
        other => Err(anyhow!("Unknown token: {}", other)),
    }
}

pub fn decimals(mint: &str) -> u8 {
    match mint {
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" => 6,  // USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB" => 6,  // USDT
        "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263" => 5,  // BONK
        _ => 9,  // SOL + most SPL tokens
    }
}
