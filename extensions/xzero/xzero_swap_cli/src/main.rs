//! xzero-swap — X-ZERO Solana swap CLI for GodLocal agents
//!
//! Usage:
//!   xzero-swap quote --in SOL --out USDC --amount 0.5
//!   xzero-swap swap  --in SOL --out USDC --amount 0.5 --slippage 50
//!   xzero-swap dca   --in USDC --out SOL --amount 10 --every 3600 --over 86400
//!   xzero-swap price --token SOL
//!   xzero-swap balance
//!
//! Env vars:
//!   XZERO_PRIVKEY  — base58 Solana private key (agent wallet)
//!   XZERO_RPC      — Solana RPC URL (default: Helius mainnet)

mod jupiter;
mod dca;
mod tokens;

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;

#[derive(Parser)]
#[command(name = "xzero-swap", about = "X-ZERO Solana agent wallet CLI", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Get a swap quote (no execution)
    Quote {
        #[arg(long)] r#in:   String,
        #[arg(long)] out:    String,
        #[arg(long)] amount: f64,
    },
    /// Execute a swap via Jupiter V6
    Swap {
        #[arg(long)] r#in:     String,
        #[arg(long)] out:      String,
        #[arg(long)] amount:   f64,
        /// Slippage in basis points (default 50 = 0.5%)
        #[arg(long, default_value = "50")] slippage: u32,
    },
    /// Set up a DCA order (USDC->SOL every N seconds)
    Dca {
        #[arg(long)] r#in:    String,
        #[arg(long)] out:     String,
        #[arg(long)] amount:  f64,
        /// Cycle interval in seconds
        #[arg(long)] every:   u64,
        /// Total duration in seconds
        #[arg(long)] over:    u64,
    },
    /// Get current token price in USD
    Price {
        #[arg(long)] token: String,
    },
    /// Show agent wallet balances (SOL + top SPL tokens)
    Balance,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Quote { r#in, out, amount } => {
            let mint_in  = tokens::resolve(&r#in)?;
            let mint_out = tokens::resolve(&out)?;
            let quote    = jupiter::get_quote(&mint_in, &mint_out, amount, 50).await?;
            println!("{}", format!(
                "Quote: {} {} -> {} {}   impact: {:.3}%   route: {}",
                amount, r#in.to_uppercase().green(),
                quote.out_amount_ui, out.to_uppercase().cyan(),
                quote.price_impact_pct * 100.0,
                quote.route_label,
            ).bold());
        }

        Cmd::Swap { r#in, out, amount, slippage } => {
            let mint_in  = tokens::resolve(&r#in)?;
            let mint_out = tokens::resolve(&out)?;
            let quote = jupiter::get_quote(&mint_in, &mint_out, amount, slippage).await?;
            println!("{} {} -> {} {} @ slippage {}bps",
                "Swapping".yellow(), amount, quote.out_amount_ui, out.cyan(), slippage);
            let sig = jupiter::execute_swap(&mint_in, &mint_out, amount, slippage).await?;
            println!("{} https://solscan.io/tx/{}", "Confirmed:".green().bold(), sig);
        }

        Cmd::Dca { r#in, out, amount, every, over } => {
            let mint_in  = tokens::resolve(&r#in)?;
            let mint_out = tokens::resolve(&out)?;
            let cycles   = over / every;
            println!("{} {} {} -> {} every {}s for {} cycles",
                "DCA:".cyan().bold(), amount, r#in.green(), out.cyan(), every, cycles);
            let order_id = dca::create_order(&mint_in, &mint_out, amount, every, over).await?;
            println!("{} order_id={}", "DCA active:".green().bold(), order_id);
        }

        Cmd::Price { token } => {
            let mint = tokens::resolve(&token)?;
            let price = jupiter::get_price(&mint).await?;
            println!("{}: ${:.4}", token.to_uppercase().cyan().bold(), price);
        }

        Cmd::Balance => {
            let balances = jupiter::get_balances().await?;
            println!("{}", "Agent Wallet Balances:".bold().cyan());
            for b in &balances {
                println!("  {:>8}: {:.6}  (${:.2})", b.symbol.green(), b.amount, b.usd_value);
            }
            let total: f64 = balances.iter().map(|b| b.usd_value).sum();
            println!("{} ${:.2}", "  Total:".bold(), total);
        }
    }
    Ok(())
}
