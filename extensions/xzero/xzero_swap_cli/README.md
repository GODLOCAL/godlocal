# xzero-swap

X-ZERO Solana swap CLI for GodLocal agents. Single Rust binary — Steam Deck ready.

## Build

```bash
cd extensions/xzero/xzero_swap_cli
cargo build --release
# Binary: target/release/xzero-swap
```

## Setup

```env
XZERO_PRIVKEY=<base58-agent-wallet-private-key>
XZERO_RPC=https://mainnet.helius-rpc.com/?api-key=YOUR_KEY
HELIUS_API_KEY=YOUR_KEY
```

## Commands

```bash
xzero-swap quote --in SOL --out USDC --amount 0.5
xzero-swap swap  --in SOL --out USDC --amount 0.5 --slippage 50
xzero-swap dca   --in USDC --out SOL --amount 10 --every 3600 --over 86400
xzero-swap price --token SOL
xzero-swap balance
```

## Steam Deck cross-compile (Mac -> x86_64)

```bash
rustup target add x86_64-unknown-linux-gnu
cargo build --release --target x86_64-unknown-linux-gnu
scp target/x86_64-unknown-linux-gnu/release/xzero-swap deck@steamdeck:~/
```

## Status
- quote, price, DCA order creation, balance display: ready
- execute_swap signing: stub — set XZERO_PRIVKEY to activate
- X100 mint: update after pump.fun launch
