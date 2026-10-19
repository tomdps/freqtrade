# Kraken continuous funding

This fork starts from Freqtrade 2026.8. The local package version
`2026.8+kraken.5` identifies its funding corrections. The patch targets
backtesting and dry-run accounting; it has not been reconciled against a live
account ledger.

Kraken's linear perpetual contracts accrue funding continuously. Each hourly
rate applies from its timestamp until the next hour, and only the time actually
held contributes to a payment. See the public
[contract specification](https://support.kraken.com/articles/4844359082772-linear-multi-collateral-derivatives-contract-specifications)
and [historical funding schema](https://docs.kraken.com/api-reference/historical-funding-rates/historical-funding-rates).

The calculator uses Kraken's absolute hourly rate per base unit. A relative rate
multiplied by a mark candle is not an exact substitute. Downloaded funding frames
retain the existing relative `funding_rate` for strategy signals and an optional
`funding_rate_absolute` for payments, using the existing JSON, Feather or Parquet
format. Other exchanges keep their existing funding columns and payment calculators.
Binance's opt-in isolated cash/margin allocation is described in
[Funding and isolated collateral](leverage.md#funding-and-isolated-collateral-in-simulations).

Old Kraken files containing only relative rates are rejected for payment
calculation. Download the required funding history into a separate directory;
keep existing research data intact. Missing or invalid hourly rates are errors,
not zero payments or permission to carry a rate across a gap. Relative funding
signals still need their normal publication-time safeguards.

Funding is accrued before a simulated fill changes exposure and before decisions
use profit or available balance. Tests cover elapsed-time boundaries, signs,
rate changes, position adjustments, repeated order updates and open-position
funding. The dry-run fill model does not establish real exchange queue priority
or intraminute partial-fill fidelity.

Kraken publishes each hourly rate shortly after the hour starts, and its history
can be briefly unreachable. In dry-run and live trading, a funding update that
needs a rate not yet published, or a history that cannot be fetched, keeps the
previous funding and is retried on a later loop instead of stopping the bot. A
response that cannot be read counts as a history that cannot be fetched. An order
that fills in that window keeps its fill time, price and fee, and is booked once
the funding up to its fill is known, usually one loop later. Until then it stays
open, and a later exit or added entry does not replace it. If a pair's history
stays unavailable, its funding and bookings keep waiting, with a warning on every
loop. A gap inside the published history still stops the bot, and backtests still
refuse any history that does not cover a holding.

Backtests ask for each open trade's funding at every candle, and at every detail
candle with `--timeframe-detail`. The calculator prepares each funding frame once
and keeps a running total per holding: hours already held in full are summed
once, in time order, and each call adds only the hours since the previous one. The
additions are the same as summing the whole holding again, so results are
unchanged, and a long backtest no longer slows down quadratically. In detail
minutes without an open trade, the backtest still records the wallet but skips
the funding and wallet refresh, since nothing accrues.

## Maintenance

Keep changes limited to these funding/collateral corrections and their regression tests. Before adopting
an upstream release, replay the patch in a branch, run the offline financial and
data tests from `.github/workflows/ci.yml`, review the exact commit and require
passing CI. Remove the local correction when the upstream implementation passes
the same cases. Pin consumers to a reviewed commit rather than a moving branch.

Use only generic synthetic fixtures and public documentation in changes, commits,
PRs and test output. Keep downstream configurations, strategies, datasets, results
and operational information outside this repository.

The fork's CI runs offline tests with internet sockets blocked. Publishing and
credential-dependent upstream maintenance jobs are restricted to the upstream
repository. Updating this library does not itself update a running installation.
