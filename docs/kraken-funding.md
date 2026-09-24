# Kraken continuous funding

This fork starts from Freqtrade 2026.8. The local package version
`2026.8+kraken.2` identifies its Kraken funding correction. The patch targets
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
format. Other exchanges keep their existing columns and accounting.

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

Backtests recompute each open trade's funding at every candle. The calculator
finds the held hours by binary search and sums them with array arithmetic, so a
long holding does not make a backtest slow down quadratically.

## Maintenance

Keep changes limited to this correction and its regression tests. Before adopting
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
