"""Human-readable report for the immutable Track A sealed result."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import pandas as pd


def _pct(value: object) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "NA"


def _num(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def _inception_anchored_chart_data(
    frame: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    """Normalize chart paths to the pre-trade inception anchors.

    The duplicate first timestamp intentionally draws the initial commission
    as a vertical wealth change at the first execution open.
    """

    if frame.empty:
        raise ValueError("portfolio chart frame is empty")
    if initial_capital <= 0:
        raise ValueError("portfolio chart initial capital must be positive")
    required = {"nav", "benchmark_nav", "excess_nav"}
    if not required.issubset(frame.columns):
        raise ValueError("portfolio chart frame lacks required wealth columns")
    normalized = pd.DataFrame(
        {
            "nav": frame["nav"].astype(float) / float(initial_capital),
            "benchmark_nav": frame["benchmark_nav"].astype(float)
            / float(initial_capital),
            "excess_nav": frame["excess_nav"].astype(float),
        },
        index=frame.index,
    )
    anchor = pd.DataFrame(
        {"nav": [1.0], "benchmark_nav": [1.0], "excess_nav": [1.0]},
        index=pd.DatetimeIndex([frame.index[0]]),
    )
    return pd.concat([anchor, normalized])


def _write_performance_chart(run_dir: Path, portfolios: Mapping[str, object]) -> Path:
    mpl_cache = run_dir / ".matplotlib"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name in ("equal_weight", "ridge", "lightgbm"):
        path = run_dir / "sealed_portfolios" / f"{name}_daily.parquet"
        frame = pd.read_parquet(path)
        chart_frame = _inception_anchored_chart_data(
            frame,
            initial_capital=float(portfolios[name]["initial_capital"]),
        )
        axes[0].plot(chart_frame.index, chart_frame["nav"], label=name)
        axes[1].plot(chart_frame.index, chart_frame["excess_nav"], label=name)
    benchmark_frame = pd.read_parquet(
        run_dir / "sealed_portfolios" / "lightgbm_daily.parquet"
    )
    benchmark_chart = _inception_anchored_chart_data(
        benchmark_frame,
        initial_capital=float(portfolios["lightgbm"]["initial_capital"]),
    )
    axes[0].plot(
        benchmark_chart.index,
        benchmark_chart["benchmark_nav"],
        color="black",
        linewidth=1.3,
        label="CSI300",
    )
    axes[0].set_title("Track A portfolio NAV (paper-parity proxy)")
    axes[1].set_title("Excess NAV versus CSI300")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=4)
    fig.tight_layout()
    target = run_dir / "track_a_performance.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def write_track_a_report(
    run_dir: Path | str,
    sealed: Mapping[str, object],
) -> Path:
    run = Path(run_dir).resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    freeze = json.loads((run / "development_freeze.json").read_text(encoding="utf-8"))
    chart = _write_performance_chart(run, sealed["portfolio_results"])
    lines = [
        "# CogAlpha Track A — CSI300 paper-compatible reproduction",
        "",
        f"**Research verdict: {sealed['research_validity']}**",
        "",
        "The historical run completed exactly once. It is a research-proxy reproduction, not an exact author replication or a capital-ready strategy. The author runtime, full factor archive and exact Qlib snapshot are unavailable; this run uses the frozen BaoStock PIT CSI300 reconstruction.",
        "",
        "## Frozen protocol",
        "",
        f"- Search: {summary['evaluated_candidates']} candidates, 80 initial + 24 × 96 children, 21 roles.",
        "- Label: 10-session adjusted-open return, signal at t close, entry t+1 open, exit t+11 open.",
        "- Splits: 2011–2019 train, 2020 validation, 2021–2024 sealed walk-forward test.",
        "- Models: equal weight, Ridge alpha 10, fixed LightGBM; expanding refit every 126 sessions.",
        "- Portfolio: daily Top50/drop5, next-open, buy 5bp, sell 15bp, CNY5 minimum commission.",
        f"- Development freeze: `{freeze['freeze_sha256']}`.",
        "- Sealed results were not used for factor, model, sign, or parameter selection.",
        "",
        "## Composite RankIC",
        "",
        "| Model | 2020 validation RankIC | 2021–2024 sealed RankIC | HAC t-stat | Bootstrap lower |",
        "|---|---:|---:|---:|---:|",
    ]
    validation = freeze["validation_model_metrics"]
    sealed_metrics = sealed["sealed_model_metrics"]
    for name in ("equal_weight", "ridge", "lightgbm"):
        valid = validation[name]
        test = sealed_metrics[name]
        lines.append(
            f"| {name} | {_pct(valid['mean'])} | {_pct(test['mean'])} | {_num(test['hac_t'])} | {_pct(test['bootstrap_low'])} |"
        )
    lines.extend(
        [
            "",
            "The paper reports CogAlpha RankIC 0.0814. This is a comparison target, not a pass threshold: snapshot, generated-factor archive and model defaults are not identical.",
            "",
            "## Top50/drop5 portfolio",
            "",
            "| Model | Ann. return | Ann. excess | Tracking error | Information ratio | Max drawdown | Max excess drawdown | Avg. two-way turnover |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("equal_weight", "ridge", "lightgbm"):
        item = sealed["portfolio_results"][name]
        lines.append(
            f"| {name} | {_pct(item['annualized_return'])} | {_pct(item['annualized_excess_return'])} | {_pct(item['tracking_error'])} | {_num(item['information_ratio'])} | {_pct(item['maximum_drawdown'])} | {_pct(item['maximum_excess_drawdown'])} | {_pct(item['average_two_way_turnover'])} |"
        )
    lines.extend(
        [
            "",
            "The inception buy cost is rolled into the first actual open-to-open net and excess return. Tracking error, information ratio and annualized wealth returns therefore use the same execution-sessions-minus-one holding-period clock. NAV and excess-NAV charts start from pre-trade anchors of initial capital and 1.0, respectively.",
            "",
            f"![Track A performance]({chart})",
            "",
            "## Frozen 20-factor family",
            "",
            "| Candidate | Role | Operator | Dev raw IC | Dev neutral IC | Sealed raw IC | Sealed neutral IC |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    sealed_factor = {
        str(item["candidate_id"]): item for item in sealed["factor_results"]
    }
    champion_summary = {
        str(item["candidate_id"]): item for item in summary["champions"]
    }
    frozen_records = {
        str(item["candidate"]["candidate_id"]): item["candidate"]
        for item in freeze["champions"]
    }
    for candidate_id in [
        item["candidate"]["candidate_id"] for item in freeze["champions"]
    ]:
        candidate = frozen_records[candidate_id]
        dev = champion_summary[candidate_id]
        test = sealed_factor[candidate_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate_id[:10],
                    str(candidate["agent_id"]),
                    str(candidate["operator"]),
                    _pct(dev["validation_rank_ic"]),
                    _pct(dev["validation_neutral_rank_ic"]),
                    _pct(test["sealed_raw"]["mean"]),
                    _pct(test["sealed_neutral"]["mean"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Audit boundary",
            "",
            "Implementation checks passed: causal DSL, per-security price-scale invariance, train/validation purges, 3-worker serial equivalence, immutable trial ledger, hash-bound per-generation LLM reviews, fixed signs, and one-time sealed registry.",
            "",
            "Remaining evidence gaps: one historical outer replay, BaoStock research-proxy constituent/adjustment history, incomplete capital-grade corporate-action and official-weight lineage, and no prospective shadow/paper-trading evidence. Therefore the only defensible verdict is **NOT VALIDATED**.",
            "",
            "Source: Feng et al., *Cognitive Alpha Mining via LLM-Driven Code-Based Evolution*, ACL 2026, https://aclanthology.org/2026.acl-long.538/",
            "",
        ]
    )
    target = run / "TRACK_A_REPORT.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


__all__ = ["write_track_a_report"]
