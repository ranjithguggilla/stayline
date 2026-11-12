"""Charts for the README. One figure per question the reader is likely to ask."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, ACCENT, GREY, WARN = "#1f2933", "#2a6f97", "#9aa5b1", "#c8553d"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def cv_comparison(summary, shipped: str, path: Path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    s = summary.sort_values("log_loss", ascending=False)
    colors = [ACCENT if m == shipped else GREY for m in s.model]
    ax.barh(s.model, s.log_loss, xerr=s.log_loss_se, color=colors, ecolor=INK, capsize=3)
    ax.set_xlabel("cross-validated log loss (lower is better), bars show one corrected SE")
    ax.set_xlim(left=s.log_loss.min() * 0.8)
    ax.grid(axis="x", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibration(y, p, path: Path, bins: int = 8):
    order = np.argsort(p)
    xs, ys = [], []
    for idx in np.array_split(order, bins):
        xs.append(p[idx].mean())
        ys.append(y[idx].mean())
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, max(xs + ys) * 1.05], [0, max(xs + ys) * 1.05], color=GREY, ls="--", label="perfect")
    ax.plot(xs, ys, "o-", color=ACCENT, label="held-out test")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed attrition rate")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance(df, path: Path, top: int = 14):
    d = df.head(top).iloc[::-1]
    noise_max = df[df.is_noise].importance.max()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        d.feature, d.importance, xerr=d.sd, color=[WARN if n else ACCENT for n in d.is_noise], ecolor=INK, capsize=2
    )
    ax.axvline(noise_max, color=WARN, ls="--", lw=1)
    ax.set_xlabel("rise in test log loss when the column is shuffled (dashed: best noise column)")
    ax.grid(axis="x", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def budget(curves: dict, chosen: float, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = {
        "risk x replacement cost": (ACCENT, "-"),
        "risk only": ("#5b8e7d", "-"),
        "rule of thumb": (WARN, "-"),
        "random calls": (GREY, "--"),
    }
    for name, c in curves.items():
        col, ls = styles.get(name, (INK, ":"))
        ax.plot(c.fraction * 100, c.net_per_1000 / 1000, color=col, ls=ls, label=name)
    ax.axvline(chosen * 100, color=INK, lw=0.8, ls=":")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("share of staff called (%)")
    ax.set_ylabel("net value per 1,000 employees (USD thousands)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def survival(table, km: dict, path: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = [f"{r.model}\n{r.features.replace(' tenure counters', '')}" for r in table.itertuples()]
    a.bar(range(len(table)), table.c_index, color=[ACCENT if "without" in f else GREY for f in table.features])
    a.set_xticks(range(len(table)), labels, fontsize=7)
    a.set_ylim(0.5, max(0.8, table.c_index.max() + 0.03))
    a.set_ylabel("concordance on the test fold")
    a.set_title("grey: model also sees the tenure counters", fontsize=9)
    _style(a)
    for name, (t, s) in km.items():
        b.step(t, s, where="post", label=name)
    b.set_xlim(0, 25)
    b.set_xlabel("years at the company")
    b.set_ylabel("share still employed")
    b.legend(frameon=False)
    _style(b)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
