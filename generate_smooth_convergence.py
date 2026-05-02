from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

N_VALUES = [32, 64, 128]
SEEDS = list(range(1, 11))
THRESHOLDS = [1e-2, 5e-3, 1e-3, 5e-4]
RANDOM_MAX_BLOCKS = 400


def chain_f(x: np.ndarray) -> float:
    return float(0.5 * x[0] ** 2 + 0.5 * np.sum(np.diff(x) ** 2) + 0.5 * x[-1] ** 2 - x[0])


def chain_grad(x: np.ndarray) -> np.ndarray:
    n = x.size
    g = np.empty_like(x)
    if n == 1:
        g[0] = 2.0 * x[0] - 1.0
        return g
    g[0] = 2.0 * x[0] - x[1] - 1.0
    g[1:-1] = 2.0 * x[1:-1] - x[:-2] - x[2:]
    g[-1] = 2.0 * x[-1] - x[-2]
    return g


def chain_data(n: int) -> dict[str, float | np.ndarray]:
    diag = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    a = np.diag(diag) + np.diag(off, k=1) + np.diag(off, k=-1)
    e1 = np.zeros(n)
    e1[0] = 1.0
    x_star = np.linalg.solve(a, e1)
    f_star = -0.5 * float(e1 @ x_star)
    l1 = 4.0
    r2 = float(x_star @ x_star)
    scale = 0.5 * l1 * r2
    return {"n": n, "x_star": x_star, "f_star": f_star, "l1": l1, "scale": scale}


def relative_gap(value: float, data: dict[str, float | np.ndarray]) -> float:
    return max((value - float(data["f_star"])) / float(data["scale"]), 1e-14)


def theory_mu(data: dict[str, float | np.ndarray]) -> float:
    n = int(data["n"])
    eps_target = 1e-6 * float(data["scale"])
    return max((5.0 / (3.0 * (n + 4))) * math.sqrt(eps_target / (2.0 * float(data["l1"]))), 1e-8)


def add_row(rows: list[dict[str, float | int | str]], algorithm: str, data, seed: int,
            iteration: int, function_calls: int, x: np.ndarray) -> None:
    n = int(data["n"])
    value = chain_f(x)
    rows.append({
        "algorithm": algorithm,
        "n": n,
        "seed": seed,
        "iteration": iteration,
        "block": iteration / n,
        "function_calls": function_calls,
        "f_value": value,
        "relative_gap": relative_gap(value, data),
    })


def run_gm(data, rows: list[dict[str, float | int | str]]) -> None:
    n = int(data["n"])
    x = np.zeros(n)
    add_row(rows, "GM", data, 0, 0, 0, x)
    for k in range(1, 701):
        x -= (1.0 / float(data["l1"])) * chain_grad(x)
        if k % 5 == 0 or k == 700:
            add_row(rows, "GM", data, 0, k, k, x)


def run_rg_mu(data, seed: int, rows: list[dict[str, float | int | str]]) -> None:
    n = int(data["n"])
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    h = 1.0 / (4.0 * (n + 4) * float(data["l1"]))
    mu = theory_mu(data)
    max_iter = RANDOM_MAX_BLOCKS * n
    calls = 0
    add_row(rows, "RG_mu", data, seed, 0, calls, x)
    for k in range(1, max_iter + 1):
        u = rng.normal(size=n)
        fx = chain_f(x)
        fplus = chain_f(x + mu * u)
        g = ((fplus - fx) / mu) * u
        x -= h * g
        calls += 2
        if k % n == 0 or k == max_iter:
            add_row(rows, "RG_mu", data, seed, k, calls, x)


def run_rg0(data, seed: int, rows: list[dict[str, float | int | str]]) -> None:
    n = int(data["n"])
    rng = np.random.default_rng(seed + 100_000)
    x = np.zeros(n)
    h = 1.0 / (4.0 * (n + 4) * float(data["l1"]))
    max_iter = RANDOM_MAX_BLOCKS * n
    add_row(rows, "RG_0", data, seed, 0, 0, x)
    for k in range(1, max_iter + 1):
        u = rng.normal(size=n)
        g = float(chain_grad(x) @ u) * u
        x -= h * g
        if k % n == 0 or k == max_iter:
            add_row(rows, "RG_0", data, seed, k, 0, x)


def write_csv(path: Path, rows: list[dict[str, float | int | str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["algorithm"], int(row["n"]), int(row["seed"]))].append(row)

    summary = []
    for algorithm in ["GM", "RG_mu", "RG_0"]:
        for n in N_VALUES:
            seeds = [0] if algorithm == "GM" else SEEDS
            for threshold in THRESHOLDS:
                hits = []
                for seed in seeds:
                    series = sorted(by_key[(algorithm, n, seed)], key=lambda r: int(r["iteration"]))
                    hit = next((r for r in series if float(r["relative_gap"]) <= threshold), None)
                    if hit is not None:
                        hits.append(hit)
                blocks = np.array([float(h["block"]) for h in hits], dtype=float)
                calls = np.array([float(h["function_calls"]) for h in hits], dtype=float)
                summary.append({
                    "algorithm": algorithm,
                    "n": n,
                    "threshold": threshold,
                    "mean_blocks": float(np.mean(blocks)) if hits else math.inf,
                    "std_blocks": float(np.std(blocks, ddof=1)) if len(hits) > 1 else 0.0,
                    "mean_function_calls": float(np.mean(calls)) if hits else math.inf,
                    "success_rate": len(hits) / len(seeds),
                })
    return summary


def write_table(summary: list[dict[str, float | int | str]]) -> None:
    rows = [
        r for r in summary
        if int(r["n"]) == 64 and float(r["threshold"]) in {1e-2, 1e-3}
    ]
    with (TABLES / "smooth_summary_table.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lllll}\n")
        f.write("\\toprule\n")
        f.write("Algorithm & $n$ & Threshold & Mean blocks & Success rate \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            alg = {"GM": "GM", "RG_mu": "$RG_\\mu$", "RG_0": "$RG_0$"}[str(r["algorithm"])]
            blocks = "--" if math.isinf(float(r["mean_blocks"])) else f"{float(r['mean_blocks']):.1f}"
            rate = f"{100.0 * float(r['success_rate']):.0f}\\%"
            f.write(f"{alg} & {int(r['n'])} & {float(r['threshold']):.0e} & {blocks} & {rate} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def first_hit(series: list[dict[str, float | int | str]], threshold: float):
    ordered = sorted(series, key=lambda r: int(r["iteration"]))
    return next((r for r in ordered if float(r["relative_gap"]) <= threshold), None)


def write_threshold_table(rows: list[dict[str, float | int | str]], n: int = 64) -> None:
    grouped = defaultdict(list)
    for row in rows:
        if int(row["n"]) == n:
            grouped[(row["algorithm"], int(row["seed"]))].append(row)

    table_rows = []
    for threshold in THRESHOLDS:
        for algorithm in ["GM", "RG_mu", "RG_0"]:
            seeds = [0] if algorithm == "GM" else SEEDS
            hits = []
            for seed in seeds:
                hit = first_hit(grouped[(algorithm, seed)], threshold)
                if hit is not None:
                    hits.append(hit)
            if hits:
                if algorithm == "GM":
                    budgets = [float(h["iteration"]) for h in hits]
                    budget = f"{np.mean(budgets):.0f} iterations"
                    calls = f"{np.mean([float(h['function_calls']) for h in hits]):.0f}"
                else:
                    budgets = [float(h["block"]) for h in hits]
                    budget = f"{np.mean(budgets):.1f} blocks"
                    if algorithm == "RG_mu":
                        calls = f"{np.mean([float(h['function_calls']) for h in hits]):.0f}"
                    else:
                        calls = "directional"
            else:
                budget = "--"
                calls = "--"
            table_rows.append({
                "algorithm": algorithm,
                "n": n,
                "threshold": threshold,
                "budget": budget,
                "function_calls": calls,
                "success_rate": len(hits) / len(seeds),
            })

    write_csv(
        RESULTS / "smooth_threshold_table.csv",
        table_rows,
        ["algorithm", "n", "threshold", "budget", "function_calls", "success_rate"],
    )

    with (TABLES / "smooth_threshold_table.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lllll}\n")
        f.write("\\toprule\n")
        f.write("Algorithm & Threshold & Budget to hit & Function calls & Success \\\\\n")
        f.write("\\midrule\n")
        for r in table_rows:
            alg = {"GM": "GM", "RG_mu": "$RG_\\mu$", "RG_0": "$RG_0$"}[str(r["algorithm"])]
            rate = f"{100.0 * float(r['success_rate']):.0f}\\%"
            f.write(
                f"{alg} & {float(r['threshold']):.0e} & {r['budget']} & "
                f"{r['function_calls']} & {rate} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    with (FIGURES / "fig_smooth_threshold_table.tex").open("w", encoding="utf-8") as f:
        f.write("\\documentclass[border=6pt]{standalone}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{amsmath}\n")
        f.write("\\begin{document}\n")
        f.write("\\small\n")
        f.write("\\input{../tables/smooth_threshold_table.tex}\n")
        f.write("\\end{document}\n")


def write_dimension_scaling_table(rows: list[dict[str, float | int | str]]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], int(row["n"]), int(row["seed"]))].append(row)

    table_rows = []
    threshold = 1e-3
    for n in N_VALUES:
        for algorithm in ["GM", "RG_mu"]:
            seeds = [0] if algorithm == "GM" else SEEDS
            hits = []
            for seed in seeds:
                series = sorted(grouped[(algorithm, n, seed)], key=lambda r: int(r["iteration"]))
                if not series:
                    continue
                hit = next((r for r in series if float(r["relative_gap"]) <= threshold), None)
                if hit is not None:
                    hits.append(hit)
            blocks = np.array([float(h["block"]) for h in hits], dtype=float)
            calls = np.array([float(h["function_calls"]) for h in hits], dtype=float)
            table_rows.append({
                "algorithm": algorithm,
                "n": n,
                "threshold": threshold,
                "mean_blocks": float(np.mean(blocks)) if hits else math.inf,
                "std_blocks": float(np.std(blocks, ddof=1)) if len(hits) > 1 else 0.0,
                "mean_function_calls": float(np.mean(calls)) if hits else math.inf,
                "success_rate": len(hits) / len(seeds),
            })

    write_csv(
        RESULTS / "dimension_scaling_summary.csv",
        table_rows,
        [
            "algorithm",
            "n",
            "threshold",
            "mean_blocks",
            "std_blocks",
            "mean_function_calls",
            "success_rate",
        ],
    )

    with (TABLES / "dimension_scaling_table.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llll}\n")
        f.write("\\toprule\n")
        f.write("Method & $n$ & Normalized budget to $10^{-3}$ & Success \\\\\n")
        f.write("\\midrule\n")
        for r in table_rows:
            alg = {"GM": "GM", "RG_mu": "$RG_\\mu$"}[str(r["algorithm"])]
            blocks = "--" if math.isinf(float(r["mean_blocks"])) else f"{float(r['mean_blocks']):.1f}"
            rate = f"{100.0 * float(r['success_rate']):.0f}\\%"
            f.write(f"{alg} & {int(r['n'])} & {blocks} & {rate} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    with (FIGURES / "fig_dimension_scaling.tex").open("w", encoding="utf-8") as f:
        f.write("\\documentclass[border=6pt]{standalone}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{amsmath}\n")
        f.write("\\begin{document}\n")
        f.write("\\small\n")
        f.write("\\input{../tables/dimension_scaling_table.tex}\n")
        f.write("\\end{document}\n")


def plot(rows: list[dict[str, float | int | str]]) -> None:
    n = 64
    grouped = defaultdict(list)
    for row in rows:
        if int(row["n"]) == n:
            grouped[(row["algorithm"], float(row["block"]))].append(float(row["relative_gap"]))

    plt.rcParams.update({
        "font.size": 10,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
    })
    fig, ax = plt.subplots(figsize=(6.8, 4.25), constrained_layout=True)
    styles = {
        "GM": {"linestyle": "-", "linewidth": 2.2, "label": "GM"},
        "RG_mu": {"linestyle": "--", "linewidth": 2.1, "label": r"RG$_\mu$"},
        "RG_0": {
            "linestyle": "None",
            "linewidth": 0.0,
            "marker": "o",
            "markersize": 4.0,
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markeredgewidth": 1.0,
            "markevery": 10,
            "label": r"diagnostic RG$_0$",
        },
    }
    for algorithm in ["GM", "RG_mu", "RG_0"]:
        xs = sorted(block for alg, block in grouped if alg == algorithm)
        ys = [np.mean(grouped[(algorithm, x)]) for x in xs]
        ax.plot(xs, ys, color="black", **styles[algorithm])
    ax.set_xlim(0, RANDOM_MAX_BLOCKS)
    ax.set_yscale("log")
    ax.set_ylim(3.0e-4, 2.0e-2)
    ax.set_xlabel("budget: random blocks; GM iterations / n")
    ax.set_ylabel("relative objective gap")
    ax.set_title("Smooth quadratic chain, n=64, mean over 10 seeds", color="black", pad=8)
    ax.grid(True, which="both", linestyle=":", color="black", alpha=0.22)
    ax.axhline(1.0e-3, color="black", linewidth=1.0, linestyle=":", alpha=0.65)
    ax.text(305, 1.06e-3, r"$10^{-3}$ target", fontsize=9, ha="left", va="bottom")
    ax.legend(loc="upper right", frameon=True, edgecolor="black", facecolor="white", framealpha=1.0)
    ax.annotate(
        r"GM crosses $10^{-3}$ in less" + "\n" + "than one normalized block",
        xy=(1.02, 1.0e-3),
        xytext=(28, 4.8e-3),
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.0},
        ha="left",
        va="center",
        fontsize=9,
        color="black",
    )
    ax.text(
        92,
        1.65e-3,
        r"diagnostic RG$_0$ tracks RG$_\mu$:" + "\n" + "finite-difference bias is tiny here",
        fontsize=9,
        color="black",
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "black"},
    )
    for spine in ax.spines.values():
        spine.set_color("black")
    fig.savefig(FIGURES / "fig_smooth_convergence.pdf")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for n in N_VALUES:
        data = chain_data(n)
        run_gm(data, rows)
        for seed in SEEDS:
            run_rg_mu(data, seed, rows)
            run_rg0(data, seed, rows)

    run_fields = ["algorithm", "n", "seed", "iteration", "block", "function_calls", "f_value", "relative_gap"]
    write_csv(RESULTS / "smooth_runs.csv", rows, run_fields)

    summary = summarize(rows)
    summary_fields = ["algorithm", "n", "threshold", "mean_blocks", "std_blocks", "mean_function_calls", "success_rate"]
    write_csv(RESULTS / "smooth_summary.csv", summary, summary_fields)
    write_table(summary)
    write_threshold_table(rows, n=64)
    write_dimension_scaling_table(rows)
    plot(rows)


if __name__ == "__main__":
    main()
