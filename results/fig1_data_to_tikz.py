#!/usr/bin/env python3
import argparse
import math
import pickle
from pathlib import Path


def as_real(x, tol=1e-12):
    z = complex(x)
    if abs(z.imag) > tol:
        raise ValueError(f"Expected near-real value, got {z}")
    return float(z.real)


def outward_int_bounds(values, margin_frac=0.03):
    ymin = min(values)
    ymax = max(values)
    if math.isclose(ymin, ymax):
        pad = 1.0
    else:
        pad = margin_frac * (ymax - ymin)
    return math.floor(ymin - pad), math.ceil(ymax + pad)


def make_xticks(degree: int, step: int) -> str:
    if degree <= step:
        ticks = list(range(1, degree + 1))
    else:
        ticks = [1] + list(range(step, degree + 1, step))
        if ticks[-1] != degree:
            ticks.append(degree)
    return "{" + ",".join(str(t) for t in ticks) + "}"


def color_for(i: int) -> str:
    palette = [
        "blue",
        "red",
        "green!60!black",
        "orange",
        "purple",
        "cyan!60!black",
        "black",
        "magenta",
        "brown!80!black",
        "teal!70!black",
    ]
    return palette[i % len(palette)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="data/fig1/results.pkl")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--width", default="12cm")
    parser.add_argument("--height", default="8cm")
    parser.add_argument("--xtick-step", type=int, default=10)
    parser.add_argument("--all-black", action="store_true")
    parser.add_argument("--semithick", default="semithick")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        estimates = pickle.load(f)

    if not estimates:
        raise ValueError("results.pkl contains no curves")

    lengths = [len(est) for est in estimates]
    if min(lengths) < 2:
        raise ValueError("Each estimate must contain entries for r=0 and at least one positive order")

    degree = min(lengths) - 1
    rs = list(range(1, degree + 1))
    ys = [as_real(est[r]) for est in estimates for r in rs]
    ymin, ymax = outward_int_bounds(ys)

    lines = []
    lines.append("% !TEX root = main.tex")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"    \begin{axis}[")
    lines.append(f"        width={args.width},")
    lines.append(f"        height={args.height},")
    lines.append(f"        xmin=1, xmax={degree},")
    lines.append(f"        ymin={ymin}, ymax={ymax},")
    lines.append(r"        ylabel={$\log\left(\frac{\sqrt{n[r]}}{r!} \abs{f^{(r)}(\bzero)[\bpsi^{\otimes r}]}\right)$},")
    lines.append(f"        xlabel={{order $r$}}, xtick={make_xticks(degree, args.xtick_step)},")
    lines.append(r"        ylabel style={yshift=-4pt},")
    lines.append(r"        grid=both,")
    lines.append(r"        major grid style={draw=gray!30},")
    lines.append(r"        minor grid style={draw=gray!15},")
    lines.append(r"        tick align=outside,")
    lines.append(r"        line cap=round,")
    lines.append(r"        line join=round,")
    lines.append(r"    ]")
    lines.append("")

    for i, est in enumerate(estimates):
        color = "black" if args.all_black else color_for(i)
        lines.append(rf"    \addplot[{color}, {args.semithick}] coordinates {{")
        for r in rs:
            lines.append(f"        ({r}, {as_real(est[r]):.17g})")
        lines.append(r"    };")
        lines.append("")

    lines.append(r"    \end{axis}")
    lines.append(r"\end{tikzpicture}")

    tex = "\n".join(lines)

    if args.output is None:
        print(tex)
    else:
        Path(args.output).write_text(tex)


if __name__ == "__main__":
    main()
