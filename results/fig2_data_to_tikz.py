#!/usr/bin/env python3
import argparse
import math
import pickle
from pathlib import Path

import src.polys as polys  # needed so pickle can reconstruct Poly


def as_real(x, tol=1e-12):
    z = complex(x)
    if abs(z.imag) > tol:
        raise ValueError(f"Expected near-real value, got {z}")
    return float(z.real)


def eval_truncated(coeffs, x, cap=None):
    if cap is None:
        cap = len(coeffs) - 1
    out = 0.0
    xp = 1.0
    for c in coeffs[: cap + 1]:
        out += as_real(c) * xp
        xp *= x
    return out


def coeff_name(i: int) -> str:
    # A, B, ..., Z, AA, AB, ...
    s = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(ord('A') + r) + s
    return f"c{s}"


def poly_expr(cap: int) -> str:
    terms = []
    for i in range(cap + 1):
        name = '\\' + coeff_name(i)
        if i == 0:
            terms.append(name)
        elif i == 1:
            terms.append(f"{name}*x")
        else:
            terms.append(f"{name}*x^{i}")
    return " + ".join(terms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="data/fig2/results.pkl")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--width", default="12cm")
    parser.add_argument("--height", default="8cm")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--y-margin-frac", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        test_losses, poly_test_loss = pickle.load(f)

    coeffs = list(poly_test_loss.coeffs)
    degree = len(coeffs) - 1
    n = len(test_losses)
    if n < 2:
        raise ValueError("Need at least two empirical points")

    wts = [i / (n - 1) for i in range(n)]

    ys = [as_real(y) for y in test_losses]
    for cap in range(1, degree + 1):
        ys.extend(eval_truncated(coeffs, j / (args.samples - 1), cap=cap) for j in range(args.samples))

    ymin = min(ys)
    ymax = max(ys)
    if math.isclose(ymin, ymax):
        pad = 1e-3
    else:
        pad = args.y_margin_frac * (ymax - ymin)
    ymin -= pad
    ymax += pad

    curve_specs = [
        ("red", "dashed"),
        ("green!60!black", "dotted"),
        ("blue", "dashdotted"),
        ("orange", "loosely dashed"),
        ("purple", "solid"),
        ("black", "dash pattern=on 6pt off 2pt on 1pt off 2pt"),
        ("black", "dash pattern=on 1pt off 1pt"),
        ("black", "dash pattern=on 8pt off 2pt"),
        ("black", "dash pattern=on 2pt off 2pt on 6pt off 2pt"),
    ]

    lines = []
    lines.append("% !TEX root = main.tex")
    lines.append(r"\begin{tikzpicture}")
    for i, c in enumerate(coeffs):
        lines.append(rf"    \pgfmathsetmacro{{\{coeff_name(i)}}}{{{as_real(c):.17g}}}")
    lines.append("")
    lines.append(r"    \begin{axis}[")
    lines.append(f"        width={args.width},")
    lines.append(f"        height={args.height},")
    lines.append("        xmin=0, xmax=1,")
    lines.append(f"        ymin={ymin:.17g}, ymax={ymax:.17g},")
    lines.append(r"        xlabel={downweight $w$},")
    lines.append(r"        ylabel={$f(w \bone_D)$},")
    lines.append(r"        grid=both,")
    lines.append(r"        major grid style={draw=gray!30},")
    lines.append(r"        minor grid style={draw=gray!15},")
    lines.append(r"        tick align=outside,")
    lines.append(r"        legend pos=north west,")
    lines.append(r"        legend cell align=left,")
    lines.append(r"        legend style={draw=none, fill=white, font=\small},")
    lines.append(r"        line cap=round,")
    lines.append(r"        line join=round,")
    lines.append(r"    ]")
    lines.append("")

    lines.append(r"    \addplot[")
    lines.append(r"        black,")
    lines.append(r"        thick,")
    lines.append(r"        mark=*,")
    lines.append(r"        only marks,")
    lines.append(r"        mark size=1.6pt,")
    lines.append(r"    ] coordinates {")
    for x, y in zip(wts, test_losses):
        lines.append(f"        ({x:.17g}, {as_real(y):.17g})")
    lines.append(r"    };")
    lines.append(r"    \addlegendentry{empirical}")
    lines.append("")

    for cap in range(1, degree + 1):
        color, style = curve_specs[(cap - 1) % len(curve_specs)]
        lines.append(r"    \addplot[")
        lines.append(f"        {color},")
        lines.append(r"        thick,")
        lines.append(f"        {style},")
        lines.append(r"        domain=0:1,")
        lines.append(f"        samples={args.samples},")
        lines.append(r"    ] {" + poly_expr(cap) + r"};")
        lines.append(rf"    \addlegendentry{{degree {cap}}}")
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
