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


def color_style_for(i: int):
    styles = [
        ("red", "dashed", "red"),
        ("green!60!black", "dotted", "green!60!black"),
        ("blue", "dashdotted", "blue"),
        ("orange", "loosely dashed", "orange"),
        ("purple", "solid", "purple"),
        ("black", "dash pattern=on 6pt off 2pt on 1pt off 2pt", "black!20"),
        ("black", "dash pattern=on 1pt off 1pt", "black!20"),
        ("black", "dash pattern=on 8pt off 2pt", "black!20"),
        ("black", "dash pattern=on 2pt off 2pt on 6pt off 2pt", "black!20"),
    ]
    return styles[i % len(styles)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig3-input", default="results/fig3.pkl")
    parser.add_argument("--fig1-input", default="results/fig1.pkl")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--width", default="12cm")
    parser.add_argument("--height", default="8cm")
    parser.add_argument("--y-margin-frac", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.fig3_input, "rb") as f:
        curves, std_curves = pickle.load(f)

    with open(args.fig1_input, "rb") as f:
        test_losses, _ = pickle.load(f)

    if not curves or not std_curves:
        raise ValueError("fig3 results are empty")
    if len(curves) != len(std_curves):
        raise ValueError("curves and std_curves must have the same number of entries")

    num_points = len(curves[0])
    if num_points < 2:
        raise ValueError("each curve must have at least two points")
    for i, (curve, std_curve) in enumerate(zip(curves, std_curves)):
        if len(curve) != num_points or len(std_curve) != num_points:
            raise ValueError(f"inconsistent curve length at index {i}")

    n_empirical = len(test_losses)
    if n_empirical < 2:
        raise ValueError("Need at least two empirical points from fig1 results")

    xs = [i / (num_points - 1) for i in range(num_points)]
    wts = [i / (n_empirical - 1) for i in range(n_empirical)]

    ys = [as_real(y) for y in test_losses]
    for curve, std_curve in zip(curves, std_curves):
        for y, s in zip(curve, std_curve):
            yr = as_real(y)
            sr = as_real(s)
            ys.append(yr - sr)
            ys.append(yr + sr)

    ymin = min(ys)
    ymax = max(ys)
    if math.isclose(ymin, ymax):
        pad = 1e-3
    else:
        pad = args.y_margin_frac * (ymax - ymin)
    ymin -= pad
    ymax += pad

    degree = len(curves) - 1
    if degree < 1:
        raise ValueError("Expected at least one non-constant Taylor curve")

    lines = []
    lines.append("% !TEX root = main.tex")
    lines.append("% Requires in preamble: \\usepgfplotslibrary{fillbetween}")
    lines.append(r"\begin{tikzpicture}")
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
        color, style, fill = color_style_for(cap - 1)
        curve = curves[cap]
        std_curve = std_curves[cap]
        upper_name = f"upper{cap}"
        lower_name = f"lower{cap}"

        lines.append(rf"    \addplot[name path={upper_name}, draw=none, forget plot] coordinates {{")
        for x, y, s in zip(xs, curve, std_curve):
            lines.append(f"        ({x:.17g}, {as_real(y) + as_real(s):.17g})")
        lines.append(r"    };")

        lines.append(rf"    \addplot[name path={lower_name}, draw=none, forget plot] coordinates {{")
        for x, y, s in zip(xs, curve, std_curve):
            lines.append(f"        ({x:.17g}, {as_real(y) - as_real(s):.17g})")
        lines.append(r"    };")

        lines.append(
            rf"    \addplot[{fill}, fill opacity=0.2, draw=none, forget plot] "
            rf"fill between[of={upper_name} and {lower_name}];"
        )

        lines.append(r"    \addplot[")
        lines.append(f"        {color},")
        lines.append(r"        thick,")
        lines.append(f"        {style},")
        lines.append(r"    ] coordinates {")
        for x, y in zip(xs, curve):
            lines.append(f"        ({x:.17g}, {as_real(y):.17g})")
        lines.append(r"    };")
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
