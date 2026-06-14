"""Visualize the C51 Bellman projection used by FastSAC.

Run from the repository root:

    uv run python tutorial/visualize_c51_projection.py

The script uses the 5-atom example from ``fastsac_bellman_projection.md`` and
produces a four-panel figure, a geometric intuition figure, and a numeric
projection table.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ProjectionStep:
    source_atom: float
    source_prob: float
    shifted_atom: float
    b: float
    lower: int
    upper: int
    lower_weight: float
    upper_weight: float


def softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max()
    exp_logits = np.exp(centered)
    return exp_logits / exp_logits.sum()


def project_distribution(
    atoms: np.ndarray,
    probs_next: np.ndarray,
    shifted_atoms: np.ndarray,
) -> tuple[np.ndarray, list[ProjectionStep]]:
    delta = float(atoms[1] - atoms[0])
    v_min = float(atoms[0])
    num_atoms = len(atoms)

    projected = np.zeros(num_atoms)
    steps: list[ProjectionStep] = []

    for atom, prob, shifted in zip(atoms, probs_next, shifted_atoms):
        b = float((shifted - v_min) / delta)
        lower = int(np.floor(b))
        upper = int(np.ceil(b))
        lower = max(0, min(lower, num_atoms - 1))
        upper = max(0, min(upper, num_atoms - 1))

        if lower == upper:
            lower_weight = 1.0
            upper_weight = 0.0
            projected[lower] += prob
        else:
            lower_weight = float(upper - b)
            upper_weight = float(b - lower)
            projected[lower] += prob * lower_weight
            projected[upper] += prob * upper_weight

        steps.append(
            ProjectionStep(
                source_atom=float(atom),
                source_prob=float(prob),
                shifted_atom=float(shifted),
                b=b,
                lower=lower,
                upper=upper,
                lower_weight=lower_weight,
                upper_weight=upper_weight,
            )
        )

    return projected, steps


def print_projection_table(
    atoms: np.ndarray,
    probs_next: np.ndarray,
    unclipped_shifted_atoms: np.ndarray,
    shifted_atoms: np.ndarray,
    projected: np.ndarray,
    steps: list[ProjectionStep],
    r_adj: float,
    gamma: float,
) -> None:
    source_mean = float(np.sum(probs_next * atoms))
    unclipped_shifted_mean = float(np.sum(probs_next * unclipped_shifted_atoms))
    shifted_mean = float(np.sum(probs_next * shifted_atoms))
    projected_mean = float(np.sum(projected * atoms))

    print(f"adjusted reward r_adj = {r_adj:.4f}")
    print(f"source atoms          = {np.round(atoms, 4)}")
    print(f"next distribution     = {np.round(probs_next, 4)}")
    print(f"shifted atoms raw     = {np.round(unclipped_shifted_atoms, 4)}")
    print(f"shifted atoms clipped = {np.round(shifted_atoms, 4)}")
    print(f"projected target      = {np.round(projected, 4)}")
    print()
    print("per-atom projection:")
    print(
        "  z_k    p_k    shifted    b       left -> mass        right -> mass"
    )
    for step in steps:
        left_mass = step.source_prob * step.lower_weight
        right_mass = step.source_prob * step.upper_weight
        left_text = f"{atoms[step.lower]:>5.1f}: {left_mass:>6.4f}"
        right_text = (
            f"{atoms[step.upper]:>5.1f}: {right_mass:>6.4f}"
            if step.upper_weight > 0
            else "     -: 0.0000"
        )
        print(
            f"{step.source_atom:>6.1f} "
            f"{step.source_prob:>6.4f} "
            f"{step.shifted_atom:>9.4f} "
            f"{step.b:>7.4f} "
            f"{left_text}     {right_text}"
        )
    print()
    print(f"mean before Bellman   = {source_mean:.4f}")
    print(f"mean after Bellman    = {unclipped_shifted_mean:.4f}")
    print(f"mean after clamp      = {shifted_mean:.4f}")
    print(f"mean after projection = {projected_mean:.4f}")
    print(f"r_adj + gamma * mean  = {r_adj + gamma * source_mean:.4f}")
    print(f"probability sum       = {projected.sum():.4f}")


def annotate_bars(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray) -> None:
    for x, y in zip(xs, ys):
        if y > 0.005:
            ax.text(x, y + 0.015, f"{y:.2f}", ha="center", fontsize=10, fontweight="bold")


def draw_projection_figure(
    atoms: np.ndarray,
    probs_next: np.ndarray,
    shifted_atoms: np.ndarray,
    projected: np.ndarray,
    steps: list[ProjectionStep],
    current_probs: np.ndarray,
    output_path: Path,
    r_adj: float,
    gamma: float,
) -> None:
    delta = float(atoms[1] - atoms[0])
    bar_width = delta * 0.58

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flat

    source_mean = float(np.sum(probs_next * atoms))
    shifted_mean = float(np.sum(probs_next * shifted_atoms))
    target_mean = float(np.sum(projected * atoms))
    current_mean = float(np.sum(current_probs * atoms))

    blue = plt.cm.Blues(0.70)
    orange = plt.cm.Oranges(0.72)
    green = plt.cm.Greens(0.72)

    ax1.bar(atoms, probs_next, width=bar_width, color=blue, edgecolor="navy", alpha=0.88)
    annotate_bars(ax1, atoms, probs_next)
    ax1.axvline(source_mean, color="crimson", linestyle="--", linewidth=1.5)
    ax1.text(source_mean, 0.50, f"E[Q]={source_mean:.2f}", color="crimson", ha="center")
    ax1.set_title("Step 1: target critic logits -> softmax distribution", fontsize=13)
    ax1.set_xlabel("fixed support atom z_k")
    ax1.set_ylabel("probability")

    ax2.vlines(atoms, 0, 0.50, color="0.80", linewidth=1.0, label="fixed grid")
    ax2.bar(shifted_atoms, probs_next, width=bar_width, color=orange, edgecolor="darkorange", alpha=0.88)
    annotate_bars(ax2, shifted_atoms, probs_next)
    for old, new, prob in zip(atoms, shifted_atoms, probs_next):
        if abs(old - new) > 0.25:
            ax2.annotate(
                "",
                xy=(new, prob),
                xytext=(old, prob),
                arrowprops={"arrowstyle": "->", "color": "crimson", "lw": 1.4, "alpha": 0.65},
            )
    ax2.axvline(shifted_mean, color="crimson", linestyle="--", linewidth=1.5)
    ax2.set_title(f"Step 2: Bellman shift, z' = r_adj + gamma z, r_adj={r_adj:.4f}", fontsize=13)
    ax2.set_xlabel("Q value")
    ax2.set_ylabel("same probability, moved coordinate")
    ax2.legend(fontsize=9)

    bottoms = np.zeros_like(projected)
    segment_colors = plt.cm.viridis(np.linspace(0.18, 0.82, len(steps)))
    for color, step in zip(segment_colors, steps):
        lower_mass = step.source_prob * step.lower_weight
        upper_mass = step.source_prob * step.upper_weight
        if lower_mass > 0:
            ax3.bar(
                atoms[step.lower],
                lower_mass,
                width=bar_width,
                bottom=bottoms[step.lower],
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            bottoms[step.lower] += lower_mass
        if upper_mass > 0:
            ax3.bar(
                atoms[step.upper],
                upper_mass,
                width=bar_width,
                bottom=bottoms[step.upper],
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            bottoms[step.upper] += upper_mass
        if step.upper_weight > 0:
            y = max(step.source_prob + 0.05, 0.08)
            ax3.plot(
                [step.shifted_atom, atoms[step.lower]],
                [y, projected[step.lower] + 0.03],
                color=color,
                alpha=0.40,
                linewidth=1.0,
            )
            ax3.plot(
                [step.shifted_atom, atoms[step.upper]],
                [y, projected[step.upper] + 0.03],
                color=color,
                alpha=0.40,
                linewidth=1.0,
            )
    annotate_bars(ax3, atoms, projected)
    ax3.vlines(shifted_atoms, 0.52, 0.56, color="darkorange", linewidth=2.0, label="shifted atoms")
    ax3.set_title("Step 3: projection, split each shifted mass to neighboring grid atoms", fontsize=13)
    ax3.set_xlabel("fixed support atom z_j")
    ax3.set_ylabel("projected probability")
    ax3.legend(fontsize=9)

    ax4.bar(
        atoms - bar_width * 0.30,
        projected,
        width=bar_width * 0.55,
        color=green,
        edgecolor="darkgreen",
        alpha=0.88,
        label=f"target, mean={target_mean:.2f}",
    )
    ax4.bar(
        atoms + bar_width * 0.30,
        current_probs,
        width=bar_width * 0.55,
        color="salmon",
        edgecolor="firebrick",
        alpha=0.72,
        label=f"current Q, mean={current_mean:.2f}",
    )
    ax4.set_title("Step 4: critic loss compares target distribution with current distribution", fontsize=13)
    ax4.set_xlabel("fixed support atom z_j")
    ax4.set_ylabel("probability")
    ax4.legend(fontsize=9)

    for ax in axes.flat:
        ax.set_xlim(atoms[0] - delta * 0.75, atoms[-1] + delta * 0.75)
        ax.set_ylim(0, 0.58)
        ax.set_xticks(atoms)
        ax.grid(axis="y", alpha=0.28)

    fig.suptitle(
        "FastSAC C51 Bellman Projection: probabilities stay, coordinates move, then mass is projected back",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_mass_cloud(
    ax: plt.Axes,
    xs: np.ndarray,
    probs: np.ndarray,
    *,
    y: float,
    color: str,
    label: str,
) -> None:
    sizes = 2600 * probs + 130
    ax.scatter(xs, np.full_like(xs, y), s=sizes, color=color, alpha=0.72, edgecolor="black", zorder=3)
    for x, p in zip(xs, probs):
        ax.text(x, y + 0.16, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(xs[0] - 5.5, y, label, ha="right", va="center", fontsize=11, fontweight="bold")


def draw_center_of_mass(ax: plt.Axes, x: float, *, y0: float, y1: float, label: str) -> None:
    ax.vlines(x, y0, y1, color="crimson", linestyle="--", linewidth=2.0, zorder=2)
    ax.text(x, y1 + 0.08, label, color="crimson", ha="center", va="bottom", fontsize=10)


def draw_geometry_figure(
    atoms: np.ndarray,
    probs_next: np.ndarray,
    unclipped_shifted_atoms: np.ndarray,
    shifted_atoms: np.ndarray,
    projected: np.ndarray,
    steps: list[ProjectionStep],
    output_path: Path,
) -> None:
    source_mean = float(np.sum(probs_next * atoms))
    bellman_mean = float(np.sum(probs_next * unclipped_shifted_atoms))
    clipped_mean = float(np.sum(probs_next * shifted_atoms))
    projected_mean = float(np.sum(projected * atoms))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), height_ratios=[1.0, 1.05])
    ax1, ax2 = axes

    ax1.axhline(1.0, color="0.70", linewidth=1.1)
    ax1.axhline(0.0, color="0.70", linewidth=1.1)
    draw_mass_cloud(ax1, atoms, probs_next, y=1.0, color="#4c98c9", label="before")
    draw_mass_cloud(
        ax1,
        unclipped_shifted_atoms,
        probs_next,
        y=0.0,
        color="#f07f24",
        label="after z' = r_adj + gamma z",
    )
    for old, new in zip(atoms, unclipped_shifted_atoms):
        ax1.annotate(
            "",
            xy=(new, 0.08),
            xytext=(old, 0.92),
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 1.2, "alpha": 0.60},
        )
        ax1.text((old + new) / 2.0, 0.52, f"{old:.0f}->{new:.1f}", ha="center", fontsize=8)

    draw_center_of_mass(ax1, source_mean, y0=0.78, y1=1.28, label=f"old center={source_mean:.2f}")
    draw_center_of_mass(
        ax1,
        bellman_mean,
        y0=-0.25,
        y1=0.28,
        label=f"new center={bellman_mean:.2f}",
    )
    ax1.set_title("Bellman transform is an affine move of the whole probability mass cloud")
    ax1.set_yticks([])
    ax1.set_xlim(-27, 27)
    ax1.set_ylim(-0.5, 1.55)
    ax1.set_xlabel("Q coordinate")
    ax1.grid(axis="x", alpha=0.25)

    ax2.axhline(1.0, color="0.70", linewidth=1.1)
    ax2.axhline(0.0, color="0.70", linewidth=1.1)
    ax2.vlines(atoms, -0.35, 1.28, color="0.82", linewidth=1.0)
    draw_mass_cloud(ax2, shifted_atoms, probs_next, y=1.0, color="#f07f24", label="shifted/clipped")
    draw_mass_cloud(ax2, atoms, projected, y=0.0, color="#45a365", label="projected to grid")

    for step in steps:
        lower_mass = step.source_prob * step.lower_weight
        upper_mass = step.source_prob * step.upper_weight
        if lower_mass > 0:
            ax2.annotate(
                "",
                xy=(atoms[step.lower], 0.13),
                xytext=(step.shifted_atom, 0.88),
                arrowprops={"arrowstyle": "->", "color": "#2b8cbe", "lw": 1.1, "alpha": 0.55},
            )
        if upper_mass > 0:
            ax2.annotate(
                "",
                xy=(atoms[step.upper], 0.13),
                xytext=(step.shifted_atom, 0.88),
                arrowprops={"arrowstyle": "->", "color": "#2b8cbe", "lw": 1.1, "alpha": 0.55},
            )

    example = steps[0]
    ax2.plot(
        [example.shifted_atom, example.shifted_atom],
        [0.72, -0.42],
        color="crimson",
        linestyle=":",
        linewidth=1.4,
    )
    ax2.text(
        example.shifted_atom,
        -0.48,
        "each split keeps local center of mass",
        ha="center",
        va="top",
        color="crimson",
        fontsize=10,
    )

    draw_center_of_mass(ax2, clipped_mean, y0=0.74, y1=1.28, label=f"clipped center={clipped_mean:.4f}")
    draw_center_of_mass(
        ax2,
        projected_mean,
        y0=-0.32,
        y1=0.28,
        label=f"projected center={projected_mean:.4f}",
    )
    ax2.text(
        23.5,
        1.22,
        f"unclipped center was {bellman_mean:.4f}\nclamp loss = {bellman_mean - clipped_mean:.4f}",
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75"},
    )
    ax2.set_title("C51 projection redistributes mass to grid atoms while preserving mass and center")
    ax2.set_yticks([])
    ax2.set_xlim(-27, 27)
    ax2.set_ylim(-0.7, 1.55)
    ax2.set_xlabel("Q coordinate")
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Geometric view: Bellman moves the center; projection preserves the center after clipping",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    image_dir = Path(__file__).with_name("imagefastsac")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=image_dir / "c51_bellman_projection.png",
        help="where to write the visualization PNG",
    )
    parser.add_argument(
        "--geometry-output",
        type=Path,
        default=image_dir / "c51_bellman_geometry.png",
        help="where to write the geometric intuition PNG",
    )
    parser.add_argument("--show", action="store_true", help="show the figure interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Same small example as fastsac_bellman_projection.md.
    atoms = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    tutorial_probs = np.array([0.10, 0.13, 0.40, 0.20, 0.17])
    logits = np.log(tutorial_probs)
    current_logits = np.array([-0.1, 0.3, 0.8, 0.6, 0.2])

    reward = 2.0
    gamma = 0.97
    alpha = 0.1
    done = 0.0
    next_log_prob = -0.5

    probs_next = softmax(logits)
    current_probs = softmax(current_logits)
    r_adj = reward - gamma * (1.0 - done) * alpha * next_log_prob
    unclipped_shifted_atoms = r_adj + gamma * (1.0 - done) * atoms
    shifted_atoms = np.clip(unclipped_shifted_atoms, atoms[0], atoms[-1])
    projected, steps = project_distribution(atoms, probs_next, shifted_atoms)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_projection_figure(
        atoms=atoms,
        probs_next=probs_next,
        shifted_atoms=shifted_atoms,
        projected=projected,
        steps=steps,
        current_probs=current_probs,
        output_path=output_path,
        r_adj=r_adj,
        gamma=gamma,
    )
    geometry_output_path = args.geometry_output.resolve()
    geometry_output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_geometry_figure(
        atoms=atoms,
        probs_next=probs_next,
        unclipped_shifted_atoms=unclipped_shifted_atoms,
        shifted_atoms=shifted_atoms,
        projected=projected,
        steps=steps,
        output_path=geometry_output_path,
    )
    print_projection_table(
        atoms,
        probs_next,
        unclipped_shifted_atoms,
        shifted_atoms,
        projected,
        steps,
        r_adj,
        gamma,
    )
    print(f"\nwrote figure: {output_path}")
    print(f"wrote geometry figure: {geometry_output_path}")

    if args.show:
        image = plt.imread(output_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
