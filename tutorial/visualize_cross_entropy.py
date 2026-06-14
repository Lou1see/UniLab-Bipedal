"""Visualize why cross-entropy pushes current probabilities toward target.

Run from the repository root:

    uv run python tutorial/visualize_cross_entropy.py

The figures are written under ``tutorial/imagefastsac``.
"""

# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


EPS = 1e-9


def softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max()
    exp_logits = np.exp(centered)
    return exp_logits / exp_logits.sum()


def cross_entropy(target: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, EPS, 1.0)
    return float(-np.sum(target * np.log(probs)))


def mse_loss(target: np.ndarray, probs: np.ndarray) -> float:
    return float(np.sum((probs - target) ** 2))


def ce_curve_2class(target_p1: float, grid: np.ndarray) -> np.ndarray:
    target = np.array([target_p1, 1.0 - target_p1])
    probs = np.stack([grid, 1.0 - grid], axis=-1)
    probs = np.clip(probs, EPS, 1.0)
    return -np.sum(target * np.log(probs), axis=-1)


def mse_curve_2class(target_p1: float, grid: np.ndarray) -> np.ndarray:
    target = np.array([target_p1, 1.0 - target_p1])
    probs = np.stack([grid, 1.0 - grid], axis=-1)
    return np.sum((probs - target) ** 2, axis=-1)


def setup_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_loss_landscape(ax: plt.Axes) -> None:
    grid = np.linspace(0.001, 0.999, 600)
    targets = [0.1, 0.5, 0.9]
    colors = ["#3567a8", "#555555", "#bd3c37"]

    for target_p1, color in zip(targets, colors):
        ax.plot(
            grid,
            ce_curve_2class(target_p1, grid),
            color=color,
            linewidth=2.2,
            label=f"CE target=[{target_p1:.1f},{1-target_p1:.1f}]",
        )
        ax.axvline(target_p1, color=color, linestyle=":", linewidth=1.5)

    ax.set_title("1. Cross-entropy loss is lowest when current = target", fontsize=12)
    ax.set_xlabel("current probability s1")
    ax.set_ylabel("cross entropy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5.2)
    ax.legend(fontsize=8, loc="upper center", ncols=1)
    setup_axis(ax)


def draw_ce_vs_mse(ax: plt.Axes) -> None:
    grid = np.linspace(0.001, 0.999, 600)
    target_p1 = 0.1

    ce = ce_curve_2class(target_p1, grid)
    mse = mse_curve_2class(target_p1, grid)

    ax.plot(grid, ce, color="#bd3c37", linewidth=2.4, label="cross entropy")
    ax.plot(grid, mse, color="#3567a8", linewidth=2.4, linestyle="--", label="MSE")
    ax.axvline(target_p1, color="black", linestyle=":", linewidth=1.5)
    ax.text(target_p1 + 0.02, 3.9, "target s1=0.1", fontsize=9)
    ax.text(
        0.62,
        3.15,
        "CE becomes very large\nwhen target mass gets\nnear-zero current prob",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75"},
    )
    ax.set_title("2. CE punishes missing target mass more sharply than MSE", fontsize=12)
    ax.set_xlabel("current probability s1")
    ax.set_ylabel("loss")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5.2)
    ax.legend(fontsize=9)
    setup_axis(ax)


def draw_gradient_lines(ax: plt.Axes) -> None:
    grid = np.linspace(0.0, 1.0, 250)
    targets = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(targets)))

    ax.axhline(0, color="black", linewidth=1.1)
    for target_p1, color in zip(targets, colors):
        grad = grid - target_p1
        ax.plot(grid, grad, color=color, linewidth=2.0, label=f"target s1={target_p1:.1f}")
        ax.axvline(target_p1, color=color, linestyle=":", linewidth=1.0, alpha=0.8)

    ax.fill_between(grid, 0, 1, color="#bd3c37", alpha=0.08)
    ax.fill_between(grid, -1, 0, color="#3567a8", alpha=0.08)
    ax.text(0.04, 0.72, "gradient > 0\nlogit step goes down", color="#8a1f1b", fontsize=9)
    ax.text(0.58, -0.78, "gradient < 0\nlogit step goes up", color="#214f86", fontsize=9)
    ax.set_title("3. With softmax + CE, gradient = current - target", fontsize=12)
    ax.set_xlabel("current probability s1")
    ax.set_ylabel("dL / dz1")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=8, ncols=1)
    setup_axis(ax)


def draw_update_demo(ax: plt.Axes) -> None:
    target = np.array([0.1, 0.5, 0.4])
    current = np.array([0.3, 0.3, 0.4])
    logits = np.log(current)
    grad = current - target
    lr = 0.5
    new_logits = logits - lr * grad
    updated = softmax(new_logits)

    x = np.arange(target.size)
    width = 0.25
    ax.bar(x - width, target, width=width, color="#45a365", label="target")
    ax.bar(x, current, width=width, color="#f07f24", label="current")
    ax.bar(x + width, updated, width=width, color="#4c98c9", label="after one logit step")

    for i, g in enumerate(grad):
        direction = "down" if g > 0 else ("up" if g < 0 else "flat")
        ax.annotate(
            f"grad={g:+.2f}\n{direction}",
            xy=(i, current[i] + 0.025),
            xytext=(i, 0.72),
            ha="center",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 1.0},
        )

    before_loss = cross_entropy(target, current)
    after_loss = cross_entropy(target, updated)
    ax.text(
        2.35,
        0.48,
        f"CE loss\nbefore={before_loss:.3f}\nafter={after_loss:.3f}",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75"},
    )
    ax.set_title("4. One gradient step moves current distribution toward target", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["atom 1", "atom 2", "atom 3"])
    ax.set_ylabel("probability")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=9, ncols=3, loc="upper left")
    setup_axis(ax)


def draw_overview(output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    draw_loss_landscape(axes[0, 0])
    draw_ce_vs_mse(axes[0, 1])
    draw_gradient_lines(axes[1, 0])
    draw_update_demo(axes[1, 1])
    fig.suptitle("Cross-entropy for C51 critic loss: target weights log current probabilities", fontsize=15)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_contribution_matrix(output_path: Path) -> None:
    target = np.array([0.1, 0.5, 0.4])
    scenarios = np.array(
        [
            [0.30, 0.30, 0.40],
            [0.23, 0.35, 0.42],
            [0.12, 0.48, 0.40],
            [0.10, 0.50, 0.40],
            [0.01, 0.89, 0.10],
        ]
    )
    scenario_names = ["start", "1 step", "near target", "target", "bad miss"]
    contributions = -target[None, :] * np.log(np.clip(scenarios, EPS, 1.0))
    losses = contributions.sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    ax0, ax1 = axes

    im = ax0.imshow(contributions, cmap="YlOrRd", aspect="auto")
    ax0.set_title("Per-atom CE contribution: -target_k log(current_k)")
    ax0.set_xticks(np.arange(3))
    ax0.set_xticklabels(["atom 1", "atom 2", "atom 3"])
    ax0.set_yticks(np.arange(len(scenario_names)))
    ax0.set_yticklabels(scenario_names)
    for row in range(contributions.shape[0]):
        for col in range(contributions.shape[1]):
            ax0.text(col, row, f"{contributions[row, col]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax0, shrink=0.82, label="loss contribution")

    bottoms = np.zeros(len(scenario_names))
    colors = ["#45a365", "#f07f24", "#4c98c9"]
    x = np.arange(len(scenario_names))
    for atom_idx, color in enumerate(colors):
        ax1.bar(
            x,
            contributions[:, atom_idx],
            bottom=bottoms,
            color=color,
            label=f"atom {atom_idx + 1}",
            width=0.62,
        )
        bottoms += contributions[:, atom_idx]
    ax1.plot(x, losses, color="black", marker="o", linewidth=1.7, label="total CE")
    for i, loss in enumerate(losses):
        ax1.text(i, loss + 0.04, f"{loss:.2f}", ha="center", fontsize=9)
    ax1.set_title("Same numbers as stacked bars, without overlap")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, rotation=20, ha="right")
    ax1.set_ylabel("cross entropy")
    ax1.legend(fontsize=9)
    setup_axis(ax1)

    fig.suptitle("Where does the cross-entropy loss come from?", fontsize=15)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_simplex_path(output_path: Path) -> None:
    target = np.array([0.1, 0.5, 0.4])
    current = np.array([0.3, 0.3, 0.4])
    logits = np.log(current)
    lr = 0.5

    path = [current]
    losses = [cross_entropy(target, current)]
    for _ in range(9):
        probs = softmax(logits)
        grad = probs - target
        logits = logits - lr * grad
        probs = softmax(logits)
        path.append(probs)
        losses.append(cross_entropy(target, probs))
    path_arr = np.array(path)

    # Barycentric coordinates for the probability simplex.
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2.0],
        ]
    )

    def to_xy(probs: np.ndarray) -> np.ndarray:
        return probs @ vertices

    xy = to_xy(path_arr)
    target_xy = to_xy(target)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.1, 0.9]},
    )
    ax, ax_loss = axes
    triangle = plt.Polygon(vertices, closed=True, fill=False, edgecolor="0.25", linewidth=1.5)
    ax.add_patch(triangle)
    ax.text(vertices[0, 0] - 0.04, vertices[0, 1] - 0.05, "atom 1", ha="right")
    ax.text(vertices[1, 0] + 0.04, vertices[1, 1] - 0.05, "atom 2", ha="left")
    ax.text(vertices[2, 0], vertices[2, 1] + 0.04, "atom 3", ha="center")

    ax.plot(xy[:, 0], xy[:, 1], color="#3567a8", marker="o", linewidth=2.0, label="gradient steps")
    ax.scatter([target_xy[0]], [target_xy[1]], color="#bd3c37", s=120, marker="*", label="target")
    key_steps = {0: "start", 1: "1 step", 4: "step 4", 9: "step 9"}
    offsets = {
        0: (-0.15, 0.05),
        1: (-0.12, -0.08),
        4: (0.02, 0.07),
        9: (0.03, -0.08),
    }
    for step, label in key_steps.items():
        point = xy[step]
        dx, dy = offsets[step]
        ax.annotate(
            label,
            xy=point,
            xytext=(point[0] + dx, point[1] + dy),
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.9},
        )
    ax.annotate(
        "target",
        xy=target_xy,
        xytext=(target_xy[0] + 0.06, target_xy[1] + 0.08),
        color="#8a1f1b",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#8a1f1b", "lw": 1.0},
    )

    ax.set_title("Probability simplex view: repeated CE steps move current toward target")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right")
    ax_loss.plot(np.arange(len(losses)), losses, color="#3567a8", marker="o", linewidth=2.0)
    ax_loss.set_title("Cross-entropy decreases along the path")
    ax_loss.set_xlabel("gradient step")
    ax_loss.set_ylabel("CE loss")
    ax_loss.set_xticks(np.arange(len(losses)))
    ax_loss.grid(alpha=0.25)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    for step in [0, 1, 4, 9]:
        ax_loss.text(step, losses[step] + 0.006, f"{losses[step]:.3f}", ha="center", fontsize=8)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_two_class_vector_view(output_path: Path) -> None:
    target = np.array([0.2, 0.8])
    current = np.array([0.8, 0.2])
    surprise = -np.log(current)
    ce = cross_entropy(target, current)
    ordinary_dot = float(target @ current)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    ax, ax_bar = axes

    feasible = np.linspace(0.0, 1.0, 200)
    ax.plot(feasible, 1.0 - feasible, color="0.35", linewidth=2.0, label="probability constraint: x + y = 1")
    ax.fill_between(feasible, 0.0, 1.0 - feasible, color="0.92", alpha=0.7, label="x>=0, y>=0, x+y<=1")

    vectors = [
        ("p_target", target, "#45a365"),
        ("p_current", current, "#f07f24"),
        ("-log(p_current)", surprise, "#bd3c37"),
    ]
    for label, vec, color in vectors:
        ax.arrow(
            0.0,
            0.0,
            vec[0],
            vec[1],
            width=0.008,
            head_width=0.045,
            length_includes_head=True,
            color=color,
            alpha=0.88,
        )
        ax.scatter([vec[0]], [vec[1]], s=80, color=color, edgecolor="black", zorder=4)
        ax.text(vec[0] + 0.035, vec[1] + 0.035, f"{label}\n({vec[0]:.3f}, {vec[1]:.3f})", color=color, fontsize=10)

    ax.annotate(
        "probability vectors live on this line",
        xy=(0.45, 0.55),
        xytext=(0.08, 1.33),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "0.25"},
    )
    ax.annotate(
        "surprise is not a probability vector",
        xy=surprise,
        xytext=(0.42, 1.75),
        color="#8a1f1b",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#8a1f1b"},
    )

    ax.set_title("Two-class vectors in x-y coordinates")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.9)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=9, loc="upper right")
    setup_axis(ax)

    labels = ["atom 1", "atom 2"]
    x = np.arange(2)
    width = 0.28
    contributions = target * surprise
    ax_bar.bar(x - width, target, width=width, color="#45a365", label="target weight")
    ax_bar.bar(x, surprise, width=width, color="#bd3c37", label="surprise = -log current")
    ax_bar.bar(x + width, contributions, width=width, color="#4c98c9", label="product")
    for i, value in enumerate(contributions):
        ax_bar.text(i + width, value + 0.04, f"{value:.3f}", ha="center", fontsize=9)
    ax_bar.text(
        0.52,
        1.36,
        f"cross entropy = target · surprise\n= {ce:.3f}\n\nordinary dot target · current\n= {ordinary_dot:.3f}",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75"},
    )
    ax_bar.set_title("CE is a dot product with surprise, not with current probability")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("value")
    ax_bar.set_ylim(0, 1.85)
    ax_bar.legend(fontsize=9)
    setup_axis(ax_bar)

    fig.suptitle("Two-class cross-entropy geometry: probability simplex vs surprise vector", fontsize=15)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_two_class_matched_view(output_path: Path) -> None:
    target = np.array([0.2, 0.8])
    current = target.copy()
    surprise = -np.log(current)
    ce = cross_entropy(target, current)
    entropy = cross_entropy(target, target)

    fig, ax = plt.subplots(figsize=(7.2, 6.5), constrained_layout=True)
    feasible = np.linspace(0.0, 1.0, 200)
    ax.plot(feasible, 1.0 - feasible, color="0.35", linewidth=2.0, label="probability constraint: x + y = 1")
    ax.fill_between(feasible, 0.0, 1.0 - feasible, color="0.92", alpha=0.7)

    ax.arrow(
        0.0,
        0.0,
        target[0],
        target[1],
        width=0.009,
        head_width=0.045,
        length_includes_head=True,
        color="#45a365",
        alpha=0.80,
        label="p_target = p_current",
    )
    ax.scatter([target[0]], [target[1]], s=120, color="#45a365", edgecolor="black", zorder=4)
    ax.text(
        target[0] + 0.035,
        target[1] + 0.035,
        f"p_target = p_current\n({target[0]:.3f}, {target[1]:.3f})",
        color="#24753f",
        fontsize=10,
    )

    ax.arrow(
        0.0,
        0.0,
        surprise[0],
        surprise[1],
        width=0.009,
        head_width=0.045,
        length_includes_head=True,
        color="#bd3c37",
        alpha=0.80,
        label="-log(p_current)",
    )
    ax.scatter([surprise[0]], [surprise[1]], s=120, color="#bd3c37", edgecolor="black", zorder=4)
    ax.text(
        surprise[0] + 0.035,
        surprise[1] + 0.035,
        f"-log(p_current)\n({surprise[0]:.3f}, {surprise[1]:.3f})",
        color="#8a1f1b",
        fontsize=10,
    )

    ax.annotate(
        "probability vectors coincide here",
        xy=target,
        xytext=(0.46, 1.08),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "0.25"},
    )
    ax.annotate(
        "surprise vector is outside the probability line",
        xy=surprise,
        xytext=(1.08, 0.62),
        color="#8a1f1b",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#8a1f1b"},
    )

    ax.text(
        0.42,
        1.39,
        f"CE(target,current) = {ce:.3f}\nH(target) = {entropy:.3f}\nminimum is entropy, not zero",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75"},
    )
    ax.set_title("When p_target and p_current coincide, surprise still lives elsewhere")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_xlim(0, 1.85)
    ax.set_ylim(0, 1.62)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=9, loc="lower right")
    setup_axis(ax)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_two_class_constraint_scan(output_path: Path) -> None:
    target = np.array([0.2, 0.8])
    # Eight evenly spaced interior points on x + y = 1. Endpoints are avoided because -log(0) is infinite.
    xs = np.linspace(0.1, 0.9, 8)
    currents = np.stack([xs, 1.0 - xs], axis=1)
    surprise_vectors = -np.log(currents)
    ce_values = np.array([cross_entropy(target, current) for current in currents])
    target_ce = cross_entropy(target, target)
    target_surprise = -np.log(target)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.8, 6.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    ax_vec, ax_loss = axes

    feasible = np.linspace(0.0, 1.0, 200)
    ax_vec.plot(feasible, 1.0 - feasible, color="0.35", linewidth=2.0, label="probability constraint: x + y = 1")
    ax_vec.fill_between(feasible, 0.0, 1.0 - feasible, color="0.93", alpha=0.7)

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(currents)))
    for idx, (current, surprise, color) in enumerate(zip(currents, surprise_vectors, colors), start=1):
        ax_vec.scatter(current[0], current[1], color=color, edgecolor="black", s=54, zorder=4)
        ax_vec.text(current[0] + 0.015, current[1] + 0.018, str(idx), color=color, fontsize=9, fontweight="bold")
        ax_vec.plot(
            [0.0, surprise[0]],
            [0.0, surprise[1]],
            color=color,
            linestyle=(0, (3, 3)),
            linewidth=1.35,
            alpha=0.78,
        )
        ax_vec.scatter(surprise[0], surprise[1], color=color, marker="x", s=44, zorder=3)

    ax_vec.scatter(target[0], target[1], color="#bd3c37", edgecolor="black", marker="*", s=180, zorder=5)
    ax_vec.plot(
        [0.0, target_surprise[0]],
        [0.0, target_surprise[1]],
        color="#bd3c37",
        linestyle=(0, (1.5, 2.5)),
        linewidth=2.0,
        alpha=0.95,
        label="-log vector for p_current=p_target",
    )
    ax_vec.scatter(target_surprise[0], target_surprise[1], color="#bd3c37", marker="x", s=80, zorder=5)
    ax_vec.annotate(
        "special point:\np_current = p_target",
        xy=target,
        xytext=(0.44, 1.16),
        color="#8a1f1b",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#8a1f1b"},
    )
    ax_vec.annotate(
        "thin dashed vectors are -log(p_current)",
        xy=surprise_vectors[-1],
        xytext=(1.10, 1.72),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "0.25"},
    )
    ax_vec.set_title("Eight p_current points on x+y=1 and their surprise vectors")
    ax_vec.set_xlabel("component 1")
    ax_vec.set_ylabel("component 2")
    ax_vec.set_xlim(0, 2.45)
    ax_vec.set_ylim(0, 2.45)
    ax_vec.set_aspect("equal", adjustable="box")
    ax_vec.legend(fontsize=8, loc="upper right")
    setup_axis(ax_vec)

    point_ids = np.arange(1, len(currents) + 1)
    ax_loss.plot(point_ids, ce_values, color="0.25", linewidth=1.8, marker="o", label="CE at sampled p_current")
    ax_loss.scatter(point_ids, ce_values, color=colors, edgecolor="black", s=72, zorder=4)
    for idx, ce_value in zip(point_ids, ce_values):
        ax_loss.text(idx, ce_value + 0.035, f"{ce_value:.2f}", ha="center", fontsize=8)
    ax_loss.axhline(target_ce, color="#bd3c37", linestyle="--", linewidth=1.7, label=f"minimum at target: {target_ce:.3f}")
    ax_loss.scatter([2], [target_ce], color="#bd3c37", marker="*", s=180, zorder=5)
    ax_loss.annotate(
        "p_current=p_target\nnot one of the 8 sampled points",
        xy=(2, target_ce),
        xytext=(3.0, target_ce + 0.35),
        color="#8a1f1b",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#8a1f1b"},
    )
    ax_loss.set_title("Dot product: p_target · [-log(p_current)]")
    ax_loss.set_xlabel("sample point id on x+y=1")
    ax_loss.set_ylabel("cross entropy")
    ax_loss.set_xticks(point_ids)
    ax_loss.set_xlim(0.55, len(currents) + 0.45)
    ax_loss.legend(fontsize=9)
    setup_axis(ax_loss)

    fig.suptitle("Scanning p_current along the probability constraint line", fontsize=15)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_two_class_target_projection(output_path: Path) -> None:
    target = np.array([0.2, 0.8])
    target_norm = float(np.linalg.norm(target))
    target_unit = target / target_norm

    xs = np.linspace(0.1, 0.9, 8)
    currents = np.stack([xs, 1.0 - xs], axis=1)
    surprise_vectors = -np.log(currents)
    ce_values = surprise_vectors @ target
    projection_lengths = ce_values / target_norm
    projection_points = np.outer(ce_values / (target @ target), target)

    target_surprise = -np.log(target)
    target_ce = float(target @ target_surprise)
    target_projection = (target_ce / (target @ target)) * target
    target_projection_length = target_ce / target_norm

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 6.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )
    ax_vec, ax_len = axes

    feasible = np.linspace(0.0, 1.0, 200)
    ax_vec.plot(feasible, 1.0 - feasible, color="0.72", linewidth=1.5, label="probability line x+y=1")
    ray_end = target_unit * 2.35
    ax_vec.arrow(
        0.0,
        0.0,
        ray_end[0],
        ray_end[1],
        width=0.008,
        head_width=0.045,
        length_includes_head=True,
        color="#24753f",
        alpha=0.86,
        label="target direction",
    )
    ax_vec.scatter([target[0]], [target[1]], color="#45a365", edgecolor="black", s=95, zorder=5)
    ax_vec.text(target[0] + 0.035, target[1] + 0.025, "p_target", color="#24753f", fontsize=10)

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(currents)))
    for idx, (current, surprise, proj, color) in enumerate(
        zip(currents, surprise_vectors, projection_points, colors),
        start=1,
    ):
        ax_vec.scatter(current[0], current[1], color=color, edgecolor="black", s=42, zorder=4)
        ax_vec.text(current[0] + 0.014, current[1] + 0.018, str(idx), color=color, fontsize=8, fontweight="bold")
        ax_vec.plot(
            [0.0, surprise[0]],
            [0.0, surprise[1]],
            color=color,
            linestyle=(0, (3, 3)),
            linewidth=1.15,
            alpha=0.62,
        )
        ax_vec.scatter(surprise[0], surprise[1], color=color, marker="x", s=38, zorder=4)
        ax_vec.plot(
            [surprise[0], proj[0]],
            [surprise[1], proj[1]],
            color=color,
            linewidth=0.9,
            alpha=0.72,
        )
        ax_vec.scatter(proj[0], proj[1], color=color, marker="o", s=26, zorder=5)

    ax_vec.plot(
        [0.0, target_surprise[0]],
        [0.0, target_surprise[1]],
        color="#bd3c37",
        linestyle=(0, (1.5, 2.5)),
        linewidth=2.0,
        label="-log vector at target",
    )
    ax_vec.scatter(target_surprise[0], target_surprise[1], color="#bd3c37", marker="x", s=76, zorder=6)
    ax_vec.plot(
        [target_surprise[0], target_projection[0]],
        [target_surprise[1], target_projection[1]],
        color="#bd3c37",
        linewidth=1.3,
        alpha=0.9,
    )
    ax_vec.scatter(target_projection[0], target_projection[1], color="#bd3c37", marker="*", s=150, zorder=7)
    ax_vec.annotate(
        "orthogonal drops onto\nthe target ray",
        xy=projection_points[-1],
        xytext=(1.18, 1.66),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "0.25"},
    )
    ax_vec.annotate(
        "projected length controls CE",
        xy=target_projection,
        xytext=(0.70, 0.95),
        color="#8a1f1b",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#8a1f1b"},
    )
    ax_vec.set_title("Project surprise endpoints onto the p_target ray")
    ax_vec.set_xlabel("component 1")
    ax_vec.set_ylabel("component 2")
    ax_vec.set_xlim(0, 2.45)
    ax_vec.set_ylim(0, 2.45)
    ax_vec.set_aspect("equal", adjustable="box")
    ax_vec.legend(fontsize=8, loc="upper right")
    setup_axis(ax_vec)

    point_ids = np.arange(1, len(currents) + 1)
    ax_len.bar(point_ids, projection_lengths, color=colors, edgecolor="black", width=0.62, label="projected length")
    ax_len.plot(
        point_ids,
        ce_values,
        color="0.20",
        marker="o",
        linewidth=1.8,
        label="CE = ||target|| * projected length",
    )
    for idx, length, ce_value in zip(point_ids, projection_lengths, ce_values):
        ax_len.text(idx, length + 0.035, f"len {length:.2f}", ha="center", fontsize=8)
        ax_len.text(idx, ce_value + 0.035, f"CE {ce_value:.2f}", ha="center", fontsize=8, color="0.18")
    ax_len.axhline(
        target_projection_length,
        color="#bd3c37",
        linestyle="--",
        linewidth=1.7,
        label=f"target projected length={target_projection_length:.3f}",
    )
    ax_len.scatter([2], [target_projection_length], color="#bd3c37", marker="*", s=165, zorder=5)
    ax_len.set_title("Projection length and cross-entropy")
    ax_len.set_xlabel("sample point id on x+y=1")
    ax_len.set_ylabel("value")
    ax_len.set_xticks(point_ids)
    ax_len.set_xlim(0.55, len(currents) + 0.45)
    ax_len.legend(fontsize=8)
    setup_axis(ax_len)

    fig.suptitle(
        "Dot-product geometry: cross entropy is target-weighted surprise projected onto p_target",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    image_dir = Path(__file__).with_name("imagefastsac")
    image_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        image_dir / "cross_entropy_overview.png",
        image_dir / "cross_entropy_contributions.png",
        image_dir / "cross_entropy_simplex_path.png",
        image_dir / "cross_entropy_2class_vectors.png",
        image_dir / "cross_entropy_2class_matched_vectors.png",
        image_dir / "cross_entropy_2class_constraint_scan.png",
        image_dir / "cross_entropy_2class_target_projection.png",
    ]
    draw_overview(outputs[0])
    draw_contribution_matrix(outputs[1])
    draw_simplex_path(outputs[2])
    draw_two_class_vector_view(outputs[3])
    draw_two_class_matched_view(outputs[4])
    draw_two_class_constraint_scan(outputs[5])
    draw_two_class_target_projection(outputs[6])

    for output in outputs:
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
