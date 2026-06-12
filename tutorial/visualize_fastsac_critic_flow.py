"""Draw FastSAC critic gradient-flow diagrams.

Run from the repository root:

    uv run python tutorial/visualize_fastsac_critic_flow.py
"""

from __future__ import annotations

# ruff: noqa: I001

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.use("Agg")


def add_box(ax, xy, width, height, text, *, facecolor, edgecolor="#263238", fontsize=10):
    rect = plt.Rectangle(
        xy,
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6,
        joinstyle="round",
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1f2933",
        wrap=True,
    )
    return rect


def arrow(ax, start, end, *, color="#3b4252", linestyle="-", linewidth=1.8, text=None, text_offset=(0, 0)):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": linewidth,
            "linestyle": linestyle,
            "shrinkA": 4,
            "shrinkB": 4,
        },
    )
    if text:
        ax.text(
            (start[0] + end[0]) / 2 + text_offset[0],
            (start[1] + end[1]) / 2 + text_offset[1],
            text,
            color=color,
            fontsize=9,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )


def draw_critic_gradient_flow(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 9.2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    colors = {
        "data": "#f6e7b4",
        "target": "#e2ecff",
        "online": "#e7f6e9",
        "loss": "#ffe1df",
        "alpha": "#efe5ff",
        "update": "#dff4f5",
        "note": "#f7f7f7",
    }

    add_box(
        ax,
        (0.5, 6.15),
        2.2,
        1.0,
        "Replay batch\n(s, a, r, s', done, truncated)",
        facecolor=colors["data"],
    )
    add_box(
        ax,
        (0.5, 4.65),
        2.2,
        0.85,
        "optional symmetry\nbatch -> mirrored batch",
        facecolor=colors["data"],
        fontsize=9,
    )

    add_box(
        ax,
        (3.25, 7.1),
        2.25,
        0.9,
        "actor(next_obs)\nnext_actions, logπ(a'|s')",
        facecolor=colors["target"],
    )
    add_box(
        ax,
        (6.0, 7.1),
        2.35,
        0.9,
        "adjusted reward\nr_adj = r - γ·bootstrap·α·logπ",
        facecolor=colors["target"],
        fontsize=9,
    )
    add_box(
        ax,
        (8.95, 7.1),
        2.45,
        0.9,
        "target critic ensemble\nQ_target1, Q_target2",
        facecolor=colors["target"],
    )
    add_box(
        ax,
        (12.0, 7.1),
        2.45,
        0.9,
        "C51 Bellman projection\nstop-gradient target_dist",
        facecolor=colors["target"],
        fontsize=9,
    )

    add_box(
        ax,
        (3.25, 3.9),
        2.25,
        0.9,
        "online critic ensemble\nQ1(s,a), Q2(s,a)",
        facecolor=colors["online"],
    )
    add_box(
        ax,
        (6.0, 3.9),
        2.35,
        0.9,
        "log_softmax(logits)\ncritic_log_probs",
        facecolor=colors["online"],
    )
    add_box(
        ax,
        (8.95, 3.9),
        2.45,
        0.9,
        "cross entropy\n-target_dist · log p_current",
        facecolor=colors["loss"],
        fontsize=9,
    )
    add_box(
        ax,
        (12.0, 3.9),
        2.45,
        0.9,
        "qf_loss\nmean batch, sum Q1+Q2",
        facecolor=colors["loss"],
    )

    add_box(
        ax,
        (8.95, 1.7),
        2.45,
        0.85,
        "backward()\n∂L/∂logits = p_current - target",
        facecolor=colors["update"],
        fontsize=9,
    )
    add_box(
        ax,
        (12.0, 1.7),
        2.45,
        0.85,
        "q_optimizer.step()\nupdates online Q1/Q2 only",
        facecolor=colors["update"],
        fontsize=9,
    )

    add_box(
        ax,
        (3.25, 0.45),
        2.25,
        0.85,
        "alpha_loss\n-log_alpha.exp() · (logπ.detach + target_entropy)",
        facecolor=colors["alpha"],
        fontsize=8.5,
    )
    add_box(
        ax,
        (6.0, 0.45),
        2.35,
        0.85,
        "alpha_optimizer.step()\nupdates log_alpha only",
        facecolor=colors["alpha"],
        fontsize=9,
    )
    add_box(
        ax,
        (12.0, 0.45),
        2.45,
        0.85,
        "soft_update_target()\ntarget ← (1-τ)target + τ online",
        facecolor=colors["update"],
        fontsize=8.8,
    )

    # Forward/data flow.
    arrow(ax, (2.7, 6.65), (3.25, 7.55), color="#3b4252", text="next_obs", text_offset=(-0.18, 0.1))
    arrow(ax, (2.7, 6.25), (3.25, 4.35), color="#3b4252", text="critic_obs, action", text_offset=(-0.1, -0.15))
    arrow(ax, (1.6, 6.15), (1.6, 5.5), color="#3b4252", text="if enabled", text_offset=(0.35, 0))
    arrow(ax, (5.5, 7.55), (6.0, 7.55), color="#3b4252")
    arrow(ax, (8.35, 7.55), (8.95, 7.55), color="#3b4252", text="r_adj, s', a'", text_offset=(0, 0.22))
    arrow(ax, (11.4, 7.55), (12.0, 7.55), color="#3b4252")
    arrow(ax, (5.5, 4.35), (6.0, 4.35), color="#3b4252")
    arrow(ax, (8.35, 4.35), (8.95, 4.35), color="#3b4252")
    arrow(ax, (14.45, 7.1), (11.4, 4.8), color="#8b5e00", linestyle="--", text="target_dist\nno grad", text_offset=(0.25, 0.15))
    arrow(ax, (11.4, 4.35), (12.0, 4.35), color="#3b4252")

    # Gradient/update flows.
    arrow(ax, (13.2, 3.9), (10.2, 2.55), color="#c92a2a", linewidth=2.2, text="gradient", text_offset=(0.1, 0.2))
    arrow(ax, (10.2, 1.7), (13.2, 1.7), color="#c92a2a", linewidth=2.2)
    arrow(ax, (12.95, 1.7), (4.35, 3.9), color="#c92a2a", linewidth=2.2, text="updates online critic params", text_offset=(-0.5, -0.58))

    # Alpha path.
    arrow(ax, (5.35, 7.1), (5.35, 1.3), color="#7e57c2", linestyle="--", text="next_log_probs.detach()", text_offset=(0.9, 0.15))
    arrow(ax, (5.5, 0.88), (6.0, 0.88), color="#7e57c2", linewidth=2.0)

    # EMA path.
    arrow(ax, (13.2, 1.7), (13.2, 1.3), color="#007c89", linewidth=2.0)
    arrow(ax, (13.2, 0.45), (10.15, 7.1), color="#007c89", linewidth=2.0, text="EMA only\nno backward", text_offset=(0.6, 0.05))

    # no_grad boundary.
    boundary = plt.Rectangle(
        (3.05, 6.78),
        11.65,
        1.55,
        fill=False,
        edgecolor="#8b5e00",
        linewidth=2.0,
        linestyle="--",
    )
    ax.add_patch(boundary)
    ax.text(3.15, 8.42, "target construction is inside torch.no_grad()", color="#8b5e00", fontsize=11, fontweight="bold")

    add_box(
        ax,
        (0.5, 1.15),
        2.2,
        1.45,
        "Key rule\nreward and target critic create labels;\nonly online qnet receives critic gradients",
        facecolor=colors["note"],
        fontsize=9,
    )

    # Legend.
    legend_items = [
        ("solid gray", "forward/data flow"),
        ("dashed brown", "stop-gradient target"),
        ("red", "critic gradient/update"),
        ("purple", "alpha-only update"),
        ("teal", "EMA target update"),
    ]
    lx, ly = 0.5, 8.45
    for i, (name, label) in enumerate(legend_items):
        y = ly - i * 0.28
        color = {
            "solid gray": "#3b4252",
            "dashed brown": "#8b5e00",
            "red": "#c92a2a",
            "purple": "#7e57c2",
            "teal": "#007c89",
        }[name]
        linestyle = "--" if name == "dashed brown" else "-"
        ax.plot([lx, lx + 0.45], [y, y], color=color, linestyle=linestyle, linewidth=2)
        ax.text(lx + 0.55, y, label, fontsize=9, va="center")

    fig.suptitle("FastSAC Critic Update: forward targets, critic gradients, alpha update, and target EMA", fontsize=16, y=0.98)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).with_name("imagefastsac")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fastsac_critic_gradient_flow.png"
    draw_critic_gradient_flow(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
