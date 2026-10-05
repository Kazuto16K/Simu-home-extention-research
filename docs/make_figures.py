"""Builds the two figures used in PROJECT_REPORT.md: the architecture diagram and the E1 results chart.
Run: python docs/make_figures.py   (needs matplotlib)"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, TEAL, ORANGE, GREY, RED, GREEN = "#1F3A5F", "#2A9D8F", "#E9A23B", "#6B7280", "#C0392B", "#2E8B57"


def box(ax, x, y, w, h, title, body="", color=NAVY, tcolor="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec="none"))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", color=tcolor, fontsize=11.5, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.78, body, ha="center", va="top", color=tcolor, fontsize=9, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, label="", color=GREY, rad=0.0, lx=0, ly=0.15):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center", fontsize=8.5, color=color, style="italic")


def architecture():
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.text(7.5, 7.95, "Neuro-symbolic verification layer for LLM smart-home scheduling agents",
            ha="center", va="top", fontsize=15, fontweight="bold", color=NAVY)

    # top row: the existing SimuHome pipeline
    box(ax, 0.3, 5.0, 2.3, 1.7, "User request", "natural language\n\"at 11:35, i.e. 13 min\nfrom now, turn on the light\"", GREY)
    box(ax, 3.6, 5.0, 3.0, 1.7, "Agent LLM (ReAct)", "Gemma-31B (not trained)\nreads, plans, calls tools", NAVY)
    box(ax, 8.9, 5.0, 2.5, 1.7, "schedule_workflow", "SimuHome tool:\naccepts ANY schedule", RED)
    box(ax, 12.3, 5.0, 2.4, 1.7, "Simulator", "Matter devices;\nscored at goal ticks", GREY)
    arrow(ax, 2.6, 5.85, 3.6, 5.85)
    arrow(ax, 6.6, 5.85, 8.9, 5.85, "tool call + arguments")
    arrow(ax, 11.4, 5.85, 12.3, 5.85)

    # verifier band
    ax.add_patch(FancyBboxPatch((0.3, 0.35), 14.4, 3.9, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#EAF4F2", ec=TEAL, lw=2, ls="--"))
    ax.text(0.55, 4.05, "OUR CONTRIBUTION: verifier inserted in front of schedule_workflow", fontsize=11.5,
            fontweight="bold", color=TEAL, va="top")
    box(ax, 0.6, 0.65, 3.1, 2.6, "1. Intent parser", "Qwen3-1.7B + LoRA\nquery -> typed IR (JSON)\ngoals = time readings\n(\"after / before / at\",\noffset, anchor)", TEAL)
    box(ax, 4.3, 0.65, 3.1, 2.6, "2. Anchor resolver", "turns \"when dishwasher ends\"\nor \"now\" into numbers\nusing get_current_time\nand device attributes", TEAL)
    box(ax, 8.0, 0.65, 3.1, 2.6, "3. Z3 SMT check", "one real-valued time per goal\nsame event, two times?\nUNSAT => infeasible\nunsat core = which clauses", NAVY)
    box(ax, 11.7, 0.65, 2.8, 2.6, "4. Verbalizer", "template turns the\nunsat core into a plain\nsentence the agent reads", ORANGE, "white")
    arrow(ax, 3.7, 1.95, 4.3, 1.95)
    arrow(ax, 7.4, 1.95, 8.0, 1.95)
    arrow(ax, 11.1, 1.95, 11.7, 1.95)

    # connections to the pipeline (straight arrows, labels kept clear of boxes)
    arrow(ax, 10.15, 5.0, 10.15, 4.28, color=TEAL)
    ax.text(10.35, 4.62, "1  every schedule_workflow call is intercepted", fontsize=9, color=TEAL, va="center", style="italic")
    arrow(ax, 5.1, 4.28, 5.1, 5.0, color=RED)
    ax.text(4.9, 4.62, "2  REJECT (409 + reason): agent re-plans", fontsize=9, color=RED, va="center", ha="right", style="italic")
    ax.text(10.15, 4.38, "", fontsize=1)
    ax.text(7.5, 0.12, "3  feasible  =>  call passes through unchanged (fail-open if the parser output is unusable)",
            fontsize=9, color=GREEN, ha="center", style="italic")
    fig.savefig(os.path.join(HERE, "architecture.png"), dpi=170, bbox_inches="tight", facecolor="white")


def results():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    arms = ["Baseline\n(agent alone)", "With verifier"]
    for ax, title, vals, note in [
        (axes[0], "Infeasible requests (trap episodes)", [30, 100], "agent must notice the contradiction"),
        (axes[1], "Feasible requests (no trap)", [20, 20], "verifier must not block valid schedules"),
    ]:
        bars = ax.bar(arms, vals, color=[GREY, TEAL], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%  ({v // 10}/10)", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 118); ax.set_ylabel("task accuracy (%)")
        ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY); ax.set_xlabel(note, fontsize=9, color=GREY)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("E1 on SimuHome QT4-1, 10 held-out episodes per cell (small sample: preliminary)", fontsize=10.5, color=GREY)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "e1_results.png"), dpi=170, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    architecture(); results(); print("wrote", HERE)
