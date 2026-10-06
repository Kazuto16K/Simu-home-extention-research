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


def comparison():
    """SimuHome as shipped vs the paper's suggested pre-validation vs this project's verifier."""
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15); ax.set_ylim(-0.6, 10); ax.axis("off")
    ax.text(7.5, 9.85, "SimuHome vs. our architecture", ha="center", va="top", fontsize=17, fontweight="bold", color=NAVY)

    def bx(x, y, w, h, text, color, tc="white", fs=12.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold", linespacing=1.3)

    def arr(x1, y1, x2, y2, color=GREY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.9, color=color))

    def head(y, label, sub, color):
        ax.text(0.3, y + 1.62, label, fontsize=13.5, fontweight="bold", color=color, va="center")
        ax.text(0.3, y + 1.28, sub, fontsize=10.5, color=GREY, va="center", style="italic")

    H = 0.95
    X = [0.3, 3.2, 6.5, 11.2]
    W = [2.2, 2.7, 4.0, 3.5]

    # ---- A: as shipped
    y = 7.0
    head(y, "A.  SimuHome as shipped (the benchmark)", "the tool only confirms registration; nothing checks the request", RED)
    for x, w, tx, c in zip(X, W, ["User request", "Agent LLM\n(ReAct)", "schedule_workflow", "Simulator\n(Matter devices)"], [GREY, NAVY, RED, GREY]):
        bx(x, y, w, H, tx, c)
    for i in range(3):
        arr(X[i] + W[i], y + H / 2, X[i + 1], y + H / 2)
    ax.text(8.5, y - 0.38, 'returns "registered" for any schedule  =>  contradiction blindness (85% of GPT-4.1 errors on infeasible QT4)',
            ha="center", fontsize=10.5, color=RED, style="italic")

    # ---- B: the paper's suggested future work
    y = 4.2
    head(y, "B.  SimuHome paper's suggested fix (Appendix L: future work, not implemented)", "simulation-based pre-validation: rehearse the whole plan in the simulator first", ORANGE)
    for x, w, tx, c in zip(X, W, ["User request", "Agent LLM\n(ReAct)", "Run the plan in\nthe simulator", "Commit to\nthe real home"], [GREY, NAVY, ORANGE, GREY]):
        bx(x, y, w, H, tx, c)
    for i in range(3):
        arr(X[i] + W[i], y + H / 2, X[i + 1], y + H / 2)
    ax.text(8.5, y - 0.38, "needs a faithful world model and a full simulation per plan: heavier, and slower for the user",
            ha="center", fontsize=10.5, color="#B9770E", style="italic")

    # ---- C: ours
    y = 1.5
    head(y, "C.  Ours: request-level verifier in front of the tool", "check that the stated times are consistent BEFORE anything is scheduled", TEAL)
    bx(X[0], y, W[0], H, "User request", GREY)
    bx(X[1], y, W[1], H, "Agent LLM\n(ReAct)", NAVY)
    ax.add_patch(FancyBboxPatch((X[2], y - 0.12), W[2], H + 0.24, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#EAF4F2", ec=TEAL, lw=2, ls="--"))
    ax.text(X[2] + W[2] / 2, y + H - 0.1, "Verifier", ha="center", va="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(X[2] + W[2] / 2, y + 0.33, "parser  ->  Z3  ->  reason", ha="center", va="center", fontsize=11, color=TEAL, fontweight="bold")
    bx(X[3], y, W[3], H, "schedule_workflow\n+ simulator", GREY)
    arr(X[0] + W[0], y + H / 2, X[1], y + H / 2)
    arr(X[1] + W[1], y + H / 2, X[2] - 0.02, y + H / 2)
    arr(X[2] + W[2], y + H / 2, X[3], y + H / 2, GREEN)
    ax.text(X[2] + W[2] + 0.35, y + H / 2 + 0.28, "pass", ha="center", fontsize=9.5, color=GREEN, fontweight="bold")
    # reject path: verifier -> back to the agent, routed below the row
    rx, ry = X[2] + 1.2, y - 0.62
    ax.plot([rx, rx], [y - 0.12, ry], color=RED, lw=1.9)
    ax.plot([rx, X[1] + W[1] / 2], [ry, ry], color=RED, lw=1.9)
    arr(X[1] + W[1] / 2, ry, X[1] + W[1] / 2, y - 0.01, RED)
    ax.text((rx + X[1] + W[1] / 2) / 2, ry - 0.3, "contradiction: reject (409 + reason); the agent re-plans", ha="center", fontsize=10.5, color=RED, style="italic")

    ax.text(7.5, -0.2, "We do not change the benchmark, retrain the agent or simulate execution:\n"
            "we add a cheap, explainable consistency gate at the one place the tool accepts everything.",
            ha="center", va="center", fontsize=10.5, color=NAVY, linespacing=1.4)
    fig.savefig(os.path.join(HERE, "simuhome_vs_ours.png"), dpi=170, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    architecture(); results(); comparison(); print("wrote", HERE)
