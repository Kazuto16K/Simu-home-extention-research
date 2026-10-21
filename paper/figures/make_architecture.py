"""Architecture figure for the paper (vector PDF + PNG preview). Run: python make_architecture.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans"})
FS = 6.3
fig, ax = plt.subplots(figsize=(7.0, 3.1))
ax.set_xlim(-0.1, 14.1); ax.set_ylim(-0.75, 6.25); ax.axis("off")


def box(x, y, w, h, text, fs=FS, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec="black",
                                lw=0.9, ls="--" if dashed else "-"))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.22)


def arrow(p, q, text=None, dx=0.0, dy=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="black", linestyle=ls,
                                 shrinkA=0, shrinkB=0))
    if text:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, fontsize=FS - 0.3, ha="center", va="center",
                style="italic")


ax.text(0.0, 6.08, "Phase 1  Pre-flight: once per request, no large-model call", fontsize=7, weight="bold")
ax.text(0.0, 2.62, "Phase 2  Plan verification: every scheduling call of the agent", fontsize=7, weight="bold")
ax.plot([0, 14], [2.92, 2.92], color="black", lw=0.5, ls=":")

y1, h1 = 3.35, 1.75
box(0.0, y1, 1.55, h1, "User\nrequest")
box(1.85, y1, 2.45, h1, "Small LM\nformalizer\n(Qwen3-1.7B\n+ LoRA)\nrequest -> IR")
box(4.6, y1, 2.45, h1, "Deterministic\ngrounding\nand repair\n(times, device\nnames, chains)")
box(7.35, y1, 2.45, h1, "SMT check (Z3)\nsatisfiable?\nper-goal\ntime windows")
box(7.35, 5.3, 2.45, 0.5, "home: device timers", fs=FS - 0.2)
box(10.4, 4.35, 3.6, 1.25, "UNSAT: contradiction\nhint: warn the agent, or\nanswer: reply, 0 LLM calls")
box(10.4, 3.1, 3.6, 0.95, "SAT: allowed window\nfor every action")
y = y1 + h1 / 2
arrow((1.55, y), (1.85, y)); arrow((4.3, y), (4.6, y)); arrow((7.05, y), (7.35, y))
arrow((8.575, 5.3), (8.575, y1 + h1))
arrow((9.8, y + 0.35), (10.4, 4.95), "no", dy=0.2); arrow((9.8, y - 0.35), (10.4, 3.6), "yes", dy=-0.2)

y2, h2 = 0.25, 1.6
box(0.0, y2, 2.1, h2, "LLM agent\n(large model,\nunchanged)\ncalls\nschedule_workflow")
box(2.55, 0.05, 7.45, 2.1, "", dashed=True)
ax.text(2.7, 1.95, "Verification gate around the scheduling tool", fontsize=FS, weight="bold", ha="left", va="center")
box(2.75, y2, 2.15, 1.45, "A  request\nimpossible?\n-> final refusal")
box(5.15, y2, 2.35, 1.45, "C  dry run on a\nhome copy: fails?\nno effect? which\ngoal does it serve?")
box(7.75, y2, 2.05, 1.45, "B  start time\ninside that goal's\nwindow?")
box(10.4, 0.45, 3.6, 1.05, "Home: workflow\nscheduled")
ym = y2 + 0.72
arrow((2.1, ym), (2.75, ym)); arrow((4.9, ym), (5.15, ym)); arrow((7.5, ym), (7.75, ym))
arrow((10.0, ym + 0.25), (10.4, ym + 0.25), "pass", dy=0.38)
ax.plot([6.3, 6.3], [0.05, -0.3], color="black", lw=0.8, ls="--")
ax.plot([6.3, 1.05], [-0.3, -0.3], color="black", lw=0.8, ls="--")
arrow((1.05, -0.3), (1.05, y2), ls="--")
ax.text(3.7, -0.58, "refused: reason + fix (window, real command signature, state)", fontsize=FS - 0.3, style="italic",
        ha="center")
arrow((12.2, 3.1), (9.3, 1.7), "windows", dx=0.75, dy=0.15, ls="--")
for ext in ("pdf", "png"):
    plt.savefig(f"architecture.{ext}", bbox_inches="tight", dpi=230)
