"""Builds docs/architecture_comparison.png: approach 1 (top) vs approach 2 (bottom).
Run: python docs/make_figures.py   (needs matplotlib)

Every example on the figure is a real benchmark episode (qt4-1_infeasible_seed_15,
qt4-1_feasible_seed_1); every number is measured: approach-1 costs from the session-1
traces, approach-2 behaviour from tests/test_gate.py and tests/test_e2e_fake_llm.py."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, TEAL, ORANGE, GREY, RED, GREEN, PURPLE = "#1F3A5F", "#2A9D8F", "#E08E2B", "#6B7280", "#C0392B", "#2E8B57", "#6D4C9F"
PALE_RED, PALE_TEAL, PALE_BLUE, INK = "#FBEDEC", "#EAF4F2", "#EEF2F8", "#1F2937"


def box(ax, x, y, w, h, title, body="", color=NAVY, fs=11, bfs=8.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec="none"))
    if body:
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", color="white", fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.58, body, ha="center", va="top", color="white", fontsize=bfs, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", color="white", fontsize=fs, fontweight="bold",
                linespacing=1.25)


def diamond(ax, cx, cy, w, h, text, color=NAVY):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True,
                         fc=color, ec="none"))
    ax.text(cx, cy, text, ha="center", va="center", color="white", fontsize=10, fontweight="bold", linespacing=1.2)


def arrow(ax, x1, y1, x2, y2, label="", color=GREY, rad=0.0, ls="-", lx=0.0, ly=0.14, fs=8.6, ha="center", lw=1.7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha=ha, fontsize=fs, color=color, style="italic",
                fontweight="bold")


def example(ax, x, y, text, color=GREY, w=None, ha="left"):
    ax.text(x, y, text, ha=ha, va="top", fontsize=8.3, color=color, style="italic", linespacing=1.3)


def band(ax, x, y, w, h, fc, ec, title, ls="--"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.6, ls=ls))
    ax.text(x + 0.22, y + h - 0.16, title, fontsize=11.5, fontweight="bold", color=ec, va="top")


def architecture_comparison():
    fig, ax = plt.subplots(figsize=(16, 18.2))
    ax.set_xlim(0, 16); ax.set_ylim(-2.6, 15.6); ax.axis("off")
    ax.text(8, 15.5, "Checking an LLM smart-home agent's schedules (SimuHome QT4): approach 1 vs approach 2",
            ha="center", va="top", fontsize=15.5, fontweight="bold", color=NAVY)

    # ================================================================ APPROACH 1
    band(ax, 0.2, 11.25, 15.6, 3.75, PALE_RED, RED, "APPROACH 1  -  one check, hidden inside the scheduling tool")
    y = 12.55
    box(ax, 0.45, y, 1.9, 1.15, "User request")
    box(ax, 2.85, y, 2.2, 1.15, "Agent", "large LLM, plans\nand calls tools", NAVY)
    box(ax, 5.55, y, 2.6, 1.15, "schedule_workflow", "every call is intercepted", RED)
    box(ax, 8.65, y, 2.6, 1.15, "Small model", "re-reads the REQUEST\non every call", TEAL)
    diamond(ax, 12.55, y + 0.575, 1.75, 1.15, "Request\npossible?")
    box(ax, 14.1, y, 1.5, 1.15, "Schedule", "", GREEN, fs=10.5)
    for x1, x2 in ((2.35, 2.85), (5.05, 5.55), (8.15, 8.65), (11.25, 11.68)):
        arrow(ax, x1, y + 0.575, x2, y + 0.575)
    arrow(ax, 13.43, y + 0.575, 14.1, y + 0.575, "yes", GREEN, ly=0.1)
    arrow(ax, 12.55, y, 3.95, y, color=RED, rad=-0.16, ls="--")
    ax.text(8.25, 11.62, "no -> refused (409); the agent tries ANOTHER time, which can never pass",
            ha="center", fontsize=9, color=RED, style="italic", fontweight="bold")
    ax.text(0.45, 14.48,
            "Problems seen in the session-1 traces:  it never looks at the agent's actual time or commands, so wrong "
            "times and invented commands slip through;\nimpossible requests loop (seed 15: 6 large-model calls instead "
            "of 2);  re-reading on every call made feasible episodes 377 s instead of 218 s.",
            fontsize=8.8, color="#7A2A22", va="top", linespacing=1.35)

    # ================================================================ APPROACH 2
    band(ax, 0.2, -2.5, 15.6, 13.45, "#FFFFFF", TEAL, "APPROACH 2  -  check the request once at the start, then check every action the agent takes")

    # ---------------- phase 1: before the agent starts
    band(ax, 0.4, 5.95, 15.2, 4.45, PALE_TEAL, TEAL, "PHASE 1   Before the agent starts  (once per request, no large-model call)", ls="-")
    y1 = 8.0
    box(ax, 0.6, y1, 1.85, 1.35, "User request")
    box(ax, 2.95, y1, 2.55, 1.35, "1  Small model", "turns the sentence into\n'what + when'\n(run once, saved to a file)", TEAL)
    box(ax, 6.0, y1, 2.45, 1.35, "2  Read the house", "device finish times\nfrom the simulator", TEAL)
    box(ax, 8.95, y1, 2.45, 1.35, "3  Z3 solver", "can ONE time satisfy\nevery statement?", NAVY)
    diamond(ax, 12.55, y1 + 0.675, 1.7, 1.35, "Possible?")
    for x1, x2 in ((2.45, 2.95), (5.5, 6.0), (8.45, 8.95), (11.4, 11.7)):
        arrow(ax, x1, y1 + 0.675, x2, y1 + 0.675)
    # "no" outcomes
    box(ax, 13.95, y1 + 0.78, 1.5, 0.82, "hint", "", ORANGE, fs=10.5)
    box(ax, 13.95, y1 - 0.25, 1.5, 0.82, "answer", "", GREEN, fs=10.5)
    arrow(ax, 13.4, y1 + 0.675, 13.95, y1 + 1.19, "no", RED, ly=0.06, lx=-0.05)
    arrow(ax, 13.4, y1 + 0.675, 13.95, y1 + 0.16, color=RED)
    ax.text(14.7, y1 + 1.73, "warn the agent in\nits first message", ha="center", va="bottom", fontsize=8, color=ORANGE,
            fontweight="bold")
    ax.text(14.7, y1 - 0.32, "reply to the user;\n0 large-model calls", ha="center", va="top", fontsize=8, color=GREEN,
            fontweight="bold")
    example(ax, 2.0, 7.75,
            "Example (seed 15):  clock 12:00:33, user: \"at 12:23 PM, that is 12 minutes from now\"\n"
            "  ->  12 min from now = 12:12:33,  stated time = 12:23  ->  no single time fits  ->  IMPOSSIBLE",
            RED)
    example(ax, 2.0, 6.95,
            "Example (seed 1):  \"dimmer at level 30 in 13 min, 8 min after that level 20, utility light in 29 min\"  ->  POSSIBLE\n"
            "  allowed windows:  dimmer 30 at 11:49:24-11:50:24,  dimmer 20 at 11:56:54-11:58:54 (chained to the first),\n"
            "  utility light at 12:05:24-12:06:24",
            NAVY)

    # ---------------- phase 2: while the agent works
    band(ax, 0.4, -2.3, 15.2, 8.0, PALE_BLUE, PURPLE,
         "PHASE 2   While the agent works  (each schedule_workflow call passes THROUGH the gate; the agent waits)", ls="-")
    y2 = 3.2
    # the gate is a wrapper around SimuHome's tool: one container, the call goes in, one reply comes out
    ax.add_patch(FancyBboxPatch((2.95, 2.95), 10.3, 2.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=PURPLE, lw=1.8, ls="--"))
    ax.text(7.05, 4.88, "GATE  =  our wrapper around schedule_workflow", fontsize=9.5, color=PURPLE, fontweight="bold",
            va="top")
    # windows from phase 1 feed check B; the request (+ warning) goes to the agent
    ax.plot([12.55, 12.55, 6.6], [y1, 5.2, 5.2], color=NAVY, ls=":", lw=2.0)
    arrow(ax, 6.6, 5.2, 6.6, y2 + 1.42, color=NAVY, ls=":", lw=2.0)
    ax.text(12.7, 6.15, "yes: time window\nfor every action", fontsize=8.6, color=NAVY, style="italic", fontweight="bold",
            va="center")
    arrow(ax, 1.5, y1, 1.5, y2 + 1.42, color=GREY, lw=2.0)
    ax.text(1.62, 6.12, "request\n(+ warning)", fontsize=8, color=GREY, style="italic", fontweight="bold", va="center")

    box(ax, 0.6, y2, 1.85, 1.4, "Agent", "large LLM,\nnot retrained", NAVY)
    diamond(ax, 4.0, y2 + 0.7, 1.7, 1.4, "A\nrequest\npossible?", ORANGE)
    diamond(ax, 6.6, y2 + 0.7, 1.95, 1.4, "B\ntime inside\nwindow?", PURPLE)
    diamond(ax, 9.35, y2 + 0.7, 2.15, 1.4, "C\ncommands work\non a copy?", PURPLE)
    box(ax, 10.95, y2, 2.1, 1.4, "SimuHome's", "real\nschedule_workflow\nruns it", GREEN, fs=10, bfs=8.4)
    arrow(ax, 2.45, y2 + 0.7, 3.15, y2 + 0.7)
    ax.text(2.8, y2 + 0.82, "call", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    arrow(ax, 4.85, y2 + 0.7, 5.62, y2 + 0.7, "yes", GREEN, ly=0.09)
    arrow(ax, 7.58, y2 + 0.7, 8.27, y2 + 0.7, "yes", GREEN, ly=0.09)
    arrow(ax, 10.43, y2 + 0.7, 10.95, y2 + 0.7, "yes", GREEN, ly=0.09)

    # exactly one reply goes back to the agent: a refusal (red) or the real tool's answer (green)
    red_y, green_y = 2.5, 2.1
    for cx in (4.0, 6.6, 9.35):
        ax.plot([cx, cx], [y2, red_y], color=RED, lw=1.7, ls="--")
    ax.plot([9.35, 1.85], [red_y, red_y], color=RED, lw=1.7, ls="--")
    arrow(ax, 1.85, red_y, 1.85, y2, color=RED, ls="--")
    ax.text(2.0, red_y + 0.08, "no: reply = refused + how to fix it", ha="left", va="bottom", fontsize=8.6, color=RED,
            style="italic", fontweight="bold")
    ax.plot([12.0, 12.0, 1.15], [y2, green_y, green_y], color=GREEN, lw=1.7)
    arrow(ax, 1.15, green_y, 1.15, y2, color=GREEN)
    ax.text(10.2, green_y + 0.08, "reply = scheduled", ha="left", va="bottom", fontsize=8.6, color=GREEN,
            style="italic", fontweight="bold")
    ax.text(14.35, y2 + 0.7, "the agent reads\nthe reply, thinks\nagain, may call\nthe gate again,\n... then finish",
            ha="center", va="center", fontsize=8.4, color=INK, style="italic")

    example(ax, 2.6, 1.85, "A: \"the request itself is\ncontradictory; no time can\nfix it - call finish\"", ORANGE)
    example(ax, 5.35, 1.85, "B (seed 1): agent picks 11:54:54\n-> \"use 11:49:24-11:50:24\"\n-> agent retries 11:49:54: ok", PURPLE)
    example(ax, 8.4, 1.85, "C (seed 1): command 'SetLevel'\n-> \"Command 'SetLevel' not found\nin cluster 'LevelControl'\"\n-> agent fixes it: ok", PURPLE)
    example(ax, 11.4, 1.85, "SimuHome's own scorer\nmarked this episode a\nsuccess (score 1)", GREEN)

    # ---- timeline: one episode happens strictly in order
    ax.text(0.6, 0.25, "ONE EPISODE, IN ORDER  (nothing below runs at the same time as anything else)", fontsize=9.5,
            color=PURPLE, fontweight="bold", va="top")
    steps = [("Phase 1 checks", "~0.05 s", TEAL), ("agent thinks", "~45 s (LLM call)", NAVY),
             ("calls schedule_workflow", "gate: A, B ms; C 1-5 s", PURPLE), ("reply", "refused or scheduled", GREY),
             ("agent thinks again", "~45 s", NAVY), ("... finish", "answer to user", GREEN)]
    x, w, gap = 0.6, 2.25, 0.27
    for i, (t, sub, col) in enumerate(steps):
        box(ax, x, -0.95, w, 0.82, t, sub, col, fs=9, bfs=7.8)
        if i < len(steps) - 1:
            arrow(ax, x + w, -0.54, x + w + gap, -0.54)
        x += w + gap
    ax.text(0.6, -1.2,
            "What DOES run in parallel: separate episodes. Six episodes run at once, each with its own simulator, gate and "
            "agent, sharing one rate-limited API key.\n"
            "Safety: each kind of refusal is allowed at most twice per episode, then calls go through; if the small model's "
            "output is unusable, the gate steps aside.",
            fontsize=8.7, color=INK, va="top", linespacing=1.4)

    fig.savefig(os.path.join(HERE, "architecture_comparison.png"), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def flowchart():
    """Simple top-to-bottom flowchart of ONE episode under approach 2, in time order."""
    fig, ax = plt.subplots(figsize=(12, 19))
    ax.set_xlim(0, 12); ax.set_ylim(0, 19); ax.axis("off")
    ax.text(6, 18.85, "Approach 2: what happens during one request, in order", ha="center", va="top", fontsize=15,
            fontweight="bold", color=NAVY)
    X, W = 5.6, 4.4  # main column centre and box width

    def step(y, text, color, h=0.95, x=X, w=W, fs=10):
        box(ax, x - w / 2, y - h / 2, w, h, text, "", color, fs=fs)

    def oval(y, text, color, x=X, w=3.6):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.38), w, 0.76, boxstyle="round,pad=0.02,rounding_size=0.38",
                                    fc=color, ec="none"))
        ax.text(x, y, text, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")

    def down(y1, y2, label="", color=GREY):
        arrow(ax, X, y1, X, y2, label, color, lx=0.35, ly=-0.05, ha="left")

    # ---- once, before the agent starts
    ax.text(0.3, 18.2, "ONCE, BEFORE THE AGENT STARTS  (no large-model call)", fontsize=10.5, color=TEAL, fontweight="bold")
    oval(17.6, "User sends a request", GREY)
    step(16.45, "1  Look up the small model's\n'what + when' (written once,\nbefore the experiment)", TEAL, fs=9.5)
    step(15.15, "2  Read device finish times\nfrom the simulator", TEAL)
    diamond(ax, X, 13.75, 3.9, 1.35, "3  Z3: is the request\npossible?", NAVY)
    down(17.22, 16.93); down(15.97, 15.63); down(14.67, 14.43)

    # NO branch: two modes
    step(14.75, "answer mode:\nthe gate replies to the user\n(0 large-model calls)", GREEN, h=1.15, x=9.9, w=3.5, fs=9.5)
    oval(13.4, "END", GREEN, x=9.9, w=1.6)
    arrow(ax, 9.9, 14.17, 9.9, 13.79)
    step(12.15, "hint mode:\nthe agent gets a warning:\n'impossible - explain it'", ORANGE, h=1.15, x=9.9, w=3.5, fs=9.5)
    ax.plot([X + 1.95, 7.9], [13.75, 13.75], color=RED, lw=1.7)
    arrow(ax, 7.9, 13.75, 8.15, 14.75, color=RED)
    arrow(ax, 7.9, 13.75, 8.15, 12.15, color=RED)
    ax.text(6.9, 13.9, "NO", fontsize=10, color=RED, fontweight="bold")

    step(12.15, "4  Work out the allowed time window\nfor each action", NAVY)
    down(13.07, 12.63, "YES", GREEN)

    # ---- repeated while the agent works
    ax.plot([0.25, 11.75], [11.3, 11.3], color=PURPLE, lw=1, ls=":")
    ax.text(0.3, 11.15, "REPEATS UNTIL THE AGENT FINISHES", fontsize=10.5, color=PURPLE, fontweight="bold", va="top")
    step(10.2, "5  Agent (large LLM) thinks", NAVY)
    down(11.67, 10.68)
    arrow(ax, 9.9, 11.57, X + W / 2, 10.2, color=ORANGE)  # hint mode joins the loop
    diamond(ax, X, 8.75, 3.9, 1.3, "6  What does the\nagent do?", NAVY)
    down(9.72, 9.4)
    oval(8.75, "replies to the user  ->  END", GREEN, x=9.95, w=3.4)
    arrow(ax, X + 1.95, 8.75, 8.25, 8.75, "finish", GREEN, ly=0.1)
    down(8.1, 7.6, "calls schedule_workflow(time, commands)", NAVY)

    checks = [(6.9, "7  Was the request\nimpossible?", ORANGE, "YES",
               "refused: 'no time can fix this;\ncall finish'"),
              (5.25, "8  Is the time inside\nits window?", PURPLE, "NO",
               "refused: 'use\n11:49:24-11:50:24'"),
              (3.6, "9  Dry run on a copy:\ndo the commands work?", PURPLE, "NO",
               "refused: simulator's error, e.g.\n'Command SetLevel not found'")]
    rail = 0.45
    for i, (y, text, col, bad, msg) in enumerate(checks):
        diamond(ax, X, y, 3.9, 1.3, text, col)
        step(y, msg, RED, h=0.95, x=1.85, w=2.7, fs=8.4)
        arrow(ax, X - 1.95, y, 3.2, y, bad, RED, ly=0.12)
        ax.plot([0.55, rail], [y, y], color=RED, lw=1.7, ls="--")
        good = "NO" if bad == "YES" else "YES"
        nxt = checks[i + 1][0] + 0.65 if i + 1 < len(checks) else 2.3
        down(y - 0.65, nxt, good, GREEN)
    # refusals go back to step 5
    ax.plot([rail, rail], [checks[-1][0], 10.2], color=RED, lw=1.7, ls="--")
    arrow(ax, rail, 10.2, X - W / 2, 10.2, color=RED, ls="--")
    ax.text(0.6, 10.45, "back to the agent", fontsize=8.5, color=RED, style="italic", fontweight="bold")

    step(1.85, "10  Schedule it for real (SimuHome's tool)\nand reply 'scheduled'", GREEN)
    # success also goes back to step 5 (right-hand rail)
    ax.plot([X + W / 2, 11.6, 11.6], [1.85, 1.85, 9.6], color=GREEN, lw=1.7)
    ax.plot([11.6, 9.0], [9.6, 9.6], color=GREEN, lw=1.7)
    arrow(ax, 9.0, 9.6, X + W / 2 - 0.4, 9.73, color=GREEN)
    ax.text(11.45, 5.0, "back to the agent", fontsize=8.5, color=GREEN, style="italic", fontweight="bold",
            rotation=90, ha="right", va="center")

    ax.text(0.3, 1.0,
            "Steps 1-4 happen once.  Steps 5-10 repeat: the agent waits for each reply, then thinks again.\n"
            "Each kind of refusal is allowed at most twice per episode; after that the call goes through.\n"
            "Separate episodes (not these steps) run in parallel: six at once, each with its own simulator and gate.",
            fontsize=9.2, color=INK, va="top", linespacing=1.45)
    fig.savefig(os.path.join(HERE, "approach2_flowchart.png"), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture_comparison()
    flowchart()
    print("wrote", os.path.join(HERE, "architecture_comparison.png"), "and", os.path.join(HERE, "approach2_flowchart.png"))
