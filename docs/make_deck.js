// Approach 2 deck: architecture, results so far, why it beats approach 1.
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(String.raw`C:\Users\HP\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\e104bfc0-842f-4d1c-89de-ecf9992f8fa3\47c9908d-85b2-44c1-81ef-b7fcb5c8fc8a\skills\pptx\scripts\apply_theme.js`);

const OUT = String.raw`D:\IoT and Agents\approach-2\docs\Approach2_Presentation.pptx`;

// Smart-home palette: deep slate-teal dominant, teal for the small model, amber for warnings,
// purple for the plan checks, red for refusals, green for "allowed / scheduled".
const THEME = {
  name: "Approach 2 Gate",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2933", lt1: "FFFFFF", dk2: "16384A", lt2: "EEF3F6",
    accent1: "1B998B", accent2: "E39B2D", accent3: "6D4C9F",
    accent4: "C0392B", accent5: "2E8B57", accent6: "7B8794",
    hlink: "1B998B", folHlink: "6D4C9F",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Approach 2: pre-flight gate for LLM smart-home scheduling";
pres.author = "Soumava Das";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_ONLY",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 12.3, h: 0.75,
        fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: C.text2, valign: "middle", margin: 0 },
      text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.5, y: 1.02, w: 12.3, h: 0.42,
        fontFace: THEME.bodyFontFace, fontSize: 15, color: C.accent6, valign: "top", margin: 0 }, text: "" } },
    { text: { text: "SimuHome QT4 (scheduling) · approach 2", options: { x: 0.5, y: 7.05, w: 7, h: 0.3,
        fontFace: THEME.bodyFontFace, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontFace: THEME.bodyFontFace, fontSize: 10, color: C.accent6, align: "right" },
});

// ---------- helpers ----------
function box(slide, name, x, y, w, h, title, body, fill, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill },
    line: { color: fill }, objectName: name });
  const runs = [{ text: title, options: { bold: true, fontSize: opts.tfs || 14, breakLine: !!body } }];
  if (body) runs.push({ text: body, options: { fontSize: opts.bfs || 11 } });
  slide.addText(runs, { x: x + 0.06, y, w: w - 0.12, h, color: C.background1, align: "center", valign: "middle",
    margin: 2, isTextBox: true, objectName: name + " text", paraSpaceAfter: 2 });
}
function diamond(slide, name, x, y, w, h, text, fill) {
  slide.addShape(pres.shapes.DIAMOND, { x, y, w, h, fill: { color: fill }, line: { color: fill }, objectName: name });
  slide.addText(text, { x, y, w, h, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle",
    margin: 0, isTextBox: true, objectName: name + " text" });
}
function arrow(slide, name, x1, y1, x2, y2, color, dashed) {
  slide.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
    h: Math.abs(y2 - y1) || 0.001, flipH: x2 < x1, flipV: y2 < y1,
    line: { color, width: 2, endArrowType: "triangle", dashType: dashed ? "dash" : "solid" }, objectName: name });
}
function line(slide, name, x1, y1, x2, y2, color, dashed) {
  slide.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
    h: Math.abs(y2 - y1) || 0.001, line: { color, width: 2, dashType: dashed ? "dash" : "solid" }, objectName: name });
}
function label(slide, name, text, x, y, w, h, color, o = {}) {
  slide.addText(text, { x, y, w, h, fontSize: o.fs || 11, color, bold: !!o.bold, italic: !!o.italic,
    align: o.align || "left", valign: o.valign || "middle", margin: 0, isTextBox: true, objectName: name });
}

// =========================================================== slide 1: architecture
pres.addSection({ title: "Architecture" });
let s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Architecture" });
s.addText("Approach 2: check the request once, then check every action", { placeholder: "title" });
s.addText("A small model and a solver decide before the large model acts; SimuHome's agent is unchanged",
  { placeholder: "subtitle" });

// phase 1 lane
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.6, w: 12.3, h: 2.35, rectRadius: 0.1,
  fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "phase 1 lane" });
label(s, "phase 1 label", "1  BEFORE THE AGENT STARTS  ·  once per request  ·  no large-model call", 0.75, 1.7, 9, 0.35,
  C.accent1, { bold: true, fs: 13 });
box(s, "user request 1", 0.75, 2.3, 1.75, 1.2, "User request", "e.g. \"at 12:23 PM, i.e. 12 min from now\"", HEX.accent6);
box(s, "small model", 2.85, 2.3, 2.05, 1.2, "Small model", "Qwen3-1.7B + LoRA writes\n'what + when'", HEX.accent1);
box(s, "read house", 5.25, 2.3, 2.05, 1.2, "Read the house", "device finish times from\nthe simulator", HEX.accent1);
diamond(s, "z3", 7.6, 2.12, 2.0, 1.56, "Z3: one time\nfits all?", HEX.dk2);
box(s, "hint", 10.35, 2.0, 2.25, 0.82, "No → hint", "warn the agent (1 call)", HEX.accent2, { tfs: 13, bfs: 11 });
box(s, "answer", 10.35, 2.98, 2.25, 0.82, "No → answer", "reply directly (0 calls)", HEX.accent5, { tfs: 13, bfs: 11 });
arrow(s, "a1", 2.5, 2.9, 2.85, 2.9, HEX.accent6);
arrow(s, "a2", 4.9, 2.9, 5.25, 2.9, HEX.accent6);
arrow(s, "a3", 7.3, 2.9, 7.6, 2.9, HEX.accent6);
arrow(s, "a4", 9.6, 2.9, 10.35, 2.42, HEX.accent4);
arrow(s, "a5", 9.6, 2.9, 10.35, 3.38, HEX.accent4);

// phase 2 lane
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.2, w: 12.3, h: 2.7, rectRadius: 0.1,
  fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "phase 2 lane" });
label(s, "phase 2 label", "2  WHILE THE AGENT WORKS  ·  every schedule_workflow call passes through the gate", 0.75, 4.3, 9, 0.35,
  C.accent3, { bold: true, fs: 13 });
line(s, "yes down", 8.6, 3.68, 8.6, 4.68, HEX.dk2, true);
line(s, "yes across", 6.67, 4.68, 8.6, 4.68, HEX.dk2, true);
arrow(s, "yes into B", 6.67, 4.68, 6.67, 5.12, HEX.dk2, true);
label(s, "yes label", "Yes: allowed time\nwindow per action", 8.75, 3.9, 1.6, 0.5, C.text2, { fs: 10, italic: true });
box(s, "agent", 0.75, 5.0, 1.75, 1.15, "Agent", "large LLM\n(unchanged)", HEX.dk2);
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 2.85, y: 4.75, w: 7.7, h: 1.65, rectRadius: 0.08,
  fill: { color: C.background1 }, line: { color: C.accent3, width: 1.5, dashType: "dash" }, objectName: "gate frame" });
label(s, "gate label", "GATE (our wrapper around SimuHome's tool)", 2.95, 4.78, 5, 0.3, C.accent3, { bold: true, fs: 10 });
box(s, "check a", 3.0, 5.12, 2.25, 1.1, "A  Request", "impossible? → \"call finish\"", HEX.accent2, { tfs: 13, bfs: 10.5 });
box(s, "check b", 5.55, 5.12, 2.25, 1.1, "B  Time", "too late, or clobbers an\nearlier goal? → window", HEX.accent3, { tfs: 13, bfs: 10.5 });
box(s, "check c", 8.1, 5.12, 2.3, 1.1, "C  Dry run on a copy", "command fails or does\nnothing? → error", HEX.accent3, { tfs: 13, bfs: 10.5 });
box(s, "schedule", 10.85, 5.0, 1.75, 1.15, "Schedule", "SimuHome's\nreal tool", HEX.accent5);
arrow(s, "b1", 2.5, 5.57, 3.0, 5.67, HEX.accent6);
arrow(s, "b2", 5.25, 5.67, 5.55, 5.67, HEX.accent5);
arrow(s, "b3", 7.8, 5.67, 8.1, 5.67, HEX.accent5);
arrow(s, "b4", 10.4, 5.67, 10.85, 5.57, HEX.accent5);
line(s, "back line", 1.6, 6.78, 9.25, 6.78, HEX.accent4, true);
arrow(s, "back up", 1.6, 6.78, 1.6, 6.15, HEX.accent4, true);
line(s, "back from checks", 9.25, 6.22, 9.25, 6.78, HEX.accent4, true);
label(s, "back label", "refused → back to the agent with how to fix it (max 2 per kind, then let through)", 1.75, 6.44, 7.2, 0.3,
  C.accent4, { fs: 10.5, italic: true });
s.addNotes(
  "Phase 1 runs once per request and calls no large model: the small model's reading of the request is " +
  "looked up from a cache, device finish times are read from the simulator (CountdownTime), and Z3 checks " +
  "whether one time can satisfy every statement. If not, hint mode warns the agent, answer mode replies to the " +
  "user directly. If yes, Z3 also computes an allowed time window per action. Phase 2: the agent (SimuHome's " +
  "ReAct agent, unchanged) works as usual; every schedule_workflow call passes through the gate: A refuses calls " +
  "on an impossible request, B refuses times that are too late (or would overwrite an earlier goal on the same " +
  "device; this follows SimuHome's scoring), C dry-runs the steps on a copy of the house and refuses steps that " +
  "fail or change nothing. Refusals explain how to fix the call; each kind is capped at 2 per episode.");

// =========================================================== slide 2: results
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Results" });
s.addText("Results: contradictions solved; possible requests not yet", { placeholder: "title" });
s.addText("Agent and judge: Gemma-4-31B (NVIDIA API) · held-out test episodes, 8-10 per group · gate v2.1 run, Oct 2026",
  { placeholder: "subtitle" });

const axis = { catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent6, catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  titleFontFace: "+mn-lt", valAxisMaxVal: 115, valAxisMinVal: 0, valAxisMajorUnit: 25,
  valGridLine: { color: "D9E2E8", size: 0.75 }, catGridLine: { style: "none" }, showValue: true, dataLabelPosition: "outEnd",
  dataLabelFontSize: 10, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0\"%\"", showLegend: true, legendPos: "b",
  legendFontSize: 11, legendColor: HEX.dk1, showTitle: true, titleFontSize: 14, titleColor: HEX.dk2, barGapWidthPct: 60 };

s.addChart(pres.charts.BAR, [
  { name: "Agent alone", labels: ["QT4-1", "QT4-2", "QT4-3*"], values: [40, 0, 44] },
  { name: "Prompt self-check", labels: ["QT4-1", "QT4-2", "QT4-3*"], values: [100, 100, 56] },
  { name: "Gate: hint", labels: ["QT4-1", "QT4-2", "QT4-3*"], values: [100, 100, 44] },
  { name: "Gate: answer (0 calls)", labels: ["QT4-1", "QT4-2", "QT4-3*"], values: [100, 100, 44] },
], { ...axis, x: 0.5, y: 1.6, w: 6.5, h: 3.75, barDir: "col", title: "Impossible requests: correctly refused (%)",
  chartColors: [HEX.accent6, "B8C4CE", HEX.accent2, HEX.accent5], objectName: "chart impossible" });

label(s, "possible title", "Possible requests: done correctly", 7.25, 1.6, 5.6, 0.4, C.text2, { fs: 14, bold: true, align: "center" });
const ph = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 12, align: "center" } });
const pc = (t, o = {}) => ({ text: t, options: { fontSize: 12, color: C.text1, align: "center", ...o } });
s.addTable([
  [ph(""), ph("QT4-1"), ph("QT4-2"), ph("QT4-3")],
  [pc("Agent alone", { align: "left", bold: true }), pc("2/10"), pc("5/8"), pc("2/9")],
  [pc("Prompt self-check", { align: "left", bold: true }), pc("3/10"), pc("4/8"), pc("0/9")],
  [pc("Gate full v2.0", { align: "left", bold: true }), pc("5/10"), pc("4/8"), pc("1/9")],
  [pc("Gate full v2.1", { align: "left", bold: true }), pc("2/10", { bold: true }), pc("not run", { color: C.accent6, italic: true }),
    pc("not run", { color: C.accent6, italic: true })],
], { x: 7.25, y: 2.05, w: 5.6, colW: [1.85, 1.25, 1.25, 1.25], rowH: 0.4, valign: "middle",
  border: { type: "solid", pt: 0.75, color: "D9E2E8" }, fill: { color: C.background1 }, fontFace: "Calibri",
  margin: [2, 6, 2, 6], objectName: "possible table" });
label(s, "possible finding", [
  { text: "Why no gain yet: ", options: { bold: true, color: C.accent4 } },
  { text: "SimuHome's clock runs in real time and the agent keeps the clock from the episode start. " +
          "With a slow API (episodes take 6-13 min), QT4-3 pauses landed 1-5 min early in 8/8 cases and " +
          "QT4-1 times expired before scheduling. Of 8 QT4-1 failures under the gate, only 1 was the gate's fault." },
], 7.25, 4.2, 5.6, 1.15, C.text1, { fs: 11, valign: "top" });
label(s, "impossible footnote", "* QT4-3: 5 of 9 impossible episodes are device run-time conflicts, out of scope by design (max 4/9).",
  0.5, 5.3, 6.5, 0.25, C.accent6, { fs: 9.5, italic: true });

// stat callouts
const stats = [
  ["0", "large-model calls on contradictory requests in answer mode (18/18 correct, v2.1)"],
  ["1.0 vs 3.1", "calls per QT4-1 contradiction: hint mode vs the agent alone"],
  ["1-5 min", "how early agents paused in QT4-3: a stale-clock error the gate can catch (next fix)"],
];
stats.forEach(([big, small], i) => {
  const x = 0.5 + i * 4.15;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.55, w: 3.95, h: 1.3, rectRadius: 0.08, fill: { color: C.background2 },
    line: { color: C.background2 }, objectName: `stat card ${i + 1}` });
  label(s, `stat big ${i + 1}`, big, x + 0.2, 5.63, 1.6, 1.1, C.accent1, { fs: big.length > 4 ? 24 : 34, bold: true });
  label(s, `stat small ${i + 1}`, small, x + 1.85, 5.63, 1.95, 1.14, C.text1, { fs: 11 });
});
s.addNotes(
  "Left: impossible (self-contradictory) requests. The agent alone catches 4/10 QT4-1 and 0/8 QT4-2. Gate v2.1, " +
  "hint mode: 10/10 and 8/8 with 1.0 and 2.0 large-model calls; answer mode: 10/10 and 8/8 with zero calls. A one-line " +
  "self-check prompt is as accurate for this agent but costs 1.1 and 3.5 calls. QT4-3: 4/9 for both gate modes, which is " +
  "the maximum in scope (5 of 9 are device run-time conflicts). Right: possible requests. No method beats the agent " +
  "alone yet. The v2.0 'doubling' on QT4-1 (5/10) did not repeat in v2.1 (2/10): with n = 10 and a nondeterministic " +
  "hosted model, that is within noise. Trace analysis: SimuHome's clock runs in real time; the agent uses the time from " +
  "the start of the episode. With 6-13 minute episodes, QT4-3 pauses were 1.2-5 min early in all 8 scheduled cases, and " +
  "in QT4-1 several start times were already in the past when submitted. Of the 8 QT4-1 failures with the gate, 1 was " +
  "the gate's fault (a parser error made a possible request look impossible); the gate also fixed one wrong value. " +
  "Next: the gate's early-time rule will be extended to deadline goals so it refuses these early pauses, then the " +
  "missing QT4-2/QT4-3 runs.");

// =========================================================== slide 3: vs approach 1
pres.addSection({ title: "Approach 1 vs 2" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Approach 1 vs 2" });
s.addText("Why approach 2 is better than approach 1", { placeholder: "title" });
s.addText("Same small model idea, but it now checks earlier, checks the agent's actual plan, and explains its refusals",
  { placeholder: "subtitle" });

const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13 } });
const cell = (t, o = {}) => ({ text: t, options: { fontSize: 12, color: C.text1, ...o } });
const rows = [
  [hdr(""), hdr("Approach 1"), hdr("Approach 2")],
  [cell("When it checks", { bold: true }), cell("inside every schedule_workflow call"), cell("once, before the agent starts")],
  [cell("Contradictory request", { bold: true }), cell("agent retries times that can never pass: 4.3 calls (QT4-1)"),
    cell("1 call (hint) or 0 calls (answer), 10/10")],
  [cell("What it checks", { bold: true }), cell("only the request text"), cell("the request AND the agent's time and commands")],
  [cell("Agent's mistakes", { bold: true }), cell("wrong times and invented commands pass silently"),
    cell("refused with the fix: correct window, SimuHome's error, 'would change nothing'")],
  [cell("Possible requests", { bold: true }), cell("never checked: same as the agent alone (QT4-1: 2-3/10)"),
    cell("plan checked; no gain yet (QT4-1: 5/10 v2.0, 2/10 v2.1; n = 10)")],
  [cell("Time per episode", { bold: true }), cell("re-parses on every call: 377 s vs 218 s"), cell("parsed once, from a cache (CPU only)")],
];
s.addTable(rows, { x: 0.5, y: 1.6, w: 8.3, colW: [2.2, 2.9, 3.2], rowH: 0.62, valign: "middle",
  border: { type: "solid", pt: 0.75, color: "D9E2E8" }, fill: { color: C.background1 }, fontFace: "Calibri",
  margin: [3, 6, 3, 6], objectName: "comparison table" });

const why = [
  ["4.3 → 1 → 0", "large-model calls per contradiction: approach 1 → hint → answer", C.accent2],
  ["23 of 42", "failed episodes in which v2.1 flags the agent's mistake", C.accent3],
  ["0", "correct schedules blocked when replaying 151 real calls", C.accent5],
];
why.forEach(([big, small, col], i) => {
  const y = 1.6 + i * 1.75;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.15, y, w: 3.65, h: 1.55, rectRadius: 0.08, fill: { color: C.background2 },
    line: { color: C.background2 }, objectName: `why card ${i + 1}` });
  label(s, `why big ${i + 1}`, big, 9.35, y + 0.12, 3.3, 0.65, col, { fs: big.length > 6 ? 26 : 32, bold: true });
  label(s, `why small ${i + 1}`, small, 9.35, y + 0.8, 3.3, 0.62, C.text1, { fs: 12, valign: "top" });
});
s.addNotes(
  "Approach 1 checked the request inside every scheduling call and never looked at the agent's own time or commands. " +
  "On contradictions the agent kept retrying times that could never pass (QT4-1: 4.3 calls per episode); re-parsing on " +
  "every call made possible-request episodes 377 s instead of 218 s. Approach 2 decides once before the agent starts " +
  "(hint: about 1 call, answer: 0 calls), and checks the agent's actual plan: too-late times, failing commands and " +
  "workflows that would do nothing are refused with an explanation the agent can act on. Replaying all 151 real " +
  "scheduling calls from session 1 through gate v2.1: no correct schedule blocked, mistakes flagged in 23 of 42 failed " +
  "episodes. On possible requests there is no measured gain yet: run-to-run noise at n = 10 and API latency " +
  "(the simulator clock runs in real time) dominate; the next gate fix targets the latency-induced early pauses.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
