// Leonida Rarity Forecast — GTA VI trophy earn-rate forecaster.
// Forecasts, for each of GTA VI's PS5 trophies, the Exophase earn rate at the
// case target as 5 quantiles (5/25/50/75/95%), scored by pinball loss on the
// logit scale.
//
// Model (counts-based, per trophy):
//   * Work in COUNTS: earned(t) = rate(t)/100 * tracked(t). Earned is monotone;
//     tracked grows. Forecast earned and tracked separately, then rate =
//     earned / tracked at the target.
//   * Fit log(tracked) and log(earned) against time by least squares, on the
//     full history AND on the recent 40% of the span. The relative drift
//     (gE - gT) is the logit's daily drift. When the recent drift is smaller
//     in magnitude than the full-history drift the process is decelerating
//     (rates saturate toward the share of players who ever earn the trophy),
//     so trust the recent drift; otherwise trust the full one.
//   * Shrink the fitted drift toward a category prior built from the trophy's
//     description (early story chapters rise as players progress; grind /
//     collectible / platinum trophies fall as tracked outruns earned).
//   * Extrapolate with a saturating effective horizon h_eff =
//     tau * (1 - exp(-h / tau)), tau = 4 days: a rate cannot drift
//     exponentially forever; it settles toward a steady state.
//   * Flat history (page unchanged, Exophase stale): no trend is observable,
//     so use a refresh-dilution prior. Exophase refreshes in jumps: on launch
//     night the page sat at 638 tracked for hours, then refreshed to 3,868
//     with every rate but the 100% one lower. A refresh before the target is
//     increasingly likely with horizon, and when it lands tracked jumps
//     (~x6 on the launch-night refresh) while earned grows much less, so the
//     median rate falls. The 100% trophy is exempt: it did not fall.
//   * Spread (logit sd): random-walk floor 0.5*sqrt(days), plus a refresh /
//     trend component, plus fit-residual uncertainty, plus a floor. The band
//     widens with horizon and is wide enough to cover plausible refreshes.
//
// Node 22, standard library only. No network, no randomness, no wall clock:
// everything derives from the case's own `at` / `target` / history, so two
// runs are byte-identical.

import * as fs from "node:fs";
import * as path from "node:path";

const Z: number[] = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

// Tuned on synthetic refresh scenarios + synthetic final-like backtests
// (see validation notes in README): the model beat the house baseline
// (practice loss ~1.51) and the house curve baseline on every refresh
// scenario and on the final-like backtests' mean / worst case.
const FLAT_JUMP = 1.8; // ln(3868/638): launch-night tracked jump on a refresh
const FLAT_EARNED_FRAC = 0.0; // earned counts assumed ~flat across a refresh
const FLAT_REFRESH_HOURS = 6.0; // refresh hazard time constant
const FLAT_SPREAD = 0.95;
const TREND_SPREAD = 0.35;
const FLOOR_VAR = 0.2;
const S_MIN = 0.45;
const S_MAX = 2.0;
const SAT_TAU_DAYS = 4.0;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface HistoryRead {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: (number | null)[];
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: unknown;
  trophies: Trophy[];
  history: HistoryRead[];
}

function clampRate(x: number): number {
  return Math.min(Math.max(x, 0.01), 99.99);
}

function logit(x: number): number {
  const p = clampRate(x) / 100;
  return Math.log(p / (1 - p));
}

function expit(l: number): number {
  if (l >= 0) return 100 / (1 + Math.exp(-l));
  const e = Math.exp(l);
  return (100 * e) / (1 + e);
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

/** Least squares y ~ a + b*x. Returns [slope, intercept, residualSd]. */
function ols(xs: number[], ys: number[]): [number | null, number, number] {
  const n = xs.length;
  if (n < 2) return [null, n === 1 ? ys[0] : 0, 0];
  let mx = 0;
  let my = 0;
  for (let k = 0; k < n; k++) {
    mx += xs[k];
    my += ys[k];
  }
  mx /= n;
  my /= n;
  let den = 0;
  let num = 0;
  for (let k = 0; k < n; k++) {
    den += (xs[k] - mx) * (xs[k] - mx);
    num += (xs[k] - mx) * (ys[k] - my);
  }
  if (den < 1e-12) return [null, my, 0];
  const b = num / den;
  const a = my - b * mx;
  let resid = 0;
  if (n > 2) {
    let ss = 0;
    for (let k = 0; k < n; k++) {
      const d = ys[k] - (a + b * xs[k]);
      ss += d * d;
    }
    resid = Math.sqrt(ss / n);
  }
  return [b, a, resid];
}

/**
 * Prior relative logit drift per day (gE - gT) when no trend is observable.
 * Negative = the rate falls as tracked outruns earned. Early story chapters
 * rise as the player base progresses through them; grind / collectible /
 * completion trophies fall; the platinum falls fastest.
 */
function categoryPrior(description: string, tier: string): number {
  const d = description.toLowerCase();
  if (tier === "platinum") return -0.45;
  if (d.includes("100% completion") || d.includes("final mission")) return -0.35;
  if (d.includes("complete chapter 1") || d.includes("complete chapter 2")) return 0.1;
  if (d.includes("complete chapter 3")) return 0.0;
  if (d.includes("complete chapter 4") || d.includes("complete chapter")) return -0.18;
  if (
    d.includes("250 stunts") ||
    d.includes("100 locations") ||
    d.includes("50 challenges") ||
    d.includes("100 vehicles") ||
    d.includes("card sets") ||
    d.includes("every spot") ||
    d.includes("every location") ||
    d.includes("8 mission strands") ||
    d.includes("gold medal in all") ||
    d.includes("win 20 different")
  ) {
    return -0.38;
  }
  if (
    d.includes("complete") &&
    (d.includes("mission") ||
      d.includes("run") ||
      d.includes("heist") ||
      d.includes("play") ||
      d.includes("takedown") ||
      d.includes("operation"))
  ) {
    return -0.16;
  }
  if (
    d.includes("earn $") ||
    d.includes("spend $") ||
    d.includes("purchase all") ||
    d.includes("register 10") ||
    d.includes("discover")
  ) {
    return -0.3;
  }
  return -0.2;
}

function forecastCase(c: Case): { id: number; q: number[] }[] {
  const trophies = c.trophies;
  const hist = c.history;
  const ntr = trophies.length;
  const hDays = (c.horizon_minutes || 0) / 1440;
  const hHours = (c.horizon_minutes || 0) / 60;
  const hEff =
    SAT_TAU_DAYS > 0 ? SAT_TAU_DAYS * (1 - Math.exp(-hDays / SAT_TAU_DAYS)) : hDays;

  // Last observed rate per trophy (0.5 fallback if never observed).
  const lastR: number[] = [];
  for (let i = 0; i < ntr; i++) {
    let r: number | null = null;
    for (let k = hist.length - 1; k >= 0; k--) {
      if (hist[k].rates[i] !== null && hist[k].rates[i] !== undefined) {
        r = hist[k].rates[i] as number;
        break;
      }
    }
    lastR.push(r === null ? 0.5 : r);
  }

  if (hist.length === 0) {
    // No history at all: last-rate anchor + wide spread.
    return trophies.map((t, i) => {
      const s = S_MAX;
      const med = logit(lastR[i]);
      return { id: t.id, q: Z.map((z) => round4(expit(med + z * s))) };
    });
  }

  const tFirst = Date.parse(hist[0].at);
  const times: number[] = hist.map((h) => (Date.parse(h.at) - tFirst) / 86400000);
  const spanDays = times.length > 1 ? times[times.length - 1] - times[0] : 0;
  const tracked: number[] = hist.map((h) => h.players_tracked);
  const distinctTracked = new Set(tracked).size;
  const flat = distinctTracked === 1;

  const allIdx: number[] = hist.map((_, k) => k);
  const recCut = times[times.length - 1] - 0.4 * spanDays;
  const recIdx: number[] = allIdx.filter((k) => times[k] >= recCut);

  function slopeOf(xs: number[], ys: number[]): [number | null, number] {
    const [b, , res] = ols(xs, ys);
    return [b, res];
  }

  function fitTracked(idxs: number[]): [number | null, number] {
    if (idxs.length < 2) return [null, 0];
    const xs = idxs.map((k) => times[k]);
    const ys = idxs.map((k) => Math.log(Math.max(tracked[k], 1)));
    return slopeOf(xs, ys);
  }

  const [gTFull] = fitTracked(allIdx);
  const [gTRec] = recIdx.length >= 3 ? fitTracked(recIdx) : [gTFull, 0];

  function fitEarned(i: number, idxs: number[]): [number | null, number] {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const k of idxs) {
      const r = hist[k].rates[i];
      if (r === null || r === undefined || r <= 0) continue;
      const e = (r / 100) * hist[k].players_tracked;
      if (e <= 0) continue;
      xs.push(times[k]);
      ys.push(Math.log(e));
    }
    if (xs.length < 3) return [null, 0];
    if (new Set(ys).size < 2) return [null, 0];
    return slopeOf(xs, ys);
  }

  const rows: { id: number; q: number[] }[] = [];
  for (let i = 0; i < ntr; i++) {
    const tr = trophies[i];
    const r = lastR[i];
    const priorRel = categoryPrior(tr.description, tr.tier);

    // The 100% trophy did not fall at the launch-night refresh; keep it ~100.
    if (r >= 99.995) {
      rows.push({
        id: tr.id,
        q: Z.map((z) => round4(expit(logit(99.99) + z * 0.35))),
      });
      continue;
    }

    let drift: number;
    let fitVar = 0.35;
    if (flat) {
      // No trend observable: refresh-dilution prior.
      const pRef = 1 - Math.exp(-hHours / FLAT_REFRESH_HOURS);
      const netJump = FLAT_JUMP * (1 - FLAT_EARNED_FRAC);
      drift = -netJump * pRef + priorRel * hDays * 0.3;
      drift = Math.min(Math.max(drift, -4), 2);
      const medLogit = logit(r) + drift;
      const rw = 0.5 * Math.sqrt(hDays);
      let s = Math.sqrt(rw * rw + FLAT_SPREAD * FLAT_SPREAD + FLOOR_VAR * FLOOR_VAR);
      s = Math.min(Math.max(s, S_MIN), S_MAX);
      rows.push({ id: tr.id, q: Z.map((z) => round4(expit(medLogit + z * s))) });
      continue;
    }

    const [gEFull, resEFull] = fitEarned(i, allIdx);
    const [gERec, resERec] = fitEarned(i, recIdx);
    if (gEFull === null || gTFull === null) {
      drift = priorRel * hEff;
      fitVar = 0.35;
    } else {
      const relFull = gEFull - gTFull;
      let relRec = relFull;
      if (gERec !== null && gTRec !== null) relRec = gERec - gTRec;
      // Deceleration (saturation): trust the recent drift when it is smaller.
      let rel = Math.abs(relRec) < Math.abs(relFull) ? relRec : relFull;
      const wFit = spanDays >= 1.5 ? 0.85 : spanDays >= 0.5 ? 0.65 : 0.5;
      rel = wFit * rel + (1 - wFit) * priorRel;
      drift = rel * hEff;
      fitVar = Math.max(resEFull, gERec !== null ? resERec : 0);
    }
    drift = Math.min(Math.max(drift, -4), 2);
    const medLogit = logit(r) + drift;
    const rw = 0.5 * Math.sqrt(hDays);
    const sRef = TREND_SPREAD * Math.sqrt(1 - Math.exp(-hDays / 1.5)) + 0.15;
    let sFit = Math.min(fitVar, 0.8) * Math.sqrt(Math.min(hDays, 2) / Math.max(spanDays, 0.25));
    sFit = Math.min(sFit, 0.9);
    let s = Math.sqrt(rw * rw + sRef * sRef + sFit * sFit + FLOOR_VAR * FLOOR_VAR);
    s = Math.min(Math.max(s, S_MIN), S_MAX);
    const qs = Z.map((z) => round4(expit(medLogit + z * s)));
    const clamped = qs.map((q) => Math.min(Math.max(q, 0), 100));
    for (let k = 1; k < 5; k++) {
      if (clamped[k] < clamped[k - 1]) clamped[k] = clamped[k - 1];
    }
    rows.push({ id: tr.id, q: clamped });
  }
  return rows;
}

/** Fallback forecast (last rate + random-walk spread) if a case cannot be modelled. */
function fallbackCase(c: Case): { id: number; q: number[] }[] {
  const hist = c.history;
  const d = (c.horizon_minutes || 0) / 1440;
  const s = 0.5 * Math.sqrt(d);
  return c.trophies.map((t, i) => {
    let r: number | null = null;
    for (let k = hist.length - 1; k >= 0; k--) {
      if (hist[k].rates[i] !== null && hist[k].rates[i] !== undefined) {
        r = hist[k].rates[i] as number;
        break;
      }
    }
    if (r === null) r = 0.5;
    const base = logit(r);
    return { id: t.id, q: Z.map((z) => round4(expit(base + z * s))) };
  });
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output/";
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const id = file.replace(/\.in$/, "");
    const raw = fs.readFileSync(path.join(inputDir, file), "utf8");
    const c = JSON.parse(raw) as Case;
    let rows: { id: number; q: number[] }[];
    try {
      rows = forecastCase(c);
    } catch {
      rows = fallbackCase(c);
    }
    const out = JSON.stringify({ trophies: rows });
    fs.writeFileSync(path.join(outputDir, `${id}.out`), out, "utf8");
  }
}

main();
