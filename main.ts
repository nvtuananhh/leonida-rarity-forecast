// Leonida Rarity Forecast — GTA VI trophy earn-rate forecaster.
// Forecasts, for each of GTA VI's PS5 trophies, the Exophase earn rate at the
// case target as 5 quantiles (5/25/50/75/95%), scored by pinball loss on the
// logit scale.
//
// Model (dilution / mix-shift counts model, per trophy):
//   * Work in COUNTS: earned(t) = rate(t)/100 * tracked(t). Forecast earned
//     and tracked separately, then rate = earned / tracked at the target.
//   * Tracked: fit log(tracked) ~ log(1 + days since first read) by least
//     squares and extrapolate to the target -> R = tracked_target / tracked_cp.
//     Launch-week tracked growth is power-law-like in (1+d); the fit is
//     near-exact on smooth growth and R is the dilution denominator.
//   * Earned: earned_target = A * earned_cp + delta * rate_cp * (tracked_target
//     - tracked_cp). A = earned growth of the checkpoint cohort as existing
//     players keep progressing (large in launch week: early story chapters
//     rise fastest, grind/platinum barely move); delta = the rate at which
//     NEW tracked players earn the trophy, as a fraction of the checkpoint
//     rate (new players reach early chapters fast, almost never finish a
//     grind or the platinum). Rate ratio = (A + delta*(R-1)) / R: for large
//     R the rate tends to delta — dilution caps every rate, and the history
//     trend (early-adopter progression) is deliberately NOT extrapolated,
//     because after the checkpoint the tracked base floods in and the rate
//     flips down. A/delta per category were tuned on the fixed 10-case
//     synthetic final-like suite + p01 refresh scenarios (see improve log).
//   * Flat history (page unchanged, Exophase stale): no trend is observable,
//     so use the refresh-dilution prior (unchanged from the previous version).
//     Exophase refreshes in jumps: on launch night the page sat at 638
//     tracked for hours, then refreshed to 3,868 with every rate but the 100%
//     one lower. A refresh before the target is increasingly likely with
//     horizon, and when it lands tracked jumps (~x6 on the launch-night
//     refresh) while earned grows much less, so the median rate falls.
//     The 100% trophy is exempt: it did not fall.
//   * Spread (logit sd): random-walk floor 0.5*sqrt(days), plus a regime
//     component that grows slowly with log(R) (bigger projected dilution =
//     more regime uncertainty), plus a floor. The band is wide enough to
//     cover a post-checkpoint collapse or a continued rise, both of which
//     the history cannot distinguish.
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
const FLOOR_VAR = 0.2;
const S_MIN = 0.45;
const S_MAX = 2.0;

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

/**
 * Dilution (mix-shift) parameters per trophy category.
 * Counts model: earned_target = A * earned_cp + delta * rate_cp * (tracked_target - tracked_cp)
 *   A     = earned growth among the checkpoint cohort as existing players progress
 *           (early story chapters keep rising; grind/platinum barely move).
 *   delta = earn rate of NEW tracked players relative to the checkpoint rate
 *           (new players reach early chapters fast, almost never finish a grind).
 * Then rate_target / rate_cp = (A + delta * (R - 1)) / R, R = tracked_target / tracked_cp.
 * For large R the rate tends to delta: dilution caps how high any rate can go.
 */
function dilutionParams(description: string, tier: string): [number, number] {
  const d = description.toLowerCase();
  if (tier === "platinum") return [2.75, 0.024];
  if (d.includes("100% completion") || d.includes("final mission")) return [2.875, 0.032];
  if (d.includes("complete chapter 1") || d.includes("complete chapter 2")) return [5.5, 0.18];
  if (d.includes("complete chapter 3")) return [4.75, 0.14];
  if (d.includes("complete chapter 4") || d.includes("complete chapter")) return [4.25, 0.12];
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
    return [2.875, 0.032];
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
    return [4, 0.112];
  }
  if (
    d.includes("earn $") ||
    d.includes("spend $") ||
    d.includes("purchase all") ||
    d.includes("register 10") ||
    d.includes("discover")
  ) {
    return [3.5, 0.072];
  }
  return [3.75, 0.08];
}

function forecastCase(c: Case): { id: number; q: number[] }[] {
  const trophies = c.trophies;
  const hist = c.history;
  const ntr = trophies.length;
  const hDays = (c.horizon_minutes || 0) / 1440;
  const hHours = (c.horizon_minutes || 0) / 60;

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
  const tracked: number[] = hist.map((h) => h.players_tracked);
  const distinctTracked = new Set(tracked).size;
  const flat = distinctTracked === 1;

  // Tracked growth to target: fit log(tracked) ~ log(1 + days since first read)
  // (launch-week tracked growth is power-law-like in 1+d) and extrapolate.
  // R = tracked_target / tracked_cp. In the synthetic final-like cases this
  // fit is near-exact; on real data it is the dilution denominator.
  let trackR = 1;
  {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let k = 0; k < hist.length; k++) {
      if (tracked[k] > 0) {
        xs.push(Math.log(1 + times[k]));
        ys.push(Math.log(tracked[k]));
      }
    }
    const [b] = ols(xs, ys);
    if (b !== null && Number.isFinite(b)) {
      const g = Math.min(Math.max(b, 0), 4.5);
      const dCp = times[times.length - 1];
      const dTg = (Date.parse(c.target) - tFirst) / 86400000;
      if (dTg > dCp) {
        trackR = Math.pow((1 + dTg) / (1 + dCp), g);
        trackR = Math.min(Math.max(trackR, 1), 30);
      }
    }
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

    // Dilution model: forecast tracked (R) and earned separately. The history
    // trend is early-adopter progression and must NOT be extrapolated: after
    // the checkpoint the tracked base floods in and new players earn each
    // trophy at only a fraction (delta) of the checkpoint rate, while the
    // checkpoint cohort keeps progressing by factor A. rate ratio =
    // (A + delta * (R - 1)) / R.
    const [aOld, deltaNew] = dilutionParams(tr.description, tr.tier);
    const mult = (aOld + deltaNew * (trackR - 1)) / trackR;
    drift = Math.log(Math.min(Math.max(mult, 0.02), 3));
    drift = Math.min(Math.max(drift, -4), 2);
    const medLogit = logit(r) + drift;
    const rw = 0.5 * Math.sqrt(hDays);
    // Regime spread: the post-checkpoint regime (collapse vs continued rise)
    // is not observable in history, so the band must cover both. Wider when
    // tracked is projected to grow a lot (bigger dilution, more uncertainty).
    const sReg = 0.35 + 0.04 * Math.log(trackR);
    let s = Math.sqrt(rw * rw + sReg * sReg + FLOOR_VAR * FLOOR_VAR);
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
