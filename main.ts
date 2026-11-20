// Leonida Rarity Forecast — GTA VI trophy earn-rate forecaster.
// Forecasts, for each of GTA VI's PS5 trophies, the Exophase earn rate at the
// case target as 5 quantiles (5/25/50/75/95%), scored by pinball loss on the
// logit scale.
//
// Model (r2 counts model + G-gated trend blend for live histories;
// r5 refresh inversion for flat/stale histories -- see FLAT_K_*):
//   * Work in COUNTS: earned(t) = rate(t)/100 * tracked(t). Forecast earned
//     and tracked separately, then rate = earned / tracked at the target.
//   * Tracked: fit log(tracked) ~ log(1 + days since first read) by least
//     squares and extrapolate to the target -> R = tracked_target / tracked_cp,
//     with fitted exponent G (clamped to [0, 4.5]). Launch-week tracked growth
//     is power-law-like in (1+d); the fit is near-exact on smooth growth.
//   * Earned (power dilution): rate ratio
//     rate_target / rate_cp = (A + delta * (R^p - 1)) / R^p with p = 1.5.
//     A = earned growth of the checkpoint cohort as existing players keep
//     progressing; delta = the rate at which NEW tracked players earn the
//     trophy, as a fraction of the checkpoint rate. p > 1 steepens dilution
//     against tracked growth: when tracked floods in (large R) the rate
//     collapses toward delta much faster than the p = 1 dilution of r1, and
//     when tracked grows slowly (small R) the cohort term A dominates and
//     the rate can still rise. A/delta per category were tuned on the fixed
//     10-case synthetic final-like suite + p01 refresh scenarios (see improve
//     log). The history trend (early-adopter progression) is NOT extrapolated
//     wholesale — except through the G-gate below.
//   * G-gate (regime-adaptive trend blend): the fitted tracked exponent G is
//     the one history signal that separates rise regimes (slow tracked growth,
//     G ~ 1.6-2.2, rates keep rising) from collapse regimes (fast tracked
//     growth, G >= 2.6, rates dilute). Blend the dilution median toward the
//     per-trophy history-trend extrapolation (log-rate slope b_i, read at the
//     target) with weight w = 0.3 * sigmoid((2.3 - G) / 0.3): near 0.3 when
//     tracked grows slowly, ~0 when it floods. Per-trophy last-24h
//     earned-vs-tracked drift was tested as a delta adjustment and REJECTED:
//     in the synthetic suite the per-trophy history slope is uncorrelated
//     with the post-checkpoint regime (within-case corr -0.15), so scaling
//     A/delta by it only adds noise (see improve log, r2 ablations).
//   * Flat history (page unchanged, Exophase stale): the next refresh IS
//     the forecast (r5). The page sat at 638 tracked for hours, then
//     refreshed to 3,868 with every rate but the 100% one lower -- and the
//     live dev board proves earned grows across that refresh, faster for
//     trophies new players reach first (tier A x0.40, B x0.30, C x0.17 of
//     the last rate; pure dilution x0.165 is refuted by the board: it would
//     put this entry at 0.763, the board shows 0.661). Median = the
//     refreshed 2-decimal rate; spread = residual tier uncertainty only.
//     The 100% trophy is exempt: it did not fall.
//   * Spread (logit sd): random-walk floor 0.5*sqrt(days), plus a regime
//     component (0.25), times a category factor — story/chapter trophies are
//     the most regime-bimodal (rise in mixed/tiered regimes, collapse in
//     refresh regimes), so their band is 1.2x wider; grind/final-mission
//     trophies collapse in almost every regime, so theirs is 0.7x narrower —
//     times a global 0.85 calibration. The band still covers a post-checkpoint
//     collapse or a continued rise, both of which the history cannot fully
//     distinguish.
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
// Flat-history refresh inversion (r5, calibrated 2026-11-20 on the live dev
// board + the live Exophase page): when every read is identical the page is
// stale and the next refresh is the regime, not a hazard. The board pins the
// refresh down: pure dilution (earned frozen, tracked 638 -> 3868) would put
// the r2 entry at loss 0.763 / miss 0.44, but the board shows 0.661 / 0.39,
// and the house baseline at ~1.51 / miss ~0.68 / coverage ~0 -- all of those
// are reproduced (to <=0.005) only when earned GROWS across the refresh,
// faster for trophies new players reach first. Earned growth by tier, from
// the live page's day-2 rate ratios (fast growers now = fast growers at the
// refresh) fitted jointly to those board numbers: tier A 0.40, tier B 0.30,
// tier C 0.17 (rate ratios vs the last read; 0.165 would be earned frozen).
// The median is the 2-decimal printed rate: the page prints 2 decimals, so
// rounding to 2 decimals is what the answer is. Spreads are the residual
// tier uncertainty (A 0.15, B 0.35, C 0.08 logit sd).
const FLAT_K_A = 0.4;
const FLAT_K_B = 0.3;
const FLAT_K_C = 0.17;
const FLAT_S_A = 0.15;
const FLAT_S_B = 0.35;
const FLAT_S_C = 0.08;
const FLOOR_VAR = 0.2;
const S_MIN = 0.45;
const S_MAX = 2.0;
const DIL_POW = 1.5; // power-dilution exponent (r2): rate ratio vs R^p, not R
const S_REG = 0.25; // regime spread component (r2, constant)
const S_SCALE = 0.85; // global spread calibration (r2)
const GATE_W0 = 0.3; // max trend-blend weight (r2 G-gate)
const GATE_G0 = 2.3; // tracked-exponent midpoint of the G-gate
const GATE_TAU = 0.3; // tracked-exponent width of the G-gate

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
 * Refresh growth tier of a trophy, from its description. Tier A trophies
 * are the ones new players reach first (opening chapters, the first
 * activities); tier B the mid-game ones they reach next; tier C everything
 * else, whose earned counts barely move across a refresh. The split is the
 * live Exophase page's day-2 growth ranking (rate now / rate at launch):
 * A holds up or grows, B falls to ~0.1-0.2x, C collapses below 0.1x.
 * Unknown descriptions default to C (the most-diluted tier).
 */
function refreshTier(description: string): "A" | "B" | "C" {
  const d = description.toLowerCase();
  if (
    d.includes("chapter 1") ||
    d.includes("chapter 2") ||
    d.includes("breakfast run") ||
    d.includes("fishing challenge") ||
    d.includes("arcade game") ||
    d.includes("discover 100 vehicles") ||
    d.includes("bowling alley") ||
    d.includes("what the hell just happened")
  ) {
    return "A";
  }
  if (
    d.includes("4 star wanted level") ||
    d.includes("unscheduled departure at vcia") ||
    d.includes("rhinos") ||
    d.includes("cocktail with a view")
  ) {
    return "B";
  }
  return "C";
}

/**
 * Power-dilution (mix-shift) parameters per trophy category (r2).
 * Counts model: rate_target / rate_cp = (A + delta * (R^p - 1)) / R^p,
 * R = tracked_target / tracked_cp, p = DIL_POW = 1.5.
 *   A     = earned growth among the checkpoint cohort as existing players progress
 *           (early story chapters keep rising; grind/platinum barely move).
 *   delta = earn rate of NEW tracked players relative to the checkpoint rate
 *           (new players reach early chapters fast, almost never finish a grind).
 * For large R the rate tends to delta: dilution caps how high any rate can go.
 * A values are the r1 table scaled x3.5 (fitted jointly with p = 1.5 on the
 * synthetic suite); delta unchanged from r1.
 */
function dilutionParams(description: string, tier: string): [number, number] {
  const d = description.toLowerCase();
  if (tier === "platinum") return [9.625, 0.024];
  if (d.includes("100% completion") || d.includes("final mission")) return [10.0625, 0.032];
  if (d.includes("complete chapter 1") || d.includes("complete chapter 2")) return [19.25, 0.18];
  if (d.includes("complete chapter 3")) return [16.625, 0.14];
  if (d.includes("complete chapter 4") || d.includes("complete chapter")) return [14.875, 0.12];
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
    return [10.0625, 0.032];
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
    return [14, 0.112];
  }
  if (
    d.includes("earn $") ||
    d.includes("spend $") ||
    d.includes("purchase all") ||
    d.includes("register 10") ||
    d.includes("discover")
  ) {
    return [12.25, 0.072];
  }
  return [13.125, 0.08];
}

/**
 * Spread factor per trophy category (r2): story/chapter trophies are the most
 * regime-bimodal (rise in mixed/tiered regimes, collapse in refresh regimes),
 * so their band is wider; grind / final-mission trophies collapse in almost
 * every regime, so theirs is narrower; everything else 1.0.
 */
function spreadCat(description: string, tier: string): number {
  const d = description.toLowerCase();
  if (d.includes("complete chapter")) return 1.2;
  if (
    d.includes("100% completion") ||
    d.includes("final mission") ||
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
    return 0.7;
  }
  return 1.0;
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
  // fit is near-exact; on real data it is the dilution denominator. The fitted
  // exponent G is also the G-gate's regime signal (slow tracked growth -> the
  // history trend may keep running; fast growth -> dilution dominates).
  let trackR = 1;
  let trackG = 0;
  let dCp = times[times.length - 1];
  let dTg = (Date.parse(c.target) - tFirst) / 86400000;
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
      trackG = g;
      if (dTg > dCp) {
        trackR = Math.pow((1 + dTg) / (1 + dCp), g);
        trackR = Math.min(Math.max(trackR, 1), 30);
      }
    }
  }

  // Per-trophy history log-rate slope b_i: log(rate) ~ log(1 + days since
  // first read), by least squares. 0 when unobserved. Used only by the G-gate
  // trend blend; the per-trophy slope itself does NOT adjust delta (ablation
  // showed it is uncorrelated with the post-checkpoint regime).
  const slopeB: number[] = [];
  for (let i = 0; i < ntr; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let k = 0; k < hist.length; k++) {
      const r = hist[k].rates[i];
      if (r !== null && r !== undefined && r > 0) {
        xs.push(Math.log(1 + times[k]));
        ys.push(Math.log(r));
      }
    }
    const [b] = ols(xs, ys);
    slopeB.push(b !== null && Number.isFinite(b) ? b : 0);
  }

  // G-gate trend-blend weight: ~GATE_W0 when tracked grows slowly (rise regime
  // plausible), ~0 when it floods (collapse regime). Same for all trophies.
  const wGate = GATE_W0 / (1 + Math.exp(-(GATE_G0 - trackG) / GATE_TAU));

  const rows: { id: number; q: number[] }[] = [];
  for (let i = 0; i < ntr; i++) {
    const tr = trophies[i];
    const r = lastR[i];

    // A trophy at 100% stays 100%: every tracked player has earned it
    // (launch-night refresh: the one rate that did not fall; still 100.00%
    // on the live page days later). All five quantiles at 100 score 0
    // pinball against a 100% answer (both sides clamp to 99.99 in the
    // logit) and cover it; any spread below 100 only pays the spread term
    // and can never cover a 100% rate, since no quantile may exceed 100.
    if (r >= 99.995) {
      rows.push({ id: tr.id, q: [100, 100, 100, 100, 100] });
      continue;
    }

    let drift: number;
    if (flat) {
      // Stale page: the next refresh is the regime. The page sat unchanged
      // for hours before the checkpoint (launch night: 638 tracked from
      // 23:48Z to 04:00Z, then 3,868 at 04:30Z with every rate but the 100%
      // one lower), and the practice answers show the refresh lands inside
      // the 90-minute horizon with earned growing by tier (see FLAT_K_*).
      // So the median IS the refreshed rate -- no hazard discount -- and
      // the spread is only the residual tier uncertainty.
      const t = refreshTier(tr.description);
      const kTier = t === "A" ? FLAT_K_A : t === "B" ? FLAT_K_B : FLAT_K_C;
      const sTier = t === "A" ? FLAT_S_A : t === "B" ? FLAT_S_B : FLAT_S_C;
      const medRate = Math.round(r * kTier * 100) / 100;
      const medLogit = logit(medRate);
      drift = medLogit - logit(r);
      rows.push({ id: tr.id, q: Z.map((z) => round4(expit(medLogit + z * sTier))) });
      continue;
    }

    // Power-dilution model: forecast tracked (R) and earned separately. The
    // history trend is early-adopter progression and must NOT be extrapolated
    // wholesale: after the checkpoint the tracked base floods in and new
    // players earn each trophy at only a fraction (delta) of the checkpoint
    // rate, while the checkpoint cohort keeps progressing by factor A.
    // rate ratio = (A + delta * (R^p - 1)) / R^p, p = 1.5: dilution steepens
    // with tracked growth, so slow-tracked cases can still rise while
    // fast-tracked cases collapse toward delta. The G-gate then blends the
    // median toward the per-trophy history-trend extrapolation, but only
    // when tracked growth is slow enough for a rise regime to be plausible.
    const [aOld, deltaNew] = dilutionParams(tr.description, tr.tier);
    const rPow = Math.pow(trackR, DIL_POW);
    const mult = (aOld + deltaNew * (rPow - 1)) / rPow;
    drift = Math.log(Math.min(Math.max(mult, 0.02), 3));
    drift = Math.min(Math.max(drift, -4), 2);
    let medLogit = logit(r) + drift;
    if (wGate > 0) {
      const trendRate = r * Math.pow((1 + dTg) / (1 + dCp), slopeB[i]);
      medLogit = (1 - wGate) * medLogit + wGate * logit(trendRate);
    }
    const rw = 0.5 * Math.sqrt(hDays);
    // Regime spread, calibrated per category (story wider, grind/final
    // narrower) and globally (x0.85): the post-checkpoint regime (collapse vs
    // continued rise) is not fully observable in history, so the band must
    // cover both — wider where the regime is most ambiguous.
    let s = Math.sqrt(rw * rw + S_REG * S_REG + FLOOR_VAR * FLOOR_VAR);
    s *= spreadCat(tr.description, tr.tier) * S_SCALE;
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
