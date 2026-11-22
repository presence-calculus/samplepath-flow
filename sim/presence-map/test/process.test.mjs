import assert from "node:assert/strict";
import test from "node:test";
import { fromJson, PresenceProcess } from "../src/process.ts";

const process = new PresenceProcess(fromJson([
  { t: 0, duration: 2 },
  { t: 1, duration: 6 },
  { t: 2, duration: 1 },
  { t: 4, duration: 6 },
  { t: 6, duration: 1 },
]));

test("counts carry-ins and arrivals that overlap the half-open window", () => {
  const { arrivalCount, carryIn, arrivals } = process.observeWindow(2, 6, false);
  assert.deepEqual({ arrivalCount, carryIn, rows: arrivals.map(({ strip, t }) => [strip, t]) }, {
    arrivalCount: 2,
    carryIn: 1,
    rows: [[1, 1], [2, 2], [3, 4]],
  });
});

test("accumulates full durations and clipped observed durations by selected strip", () => {
  const { arrivals, cumulativeObservedSojourns, sojournMeans } = process.observeWindow(2, 6, false);
  assert.deepEqual({
    full: arrivals.map(({ cumulativeSojourn }) => cumulativeSojourn),
    observed: cumulativeObservedSojourns,
    means: sojournMeans,
  }, {
    full: [6, 7, 13],
    observed: [4, 5, 7],
    means: { full: 13 / 3, observed: 7 / 3 },
  });
});

test("integrates the windowed presence steps to the same clipped presence mass", () => {
  const end = 6;
  const { presenceSteps, mass } = process.observeWindow(2, end, true);
  let area = 0;
  for (let i = 0; i < presenceSteps.length && presenceSteps[i].t < end; i++) {
    const step = presenceSteps[i];
    const nextTime = Math.min(end, presenceSteps[i + 1]?.t ?? end);
    area += step.present * (nextTime - step.t);
  }
  assert.equal(area, mass);
});

test("leaves duration means undefined for an empty window", () => {
  const { mass, sojournMeans, presenceSteps } = process.observeWindow(11, 12, true);
  assert.deepEqual({ mass, sojournMeans, presenceSteps }, { mass: 0, sojournMeans: undefined, presenceSteps: [] });
});
