import assert from "node:assert/strict";
import test from "node:test";
import { fitView, plotBox, renderChart } from "../src/chart.ts";
import { fromJson, PresenceProcess } from "../src/process.ts";

const process = new PresenceProcess(fromJson([
  { t: 1, duration: 7 },
  { t: 2, duration: 2 },
  { t: 3, duration: 6 },
]));

globalThis.Path2D = class {
  closedShapes = 0;
  points = [];
  moveTo(x, y) { this.points.push([x, y]); }
  lineTo(x, y) { this.points.push([x, y]); }
  closePath() { this.closedShapes++; }
};

function render(mode, t, camera, window, replay = process, followCamera = false) {
  const width = 800;
  const height = 400;
  const count = replay.countAt(t);
  const presenceSteps = followCamera ? replay.presenceSteps(count) : undefined;
  const view = fitView(replay, t, count, mode, presenceSteps, { camera: followCamera, showSojourn: true, window });
  const plot = plotBox(width, height, mode, view);
  const blocks = [];
  const cumulativeBars = [];
  const contours = [];
  let presencePath;
  const context = new Proxy({
    measureText(text) { return { width: text.length * 6 }; },
    fillRect(x, y, width, height) {
      if (y < plot.top || y >= plot.baseline) return;
      (x >= plot.origin ? blocks : cumulativeBars).push([x, y, width, height]);
    },
    stroke(path) {
      if (path && this.strokeStyle === "#a0442c") presencePath = path.points;
      else if (path) contours.push(path.closedShapes);
    },
  }, {
    get(target, name) { return target[name] ?? (() => {}); },
  });
  renderChart(context, replay, t, view, width, height, {
    mode,
    camera,
    colorObserved: false,
    reducedMotion: true,
    cumulativeObservedSojourns: window?.cumulativeObservedSojourns ?? replay.cumulativeObservedSojourns(t),
    presenceSteps,
    window,
  });
  return { plot, blocks, cumulativeBars, contours, presencePath, view };
}

test("Camera preserves the full block geometry before departure", () => {
  assert.deepEqual(render("full", 3.5, true), render("full", 3.5, false));
});

for (const mode of ["full", "observed"]) {
  test(`Camera removes the departed block at its right edge in ${mode} view`, () => {
    const normal = render(mode, 4, false);
    assert.deepEqual(render(mode, 4, true), { ...normal, blocks: [normal.blocks[0], normal.blocks[2]] });
  });
}

test("Camera splits the contour around an empty strip between surviving blocks", () => {
  const normal = render("contour", 4, false);
  assert.deepEqual(render("contour", 4, true), { ...normal, contours: [2, 2] });
});

test("Camera retains all cumulative duration bars within a finite observation window", () => {
  const window = process.observeWindow(3, 5, false);
  const normal = render("full", 5, false, window);
  assert.deepEqual(render("full", 5, true, window), { ...normal, blocks: [normal.blocks[0], normal.blocks[2]] });
});

const longBlock = new PresenceProcess(fromJson([
  { t: 4, duration: 40 },
  ...Array.from({ length: 35 }, (_, i) => ({ t: 5 + i, duration: 1 })),
]));

test("Camera fits both the oldest active block and the latest arrivals", () => {
  const view = fitView(longBlock, 39, 36, "contour", undefined, { camera: true });
  assert.ok(view.timeStart <= 4 && view.timeStart + view.horizon >= 44 &&
    view.firstStrip === 1 && view.lanes >= 36);
});

test("Camera keeps its scale and follows the sweep through an empty interval", () => {
  const previousCamera = fitView(longBlock, 39, 36, "full", undefined, { camera: true });
  const empty = fitView(longBlock, 90, 36, "full", undefined, { camera: true, previousCamera });
  assert.ok(empty.timeStart > previousCamera.timeStart && empty.timeStart <= 90 &&
    empty.timeStart + empty.horizon >= 90 && empty.horizon === previousCamera.horizon &&
    empty.firstStrip === previousCamera.firstStrip && empty.lanes === previousCamera.lanes);
});

const lateBlocks = new PresenceProcess(fromJson([
  ...Array.from({ length: 105 }, (_, i) => ({ t: i, duration: 1.25 })),
  { t: 105, duration: 7 },
  { t: 108, duration: 2 },
]));

test("Camera pans to absolute strip numbers and renders only the visible duration rows", () => {
  const { view, cumulativeBars } = render("full", 109, true, undefined, lateBlocks, true);
  assert.ok(view.firstStrip > 90 && view.firstStrip <= 106 && view.firstStrip + view.lanes > 107 &&
    cumulativeBars.length === 108 - view.firstStrip && view.sojournHorizon >= 140.25);
});

test("Camera starts the presence curve at the existing level on the left edge", () => {
  const { view, plot, presencePath } = render("full", 109, true, undefined, lateBlocks, true);
  const prior = lateBlocks.measure(view.timeStart).present;
  assert.ok(view.timeStart > 95 && prior > 0 && presencePath[0][0] === plot.origin &&
    presencePath[0][1] === plot.baseline + prior * plot.presenceUnitHeight &&
    presencePath.every(([x]) => x >= plot.origin));
});
