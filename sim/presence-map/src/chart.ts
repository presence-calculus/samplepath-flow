import type { Arrival, ObservationWindow, PresenceProcess, PresenceStep, SojournMeans } from "./process.js";

type Viewport = { horizon: number; lanes: number; timeStart?: number; firstStrip?: number };
export type View = Viewport & {
  mirrorDepth: number; sojournHorizon: number; matchStripScale?: boolean;
  showAveragePresence?: boolean; showSojournAverages?: boolean;
};
export type BlockView = "full" | "observed" | "contour";
export type Plot = {
  left: number; top: number; width: number; height: number;
  origin: number; timeWidth: number; baseline: number; laneHeight: number; presenceUnitHeight: number;
};
type Context = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
type OutlineBlock = { strip: number; left: number; right: number; top: number; bottom: number; landed: boolean };
const green = "#098478";
const blue = "#264e80";
const presenceColor = "#a0442c";
const presenceFill = "#f8ebe5";
const averageColor = "#243751";
const sojournMeanColor = "#0f1c2e";
const observedSojournMeanColor = "#5187cf";
const averageLabelWidth = 66;
const averageLabelGap = 8;
const averageLabelPadding = 5;
const blockAxisX = 15;
const tickLabelGap = 9;
const contourWidth = 2;
const remainingDash = [4, 3];
const windowStartColor = "#755293";
export const dropDuration = 0.35;

function fitCameraViewport(arrivals: readonly Arrival[], count: number, t: number, previous?: Viewport): Viewport {
  let earliest = t;
  let latest = t;
  let firstActive: number | undefined;
  let lastActive: number | undefined;
  for (let i = 0; i < count; i++) {
    const event = arrivals[i];
    if (event.end <= t) continue;
    earliest = Math.min(earliest, event.t);
    latest = Math.max(latest, event.end);
    firstActive ??= event.strip;
    lastActive = event.strip;
  }

  const recentStrip = arrivals[count - 1]?.strip ?? 1;
  const first = firstActive ?? recentStrip;
  const last = lastActive ?? recentStrip;
  const activeSpan = latest - earliest;
  const targetHorizon = paddedHorizon(activeSpan * 1.2, 12);
  const targetLanes = Math.max(12, Math.ceil((last - first + 5) / 4) * 4);

  // Keep the previous scale through small changes and empty intervals.
  const retainScale = (target: number, prior?: number) =>
    prior !== undefined && (firstActive === undefined || target <= prior && target > prior * 0.65) ? prior : target;
  const horizon = retainScale(targetHorizon, previous?.horizon);
  const lanes = retainScale(targetLanes, previous?.lanes);
  const padding = Math.min(horizon * 0.08, (horizon - activeSpan) / 2);
  const leftBound = Math.max(0, latest - (horizon - padding));
  const rightBound = Math.max(leftBound, earliest - padding);
  const timeStart = Math.max(leftBound, Math.min(t - horizon * 0.68, rightBound));

  const previousFirst = previous?.firstStrip;
  const staysInView = previousFirst !== undefined && previous?.lanes === lanes &&
    (firstActive === undefined || first >= previousFirst && last < previousFirst + lanes - 1);
  const firstStrip = staysInView ? previousFirst : Math.max(1, Math.min(first - 1, last + 4 - lanes));
  return { horizon, lanes, timeStart, firstStrip };
}

function firstPresenceStep(steps: readonly PresenceStep[], start: number) {
  let low = 0;
  let high = steps.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (steps[middle].t < start) low = middle + 1;
    else high = middle;
  }
  return low;
}

export function fitView(
  process: PresenceProcess, t: number, count: number, mode: BlockView, presenceSteps?: readonly PresenceStep[],
  { showSojourn = false, matchStripScale = false, showAveragePresence = false, showSojournAverages = false,
    window, camera = false, previousCamera, averagePresence }: {
    showSojourn?: boolean; matchStripScale?: boolean; showAveragePresence?: boolean; showSojournAverages?: boolean;
    window?: ObservationWindow; camera?: boolean; previousCamera?: Viewport; averagePresence?: number;
  } = {},
): View {
  const arrivals = window?.arrivals ?? process.arrivals;
  const rowCount = window?.arrivals.length ?? count;
  let viewport: Viewport;
  if (camera) {
    viewport = fitCameraViewport(arrivals, rowCount, t, previousCamera?.firstStrip === undefined ? undefined : previousCamera);
  } else {
    const timeStart = window && (mode === "observed" ? window.start : Math.min(window.start, arrivals[0]?.t ?? window.start));
    let extent = t;
    if (mode !== "observed") {
      for (let i = 0; i < rowCount; i++) {
        extent = Math.max(extent, arrivals[i].end);
      }
    }
    viewport = {
      horizon: paddedHorizon(extent - (timeStart ?? 0), window ? 0.01 : 12),
      timeStart,
      lanes: Math.max(8, Math.ceil((rowCount + 2) / 4) * 4),
    };
  }
  let mirrorDepth = 0;
  if (presenceSteps) {
    const start = Math.max(viewport.timeStart ?? 0, window?.start ?? 0);
    const firstStep = firstPresenceStep(presenceSteps, start);
    mirrorDepth = Math.max(3, (presenceSteps[firstStep - 1]?.present ?? 0) + 1);
    if (showAveragePresence && averagePresence !== undefined) {
      mirrorDepth = Math.max(mirrorDepth, Math.ceil(averagePresence) + 1);
    }
    for (let i = firstStep; i < presenceSteps.length; i++) {
      const step = presenceSteps[i];
      if (step.t > t) break;
      mirrorDepth = Math.max(mirrorDepth, step.present + 1);
    }
  }
  const cumulativeSojourn = arrivals[rowCount - 1]?.cumulativeSojourn ?? 0;
  return {
    ...viewport,
    mirrorDepth,
    sojournHorizon: showSojourn ? paddedHorizon(cumulativeSojourn, 1) : 0,
    matchStripScale,
    showAveragePresence: Boolean(presenceSteps) && showAveragePresence,
    showSojournAverages: showSojourn && showSojournAverages,
  };
}

export function plotBox(width: number, height: number, mode: BlockView, view: View): Plot {
  const tickCharacters = Math.max(
    number((view.firstStrip ?? 1) + view.lanes - 1).length,
    view.mirrorDepth ? number(view.mirrorDepth).length + 1 : 0,
  );
  const tickMargin = blockAxisX + 14 + tickLabelGap + 8 * tickCharacters;
  const left = Math.max(width < 440 ? 42 : 61, tickMargin);
  const top = view.showSojournAverages ? 64 : mode === "observed" ? 38 : 56;
  const plotHeight = height - top - 76;
  const sharedUnitHeight = plotHeight / (view.lanes + view.mirrorDepth);
  const minimumMirrorHeight = view.mirrorDepth && !view.matchStripScale ? plotHeight / 3 : 0;
  const mirrorHeight = Math.max(view.mirrorDepth * sharedUnitHeight, minimumMirrorHeight);
  const laneHeight = (plotHeight - mirrorHeight) / view.lanes;
  const presenceUnitHeight = view.mirrorDepth ? mirrorHeight / view.mirrorDepth : 0;
  const baseline = top + plotHeight - mirrorHeight;
  const rightMargin = view.showAveragePresence ? averageLabelWidth + averageLabelGap + 4 : 22;
  const plotWidth = width - left - rightMargin;
  const timeWidth = view.sojournHorizon > 0 ? plotWidth * 0.6 : plotWidth;
  const origin = left + (plotWidth - timeWidth);
  return { left, top, width: plotWidth, height: plotHeight, origin, timeWidth, baseline, laneHeight, presenceUnitHeight };
}

function paddedHorizon(extent: number, minimum: number): number {
  const bounded = Math.min(extent, Number.MAX_VALUE);
  const step = niceStep(Math.max(bounded, minimum) / 8);
  return Math.min(Number.MAX_VALUE, Math.max(minimum, Math.ceil((bounded + step * 0.38) / step) * step));
}

function niceStep(raw: number) {
  const base = 10 ** Math.floor(Math.log10(raw));
  return ([1, 2, 2.5, 5, 10].find(value => value * base >= raw) ?? 10) * base;
}

const format = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const timestampFormat = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function number(value: number): string {
  return format.format(value);
}

export function renderChart(
  context: Context, process: PresenceProcess, t: number, view: View, width: number, height: number,
  options: {
    mode: BlockView; camera?: boolean; colorObserved?: boolean; cursor?: number; reducedMotion?: boolean; stepDropAge?: number;
    title?: boolean; presenceSteps?: readonly PresenceStep[]; averagePresence?: number;
    cumulativeObservedSojourns?: readonly number[]; sojournMeans?: SojournMeans; window?: ObservationWindow;
  },
) {
  const fullBlocks = options.mode !== "observed";
  const contour = options.mode === "contour";
  const { camera, colorObserved = true, presenceSteps, cumulativeObservedSojourns, window } = options;
  const averagePresence = presenceSteps ? options.averagePresence : undefined;
  const showSojourn = cumulativeObservedSojourns !== undefined;
  const compareAreas = Boolean(presenceSteps || showSojourn);
  const plot = plotBox(width, height, options.mode, view);
  const { left, top, origin, timeWidth, baseline, laneHeight, presenceUnitHeight } = plot;
  const averageLine = averagePresence === undefined ? undefined : {
    y: baseline + averagePresence * presenceUnitHeight,
    label: number(averagePresence),
  };
  const right = left + plot.width;
  const bottom = top + plot.height;
  const timeStart = view.timeStart ?? 0;
  const firstStrip = view.firstStrip ?? 1;
  const stripOffset = firstStrip - 1;
  const x = (time: number) => origin + (time - timeStart) / view.horizon * timeWidth;
  const sojournX = (seconds: number) => origin - Math.min(seconds / view.sojournHorizon, 1) * (origin - left);
  const arrivals = window?.arrivals ?? process.arrivals;
  const count = window?.arrivals.length ?? process.countAt(t);
  const sojournLines: { x: number; label: string; color: string; dash: number[] }[] = [];
  const sojournMeans = showSojourn ? options.sojournMeans : undefined;
  if (sojournMeans) {
    sojournLines.push({
      x: sojournX(sojournMeans.full), label: `Full avg ${number(sojournMeans.full)} s`,
      color: sojournMeanColor, dash: [],
    });
    if (colorObserved && sojournMeans.observed !== undefined) {
      sojournLines.push({
        x: sojournX(sojournMeans.observed), label: `Observed avg ${number(sojournMeans.observed)} s`,
        color: observedSojournMeanColor, dash: remainingDash,
      });
    }
  }

  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  context.textAlign = "left";
  context.textBaseline = "middle";
  let legendStart = left;
  if (options.title) {
    const title = "Presence map";
    context.fillStyle = "#14243b";
    context.font = "600 14px system-ui, sans-serif";
    context.fillText(title, left, 19);
    legendStart += context.measureText(title).width + 12;
  }

  if (sojournLines.length) {
    context.save();
    context.font = "600 11px system-ui, sans-serif";
    context.lineWidth = 2;
    context.fillStyle = "#324b67";
    let y = top - 25;
    for (const line of sojournLines) {
      context.strokeStyle = line.color;
      context.setLineDash(line.dash);
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(left + 14, y);
      context.stroke();
      context.fillText(line.label, left + 19, y, right - left - 19);
      y += 14;
    }
    context.restore();
  }

  if (width >= 435 || presenceSteps) {
    context.font = "12px system-ui, sans-serif";
    const labels: [color: string, label: string, dashed?: boolean][] = [];
    const unobservedLabel = window ? "Outside window" : "Remaining";
    if (width >= 435) {
      if (contour) labels.push([colorObserved ? blue : green, "Observed"], [green, unobservedLabel, true]);
      else if (!colorObserved) labels.push([green, fullBlocks ? "Full duration" : "Observed"]);
      else if (fullBlocks) labels.push([blue, "Observed"], [green, unobservedLabel]);
      else {
        labels.push([green, "Present"]);
        if (!camera) labels.push([blue, "Departed"]);
      }
    }
    if (presenceSteps) labels.push([presenceColor, "−N(t)"]);
    if (averageLine) labels.push([averageColor, window ? "Window avg." : "L(T)", true]);
    let rightEdge = right;
    for (const [color, label, dashed] of labels.reverse()) {
      const labelX = rightEdge - context.measureText(label).width;
      const lineSample = contour || color === presenceColor || dashed;
      if (labelX - (lineSample ? 22 : 14) < legendStart) break;
      if (lineSample) {
        context.save();
        context.strokeStyle = color;
        context.lineWidth = contourWidth;
        context.setLineDash(dashed ? remainingDash : []);
        context.beginPath();
        context.moveTo(labelX - 22, 19);
        context.lineTo(labelX - 5, 19);
        context.stroke();
        context.restore();
      } else {
        context.fillStyle = color;
        context.fillRect(labelX - 14, 15, 9, 9);
      }
      context.fillStyle = "#52677e";
      context.fillText(label, labelX, 19);
      rightEdge = labelX - (lineSample ? 32 : 27);
    }
  }

  context.fillStyle = "#52677e";
  context.font = "12px system-ui, sans-serif";
  const presenceLabelX = (showSojourn ? origin : left) - tickLabelGap;
  const presenceTickCount = Math.floor((bottom - baseline) / 20);
  const presenceStep = presenceSteps && presenceTickCount > 0
    ? Math.max(1, Math.ceil(niceStep(view.mirrorDepth / presenceTickCount)))
    : 0;
  const durationLabelRight = showSojourn && presenceStep > 0
    ? presenceLabelX - context.measureText(`−${number(view.mirrorDepth)}`).width - 6
    : origin;
  const durationTickRight = presenceStep * presenceUnitHeight < 30 ? durationLabelRight : origin;
  const timeStep = niceStep(view.horizon / Math.max(3, Math.floor(timeWidth / 94)));
  const firstTickOffset = (timeStep - timeStart % timeStep) % timeStep;
  context.beginPath();
  context.textAlign = "center";
  for (let i = 0; firstTickOffset + i * timeStep <= view.horizon + timeStep * 1e-9; i++) {
    const offset = firstTickOffset + i * timeStep;
    const position = origin + offset / view.horizon * timeWidth;
    context.moveTo(position, top);
    context.lineTo(position, bottom);
    context.fillStyle = "#52677e";
    context.fillText(number(timeStart + offset), position, bottom + 16);
  }
  if (showSojourn) {
    const durationStep = niceStep(view.sojournHorizon / Math.max(2, Math.floor((origin - left) / 68)));
    let previousLabel = origin;
    for (let i = 1; i <= Math.floor(view.sojournHorizon / durationStep + 1e-9); i++) {
      const value = i * durationStep;
      const position = sojournX(value);
      context.moveTo(position, top);
      context.lineTo(position, baseline);
      const label = number(value);
      if (previousLabel - position >= 24 && position + context.measureText(label).width / 2 <= durationTickRight) {
        context.fillText(label, position, baseline + 16);
        previousLabel = position;
      }
    }
    if (previousLabel === origin && left < durationTickRight) {
      context.fillText(number(view.sojournHorizon), left, baseline + 16, 2 * (durationTickRight - left));
    }
  }
  const stripTickCount = Math.max(4, Math.floor((baseline - top) / 26));
  const laneStep = Math.max(1, Math.ceil(niceStep(view.lanes / stripTickCount)));
  context.textAlign = "right";
  for (let lane = 0; lane < view.lanes; lane += laneStep) {
    const position = baseline - lane * laneHeight;
    context.moveTo(left, position);
    context.lineTo(right, position);
    const labelY = position - laneHeight / 2;
    if (!presenceSteps || labelY <= baseline - 13) context.fillText(number(firstStrip + lane), left - tickLabelGap, labelY);
  }
  if (presenceSteps) {
    const shiftedSojourn = firstStrip > 1 && showSojourn;
    context.fillText("0", shiftedSojourn ? presenceLabelX : left - tickLabelGap, baseline + (shiftedSojourn ? 8 : 0));
    if (presenceStep > 0) {
      for (let level = presenceStep; level <= view.mirrorDepth; level += presenceStep) {
        const position = baseline + level * presenceUnitHeight;
        context.moveTo(origin, position);
        context.lineTo(right, position);
        context.fillText(`−${number(level)}`, presenceLabelX, position);
      }
    }
  }
  context.strokeStyle = contour ? "#eff3f8" : "#e4eaf2";
  context.stroke();
  context.strokeStyle = "#9eafc3";
  context.strokeRect(origin, top, timeWidth, plot.height);
  if (showSojourn) context.strokeRect(left, top, origin - left, baseline - top);
  if (presenceSteps) {
    context.beginPath();
    context.moveTo(left, baseline);
    context.lineTo(right, baseline);
    context.stroke();
  }
  context.textAlign = "center";
  context.fillStyle = "#324b67";
  context.font = "600 13px system-ui, sans-serif";
  context.fillText("Time (seconds)", origin + timeWidth / 2, bottom + 39);
  if (showSojourn) {
    const title = "Total Duration";
    const titleWidth = Math.min(context.measureText(title).width, Math.max(1, durationLabelRight - blockAxisX));
    const titleX = Math.min((left + origin) / 2, durationLabelRight - titleWidth / 2);
    context.fillText(title, titleX, baseline + 39, titleWidth);
  }

  context.save();
  context.translate(blockAxisX, top + plot.height / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("Blocks", 0, 0);
  context.restore();

  context.save();
  context.beginPath();
  const plotPadding = presenceSteps ? 2 : contour ? contourWidth / 2 : 0;
  context.rect(left - plotPadding, top - plotPadding, plot.width + 2 * plotPadding, plot.height + 2 * plotPadding);
  context.clip();
  const now = x(t);
  const windowStartX = window && window.start >= timeStart ? x(window.start) : undefined;
  const observedFrom = windowStartX ?? origin;
  const presenceStart = Math.max(timeStart, window?.start ?? 0);
  const barHeight = compareAreas ? laneHeight : laneHeight - Math.min(1, laneHeight * 0.12);
  const drops: { center: number; left: number; width: number }[] = [];
  const outlines: OutlineBlock[] = [];

  const visibleEnd = Math.min(count, stripOffset + view.lanes);
  for (let i = stripOffset; i < visibleEnd; i++) {
    const event = arrivals[i];
    const lane = event.strip - stripOffset;
    const center = baseline - (lane - 0.5) * laneHeight;
    if (cumulativeObservedSojourns) {
      const fullStart = sojournX(event.cumulativeSojourn);
      const stripTop = center - laneHeight / 2;
      context.fillStyle = green;
      context.fillRect(fullStart, stripTop, origin - fullStart, laneHeight);
      if (colorObserved) {
        const observedStart = sojournX(cumulativeObservedSojourns[i]);
        context.fillStyle = blue;
        context.fillRect(observedStart, stripTop, origin - observedStart, laneHeight);
      }
    }
    // Camera only affects blocks on the time chart.
    if (camera && event.end <= t) continue;

    const leftEdge = x(event.t);
    const observedEdge = window ? Math.max(leftEdge, observedFrom) : leftEdge;
    const split = x(Math.min(event.end, t));
    const barWidth = split - observedEdge;
    let age = t - event.t;
    if (options.stepDropAge !== undefined) {
      age = event.t === t ? options.stepDropAge : dropDuration;
    }
    const dropping = !options.reducedMotion && age < dropDuration;
    const easing = dropping ? (age / dropDuration) ** 2 : 1;
    const movingBlock = dropping && !compareAreas;

    if (dropping && compareAreas) {
      // Keep measured blocks in their strips so the falling guide cannot obscure their area.
      const entryCenter = top + barHeight / 2;
      drops.push({
        center: entryCenter + easing * (center - entryCenter),
        left: fullBlocks ? leftEdge : observedEdge,
        width: fullBlocks ? x(event.end) - leftEdge : Math.max(4, Math.min(20, barWidth)),
      });
    }

    if (contour) {
      const stripTop = baseline - lane * laneHeight;
      const blockTop = movingBlock ? top + easing * (stripTop - top) : stripTop;
      outlines.push({
        strip: event.strip,
        left: leftEdge,
        right: x(event.end),
        top: blockTop,
        bottom: movingBlock ? blockTop + laneHeight : baseline - (lane - 1) * laneHeight,
        landed: !movingBlock,
      });
      continue;
    }

    if (fullBlocks) {
      const entryCenter = top + barHeight / 2;
      const blockCenter = movingBlock
        ? entryCenter + easing * (center - entryCenter)
        : center;
      const blockTop = blockCenter - barHeight / 2;
      const greenStart = colorObserved ? split : leftEdge;
      const greenWidth = x(event.end) - greenStart;
      if (colorObserved && observedEdge > leftEdge) {
        context.fillStyle = green;
        context.fillRect(leftEdge, blockTop, observedEdge - leftEdge, barHeight);
      }
      if (colorObserved && barWidth > 0) {
        context.fillStyle = blue;
        context.fillRect(observedEdge, blockTop, barWidth, barHeight);
      }
      if (greenWidth > 0) {
        context.fillStyle = green;
        context.fillRect(greenStart, blockTop, greenWidth, barHeight);
      }
      continue;
    }

    const active = t < event.end;
    context.fillStyle = colorObserved && !active ? blue : green;
    if (barWidth > 0) {
      context.fillRect(observedEdge, center - barHeight / 2, barWidth, barHeight);
    } else if (options.reducedMotion && !compareAreas) {
      context.fillRect(leftEdge - 1, center - barHeight / 2, 2, barHeight);
    }

    if (movingBlock) {
      drops.push({
        center: top + barHeight / 2 + easing * (center - top - barHeight / 2),
        left: observedEdge,
        width: Math.max(4, Math.min(20, barWidth)),
      });
    }
  }
  if (outlines.length) {
    const contourStart = window && window.start > timeStart ? observedFrom : undefined;
    strokeContour(context, outlines, now, plot, colorObserved ? blue : green, contourStart);
  }
  context.save();
  context.fillStyle = colorObserved ? "#185dcb" : green;
  context.strokeStyle = "#9eafc3";
  context.setLineDash([3, 3]);
  for (const drop of drops) {
    const dropHeight = compareAreas && fullBlocks ? barHeight : Math.min(16, barHeight);
    const dropTop = drop.center - dropHeight / 2;
    if (compareAreas) context.strokeRect(drop.left, dropTop, drop.width, dropHeight);
    else context.fillRect(drop.left, dropTop, drop.width, dropHeight);
  }
  context.restore();
  if (sojournLines.length) {
    context.save();
    context.beginPath();
    context.rect(left, top, origin - left, baseline - top);
    context.clip();
    // Draw the white outlines first so both styles remain visible when the averages coincide.
    context.beginPath();
    for (const line of sojournLines) {
      context.moveTo(line.x, top);
      context.lineTo(line.x, baseline);
    }
    context.strokeStyle = "#ffffff";
    context.lineWidth = 4;
    context.setLineDash([]);
    context.stroke();
    context.lineWidth = 2;
    for (const line of sojournLines) {
      context.beginPath();
      context.moveTo(line.x, top);
      context.lineTo(line.x, baseline);
      context.strokeStyle = line.color;
      context.setLineDash(line.dash);
      context.stroke();
    }
    context.restore();
  }
  if (presenceSteps) {
    drawPresenceMirror(context, presenceSteps, t, plot, x, presenceStart);
  }
  if (averageLine) {
    context.save();
    context.beginPath();
    context.strokeStyle = averageColor;
    context.lineWidth = 2;
    context.setLineDash(remainingDash);
    context.moveTo(observedFrom, averageLine.y);
    context.lineTo(now, averageLine.y);
    context.stroke();
    context.restore();
  }
  context.beginPath();
  context.strokeStyle = "#185dcb";
  context.setLineDash([4, 4]);
  context.moveTo(now, top);
  context.lineTo(now, bottom);
  context.stroke();
  if (windowStartX !== undefined) {
    context.beginPath();
    context.strokeStyle = windowStartColor;
    context.setLineDash([4, 4]);
    context.moveTo(windowStartX, top);
    context.lineTo(windowStartX, bottom);
    context.stroke();
  }
  if (options.cursor !== undefined) {
    context.beginPath();
    context.strokeStyle = "#53647f";
    context.setLineDash([]);
    context.moveTo(x(options.cursor), top);
    context.lineTo(x(options.cursor), bottom);
    context.stroke();
  }
  context.restore();

  if (windowStartX !== undefined) {
    context.fillStyle = windowStartColor;
    context.font = "600 12px system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText("S", windowStartX, top - 9);
  }
  if (averageLine) {
    const labelLeft = now + averageLabelGap;
    const textWidth = averageLabelWidth - 2 * averageLabelPadding;
    context.font = "600 12px system-ui, sans-serif";
    const boxWidth = Math.min(context.measureText(averageLine.label).width, textWidth) + 2 * averageLabelPadding;
    context.fillStyle = "#eff3f8";
    context.fillRect(labelLeft, averageLine.y - 9, boxWidth, 18);
    context.fillStyle = averageColor;
    context.textAlign = "left";
    context.fillText(averageLine.label, labelLeft + averageLabelPadding, averageLine.y, textWidth);
  }

  const labelWidth = 124;
  const labelX = right - labelWidth;
  const labelY = height - 27;
  context.fillStyle = "#e9f1ff";
  context.fillRect(labelX, labelY, labelWidth, 20);
  context.fillStyle = "#184d9b";
  context.font = "600 12px system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("T =", labelX + 7, labelY + 10);
  context.textAlign = "right";
  context.fillText("s", right - 7, labelY + 10);
  context.font = "600 12px ui-monospace, monospace";
  context.fillText(timestampFormat.format(t), right - 21, labelY + 10, 70);
}

function drawPresenceMirror(
  context: Context, steps: readonly PresenceStep[], t: number, plot: Plot, x: (time: number) => number,
  start: number,
) {
  const firstStep = firstPresenceStep(steps, start);
  const left = x(start);
  const path = new Path2D();
  let level = plot.baseline + (steps[firstStep - 1]?.present ?? 0) * plot.presenceUnitHeight;
  path.moveTo(left, level);
  for (let i = firstStep; i < steps.length; i++) {
    const step = steps[i];
    if (step.t > t) break;
    const at = x(step.t);
    path.lineTo(at, level);
    level = plot.baseline + step.present * plot.presenceUnitHeight;
    path.lineTo(at, level);
  }
  const now = x(t);
  path.lineTo(now, level);
  const area = new Path2D(path);
  area.lineTo(now, plot.baseline);
  area.lineTo(left, plot.baseline);
  area.closePath();

  context.save();
  context.fillStyle = presenceFill;
  context.fill(area);
  context.lineJoin = "round";
  context.strokeStyle = presenceColor;
  context.lineWidth = 2;
  context.stroke(path);
  context.restore();
}

function strokeContour(
  context: Context, blocks: OutlineBlock[], now: number, plot: Plot, observedColor: string, observedFrom?: number,
) {
  const path = new Path2D();
  for (let start = 0; start < blocks.length;) {
    // Join only landed blocks with overlapping times in adjacent strips.
    let end = start + 1;
    while (end < blocks.length && blocks[end - 1].landed && blocks[end].landed &&
           blocks[end].strip === blocks[end - 1].strip + 1 && blocks[end].left < blocks[end - 1].right) {
      end++;
    }

    path.moveTo(blocks[start].left, blocks[start].bottom);
    for (let i = start; i < end; i++) {
      const block = blocks[i];
      path.lineTo(block.right, block.bottom);
      path.lineTo(block.right, block.top);
    }
    for (let i = end - 1; i >= start; i--) {
      const block = blocks[i];
      path.lineTo(block.left, block.top);
      path.lineTo(block.left, block.bottom);
    }
    path.closePath();
    start = end;
  }

  // Clip the strokes to the observation boundaries without adding edges to the perimeter.
  const padding = contourWidth / 2;
  const left = plot.origin - padding;
  const sides = [
    { left: observedFrom ?? left, right: now, color: observedColor, dash: [] },
    { left: now, right: plot.left + plot.width + padding, color: green, dash: remainingDash },
  ];
  if (observedFrom !== undefined) {
    sides.push({ left, right: observedFrom, color: green, dash: remainingDash });
  }
  for (const side of sides) {
    context.save();
    context.beginPath();
    context.rect(side.left, plot.top - padding, side.right - side.left, plot.height + 2 * padding);
    context.clip();
    context.strokeStyle = side.color;
    context.lineWidth = contourWidth;
    context.setLineDash(side.dash);
    context.stroke(path);
    context.restore();
  }
}
