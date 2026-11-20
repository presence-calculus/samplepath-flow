import type { PresenceProcess } from "./process.js";

export type View = { horizon: number; lanes: number };
export type BlockView = "full" | "observed" | "contour";
export type Plot = { left: number; top: number; width: number; height: number };
type Context = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
type OutlineBlock = { left: number; right: number; top: number; bottom: number; landed: boolean };
const green = "#098478";
const blue = "#264e80";
const contourWidth = 2;
const remainingDash = [4, 3];
export const dropDuration = 0.35;

export function fitView(process: PresenceProcess, t: number, count: number, mode: BlockView): View {
  let extent = t;
  if (mode !== "observed") {
    for (let i = 0; i < count; i++) {
      extent = Math.max(extent, process.arrivals[i].end);
    }
  }
  const step = niceStep(Math.max(extent, 10) / 8);
  return {
    horizon: Math.min(Number.MAX_VALUE, Math.max(12, Math.ceil((extent + step * 0.38) / step) * step)),
    lanes: Math.max(8, Math.ceil((count + 2) / 4) * 4),
  };
}

export function plotBox(width: number, height: number, mode: BlockView): Plot {
  const left = width < 440 ? 42 : 61;
  const top = mode === "observed" ? 38 : 56;
  return { left, top, width: width - left - 22, height: height - top - 52 };
}

function niceStep(raw: number) {
  const base = 10 ** Math.floor(Math.log10(raw));
  return ([1, 2, 2.5, 5, 10].find(value => value * base >= raw) ?? 10) * base;
}

const format = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

export function number(value: number): string {
  return format.format(value);
}

export function renderChart(
  context: Context, process: PresenceProcess, t: number, view: View, width: number, height: number,
  options: { mode: BlockView; cursor?: number; reducedMotion?: boolean; stepDropAge?: number; title?: boolean },
) {
  const fullBlocks = options.mode !== "observed";
  const contour = options.mode === "contour";
  const plot = plotBox(width, height, options.mode);
  const { left, top } = plot;
  const right = left + plot.width;
  const bottom = top + plot.height;
  const laneHeight = plot.height / view.lanes;
  const x = (time: number) => left + time / view.horizon * plot.width;
  const count = process.countAt(t);

  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#14243b";
  context.textAlign = "left";
  context.textBaseline = "middle";
  context.font = "600 14px system-ui, sans-serif";
  context.fillText(options.title ? "Presence map  /  Arrival strip" : "Arrival strip", left, 19);

  if (width >= 435) {
    context.font = "12px system-ui, sans-serif";
    const labels = fullBlocks ? [[blue, "Observed"], [green, "Remaining"]]
      : [[green, "Present"], [blue, "Departed"]];
    let rightEdge = right;
    for (const [color, label] of labels.reverse()) {
      const labelX = rightEdge - context.measureText(label).width;
      if (contour) {
        context.save();
        context.strokeStyle = color;
        context.lineWidth = contourWidth;
        context.setLineDash(color === green ? remainingDash : []);
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
      rightEdge = labelX - (contour ? 32 : 27);
    }
  }

  context.fillStyle = "#52677e";
  context.font = "12px system-ui, sans-serif";
  const timeStep = niceStep(view.horizon / Math.max(3, Math.floor(plot.width / 94)));
  context.beginPath();
  context.textAlign = "center";
  for (let i = 0; i <= Math.floor(view.horizon / timeStep + 1e-9); i++) {
    const value = i * timeStep;
    const position = x(value);
    context.moveTo(position, top);
    context.lineTo(position, bottom);
    context.fillStyle = "#52677e";
    context.fillText(number(value), position, bottom + 16);
  }
  const laneStep = Math.max(1, Math.ceil(niceStep(view.lanes / Math.max(4, Math.floor(plot.height / 26)))));
  context.textAlign = "right";
  for (let strip = 1; strip <= view.lanes; strip += laneStep) {
    const position = bottom - (strip - 1) * laneHeight;
    context.moveTo(left, position);
    context.lineTo(right, position);
    context.fillText(number(strip), left - 9, position - laneHeight / 2);
  }
  context.strokeStyle = contour ? "#eff3f8" : "#e4eaf2";
  context.stroke();
  context.strokeStyle = "#9eafc3";
  context.strokeRect(left, top, plot.width, plot.height);
  context.textAlign = "center";
  context.fillStyle = "#324b67";
  context.font = "600 13px system-ui, sans-serif";
  context.fillText("Time (seconds)", left + plot.width / 2, height - 13);

  context.save();
  context.beginPath();
  const plotPadding = contour ? contourWidth / 2 : 0;
  context.rect(left - plotPadding, top - plotPadding, plot.width + 2 * plotPadding, plot.height + 2 * plotPadding);
  context.clip();
  const now = x(t);
  const barHeight = laneHeight - Math.min(1, laneHeight * 0.12);
  const drops: { center: number; left: number; width: number }[] = [];
  const outlines: OutlineBlock[] = [];

  for (let i = 0; i < count; i++) {
    const event = process.arrivals[i];
    const leftEdge = x(event.t);
    const split = x(Math.min(event.end, t));
    const barWidth = split - leftEdge;
    const center = bottom - (event.strip - 0.5) * laneHeight;
    let age = t - event.t;
    if (options.stepDropAge !== undefined) {
      age = event.t === t ? options.stepDropAge : dropDuration;
    }
    const dropping = !options.reducedMotion && age < dropDuration;

    if (contour) {
      const stripTop = bottom - event.strip * laneHeight;
      const blockTop = dropping ? top + (age / dropDuration) ** 2 * (stripTop - top) : stripTop;
      outlines.push({
        left: leftEdge,
        right: x(event.end),
        top: blockTop,
        bottom: dropping ? blockTop + laneHeight : bottom - (event.strip - 1) * laneHeight,
        landed: !dropping,
      });
      continue;
    }

    if (fullBlocks) {
      const entryCenter = top + barHeight / 2;
      const blockCenter = dropping
        ? entryCenter + (age / dropDuration) ** 2 * (center - entryCenter)
        : center;
      const blockTop = blockCenter - barHeight / 2;
      const remainingWidth = x(event.end) - split;
      if (barWidth > 0) {
        context.fillStyle = blue;
        context.fillRect(leftEdge, blockTop, barWidth, barHeight);
      }
      if (remainingWidth > 0) {
        context.fillStyle = green;
        context.fillRect(split, blockTop, remainingWidth, barHeight);
      }
      continue;
    }

    const active = t < event.end;
    context.fillStyle = active ? green : blue;
    if (barWidth > 0) {
      context.fillRect(leftEdge, center - barHeight / 2, barWidth, barHeight);
    } else if (options.reducedMotion) {
      context.fillRect(leftEdge - 1, center - barHeight / 2, 2, barHeight);
    }

    if (dropping) {
      const progress = age / dropDuration;
      const eased = progress * progress;
      drops.push({
        center: top + barHeight / 2 + eased * (center - top - barHeight / 2),
        left: leftEdge,
        width: Math.max(4, Math.min(20, barWidth)),
      });
    }
  }
  context.fillStyle = "#185dcb";
  for (const drop of drops) {
    const dropHeight = Math.min(16, barHeight);
    context.fillRect(drop.left, drop.center - dropHeight / 2, drop.width, dropHeight);
  }
  if (outlines.length) strokeContour(context, outlines, now, plot);
  context.beginPath();
  context.strokeStyle = "#185dcb";
  context.setLineDash([4, 4]);
  context.moveTo(now, top);
  context.lineTo(now, bottom);
  context.stroke();
  if (options.cursor !== undefined) {
    context.beginPath();
    context.strokeStyle = "#53647f";
    context.setLineDash([]);
    context.moveTo(x(options.cursor), top);
    context.lineTo(x(options.cursor), bottom);
    context.stroke();
  }
  context.restore();

  const nowLabel = `T = ${number(t)} s`;
  context.font = "600 12px system-ui, sans-serif";
  const labelWidth = context.measureText(nowLabel).width + 14;
  const labelX = Math.min(right - labelWidth, Math.max(left, now - labelWidth / 2));
  const labelY = fullBlocks ? top - 23 : top + 5;
  context.fillStyle = "#e9f1ff";
  context.fillRect(labelX, labelY, labelWidth, 20);
  context.fillStyle = "#184d9b";
  context.textAlign = "center";
  context.fillText(nowLabel, labelX + labelWidth / 2, labelY + 10);
}

function strokeContour(context: Context, blocks: OutlineBlock[], now: number, plot: Plot) {
  const path = new Path2D();
  for (let start = 0; start < blocks.length;) {
    // Consecutive arrivals occupy adjacent strips. Join only landed blocks with overlapping times.
    let end = start + 1;
    while (end < blocks.length && blocks[end - 1].landed && blocks[end].landed &&
           blocks[end].left < blocks[end - 1].right) {
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

  // Stroke the same perimeter on both sides, so the sweep adds no edge to the shape.
  const padding = contourWidth / 2;
  const sides = [
    { left: plot.left - padding, right: now, color: blue, dash: [] },
    { left: now, right: plot.left + plot.width + padding, color: green, dash: remainingDash },
  ];
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
