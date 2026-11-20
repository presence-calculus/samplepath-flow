import type { PresenceProcess } from "./process.js";

export type View = { horizon: number; lanes: number };
export type BlockView = "full" | "observed";
export type Plot = { left: number; top: number; width: number; height: number };
type Context = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
const green = "#098478";
const blue = "#78a6dc";
const dropDuration = 0.35;

export function fitView(process: PresenceProcess, t: number, count: number, mode: BlockView): View {
  let extent = t;
  if (mode === "full") {
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
  const top = mode === "full" ? 56 : 38;
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
  options: { mode: BlockView; cursor?: number; reducedMotion?: boolean; title?: boolean },
) {
  const fullBlocks = options.mode === "full";
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
      context.fillStyle = color;
      context.fillRect(labelX - 14, 15, 9, 9);
      context.fillStyle = "#52677e";
      context.fillText(label, labelX, 19);
      rightEdge = labelX - 27;
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
  context.strokeStyle = "#e4eaf2";
  context.stroke();
  context.strokeStyle = "#9eafc3";
  context.strokeRect(left, top, plot.width, plot.height);
  context.textAlign = "center";
  context.fillStyle = "#324b67";
  context.font = "600 13px system-ui, sans-serif";
  context.fillText("Time (seconds)", left + plot.width / 2, height - 13);

  context.save();
  context.beginPath();
  context.rect(left, top, plot.width, plot.height);
  context.clip();
  const now = x(t);
  const barHeight = laneHeight - Math.min(1, laneHeight * 0.12);
  const drops: { center: number; left: number; width: number }[] = [];

  for (let i = 0; i < count; i++) {
    const event = process.arrivals[i];
    const leftEdge = x(event.t);
    const split = x(Math.min(event.end, t));
    const barWidth = split - leftEdge;
    const center = bottom - (event.strip - 0.5) * laneHeight;
    const age = t - event.t;
    const dropping = !options.reducedMotion && age < dropDuration;

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
