import type { PresenceProcess } from "./process.js";

export type View = { horizon: number; lanes: number };
export type Plot = { left: number; top: number; width: number; height: number };
type Context = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

export function fitView(t: number, count: number): View {
  const step = niceStep(Math.max(t, 10) / 8);
  return {
    horizon: Math.max(12, Math.ceil((t + step * 0.38) / step) * step),
    lanes: Math.max(8, Math.ceil((count + 2) / 4) * 4),
  };
}

export function plotBox(width: number, height: number): Plot {
  const left = width < 440 ? 42 : 61;
  return { left, top: 38, width: width - left - 22, height: height - 90 };
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
  options: { cursor?: number; reducedMotion?: boolean; title?: boolean } = {},
) {
  const plot = plotBox(width, height);
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
    context.fillStyle = "#098478";
    context.fillRect(right - 149, 15, 9, 9);
    context.fillStyle = "#52677e";
    context.fillText("Present", right - 135, 19);
    context.fillStyle = "#78a6dc";
    context.fillRect(right - 77, 15, 9, 9);
    context.fillStyle = "#52677e";
    context.fillText("Departed", right - 63, 19);
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
    const barWidth = x(Math.min(event.end, t)) - leftEdge;
    const center = bottom - (event.strip - 0.5) * laneHeight;
    const active = t < event.end;
    context.fillStyle = active ? "#098478" : "#78a6dc";
    if (barWidth > 0) {
      context.fillRect(leftEdge, center - barHeight / 2, barWidth, barHeight);
    } else if (options.reducedMotion) {
      context.fillRect(leftEdge - 1, center - barHeight / 2, 2, barHeight);
    }

    const age = t - event.t;
    if (!options.reducedMotion && age < 0.35) {
      const progress = age / 0.35;
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
  context.fillStyle = "#e9f1ff";
  context.fillRect(labelX, top + 5, labelWidth, 20);
  context.fillStyle = "#184d9b";
  context.textAlign = "center";
  context.fillText(nowLabel, labelX + labelWidth / 2, top + 15);
}
