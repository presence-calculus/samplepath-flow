import { fitView, renderChart, type BlockView } from "./chart.js";
import { GifEncoder } from "./gif.js";
import { PresenceProcess, type Scenario } from "./process.js";

type Request = {
  scenario: Scenario; start: number; end: number; width: number; animated?: boolean; mode: BlockView;
  colorObserved: boolean; showPresence: boolean; showSojourn: boolean;
};

self.onmessage = (message: MessageEvent<Request>) => {
  try {
    const { scenario, start, end, width, animated = true, mode, colorObserved, showPresence, showSojourn } = message.data;
    const height = Math.round(width * 0.55);
    if (typeof OffscreenCanvas === "undefined") {
      throw new Error("GIF export requires a browser with OffscreenCanvas. Try a recent version of Chrome, Safari, or Firefox.");
    }
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("The browser could not start GIF rendering.");

    const replay = new PresenceProcess(scenario);
    replay.ensureAhead(end);
    const count = replay.countAt(end);
    const presenceSteps = showPresence ? replay.presenceSteps(count) : undefined;
    const view = fitView(replay, end, count, mode, presenceSteps, showSojourn);
    const duration = end - start;
    const intervals = animated ? Math.min(180, Math.max(2, Math.ceil(duration * 12))) : 0;
    const finalDelay = animated ? 90 : 0;
    const encoder = new GifEncoder(width, height, animated);
    for (let i = 0; i <= intervals; i++) {
      const final = i === intervals;
      const t = final ? end : start + duration * i / intervals;
      let delay = finalDelay;
      if (!final) {
        const elapsed = Math.round((t - start) * 100);
        const next = Math.round(duration * (i + 1) / intervals * 100);
        delay = Math.max(2, next - elapsed);
      }
      const cumulativeObservedSojourns = showSojourn ? replay.cumulativeObservedSojourns(t) : undefined;
      renderChart(context, replay, t, view, width, height, {
        mode, colorObserved, title: true, presenceSteps, cumulativeObservedSojourns,
      });
      encoder.addFrame(context.getImageData(0, 0, width, height).data, delay);
      if (animated && i % 8 === 0) {
        self.postMessage({ kind: "progress", percent: Math.round(i / intervals * 100) });
      }
    }
    const gif = encoder.finish();
    (self as unknown as DedicatedWorkerGlobalScope).postMessage({ kind: "done", gif }, [gif.buffer]);
  } catch (error) {
    self.postMessage({ kind: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
