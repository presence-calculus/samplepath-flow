import { dropDuration, fitView, number, plotBox, renderChart, type BlockView } from "./chart.js";
import { defaultScenario, fromJson, PresenceProcess, simulation, type Scenario } from "./process.js";

function element<T extends HTMLElement = HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

const canvas = element<HTMLCanvasElement>("map");
const context = canvas.getContext("2d")!;
const tooltip = element<HTMLOutputElement>("tooltip");
const scenarioDialog = element<HTMLDialogElement>("scenario-dialog");
const exportDialog = element<HTMLDialogElement>("export-dialog");
const notice = element("notice");
const play = element<HTMLButtonElement>("play");
const next = element<HTMLButtonElement>("step");
const source = element("source");
const speed = element<HTMLSelectElement>("speed");
const viewSelect = element<HTMLSelectElement>("block-view");
const colorObserved = element<HTMLInputElement>("color-observed");
const showPresence = element<HTMLInputElement>("show-presence");
const showAveragePresence = element<HTMLInputElement>("show-average-presence");
const matchStripScale = element<HTMLInputElement>("match-strip-scale");
const showSojourn = element<HTMLInputElement>("show-sojourn");
const rateInput = element<HTMLInputElement>("rate");
const rateOutput = element<HTMLOutputElement>("rate-value");
const blockCountInput = element<HTMLInputElement>("block-count");
const runStatus = element("run-status");
const timeOutput = element("time").firstChild!;
const countOutput = element("count");
const presentOutput = element("present");
const massOutput = element("mass").firstChild!;
const arrivalRateCurrent = element("arrival-rate-current");
const sojournMeanCurrent = element("sojourn-mean-current");
const averagePresenceCurrent = element("average-presence-current");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

let scenario: Scenario = readScenario();
let process = new PresenceProcess(scenario);
let time = 0;
let blockView: BlockView = "full";
let running = !reducedMotion.matches;
let stepDropStartedAt: number | undefined;
let previousFrame: number | undefined;
let previousDraw = -Infinity;
let lastMetrics = -Infinity;
let dirty = true;
let cursor: { x: number; y: number } | undefined;
let noticeTimer: number | undefined;
let gifUrl: string | undefined;
let worker: Worker | undefined;

function readScenario(): Scenario {
  if (!location.hash) return defaultScenario;
  try {
    const data = JSON.parse(decodeURIComponent(location.hash.slice(1)));
    if (data.kind === "simulation") return simulation(data.seed, data.rate, data.mean, data.blockCount);
    if (data.kind === "file") return fromJson(data.events);
  } catch {
    // An invalid hash should still open the demo.
  }
  return defaultScenario;
}

function syncControls() {
  play.textContent = running ? "Pause" : "Play";
  runStatus.classList.toggle("paused", !running);
  runStatus.lastChild!.textContent = running ? " Running" : " Paused";
  source.textContent = scenario.kind === "simulation" ? "Simulated arrivals" : "Imported arrivals";
  dirty = true;
}

function syncScenarioTargets() {
  const settings = scenario.kind === "simulation" ? scenario : undefined;
  element("arrival-rate-target").textContent = settings ? number(settings.rate) : "—";
  element("sojourn-mean-target").textContent = settings ? number(settings.mean) : "—";
  element("average-presence-target").textContent = settings ? number(settings.rate * settings.mean) : "—";
}

function reset() {
  process = new PresenceProcess(scenario);
  time = 0;
  stepDropStartedAt = undefined;
  syncControls();
}

function useScenario(value: Scenario) {
  scenario = value;
  syncScenarioTargets();
  if (location.hash) history.replaceState(null, "", location.pathname + location.search);
  running = true;
  reset();
  scenarioDialog.close();
}

function draw(stepDropAge?: number) {
  const { width, height } = canvas.getBoundingClientRect();
  if (width < 1 || height < 1) return;
  const dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);

  process.ensureAhead(time);
  const completionTime = process.completionTime;
  if (running && completionTime !== undefined && time >= completionTime) {
    time = completionTime;
    running = false;
    syncControls();
  }
  const count = process.countAt(time);
  const presenceSteps = showPresence.checked ? process.presenceSteps(count) : undefined;
  const updateMetrics = performance.now() - lastMetrics > 100 || dirty;
  const showAverage = showPresence.checked && showAveragePresence.checked;
  const hasAverage = showAverage && time > 0;
  const metrics = updateMetrics || hasAverage ? process.measure(time, count) : undefined;
  const averagePresence = time > 0 && metrics ? metrics.mass / time : undefined;
  const cumulativeObservedSojourns = showSojourn.checked ? process.cumulativeObservedSojourns(time, count) : undefined;
  const view = fitView(process, time, count, blockView, presenceSteps, {
    showSojourn: showSojourn.checked,
    matchStripScale: matchStripScale.checked,
    showAveragePresence: showAverage,
  });
  const plot = plotBox(width, height, blockView, view);
  let slice: number | undefined;
  if (cursor && cursor.x >= plot.origin && cursor.x <= plot.left + plot.width &&
      cursor.y >= plot.top && cursor.y <= plot.top + plot.height) {
    const t = (cursor.x - plot.origin) / plot.timeWidth * view.horizon;
    if (t <= time) slice = t;
  }
  renderChart(context, process, time, view, width, height, {
    mode: blockView,
    colorObserved: colorObserved.checked,
    cursor: slice,
    presenceSteps,
    averagePresence: showAverage ? averagePresence : undefined,
    cumulativeObservedSojourns,
    stepDropAge,
    reducedMotion: reducedMotion.matches ||
      (!running && stepDropAge === undefined && process.arrivals[count - 1]?.t === time),
  });

  if (updateMetrics && metrics) {
    const { present, mass } = metrics;
    timeOutput.textContent = `${number(time)} `;
    countOutput.textContent = number(count);
    presentOutput.textContent = number(present);
    massOutput.textContent = `${number(mass)} `;
    arrivalRateCurrent.textContent = time > 0 ? number(count / time) : "—";
    sojournMeanCurrent.textContent = count > 0 ? number(process.arrivals[count - 1].cumulativeSojourn / count) : "—";
    averagePresenceCurrent.textContent = averagePresence === undefined ? "—" : number(averagePresence);
    next.disabled = !process.nextAfter(time);
    lastMetrics = performance.now();
  }

  let tooltipText = "";
  if (cursor) {
    const onStrip = cursor.y >= plot.top && cursor.y <= plot.baseline;
    const strip = onStrip ? Math.min(view.lanes, Math.floor((plot.baseline - cursor.y) / plot.laneHeight) + 1) : 0;
    const event = process.arrivals[strip - 1];
    if (slice !== undefined) {
      const { present } = process.measure(slice);
      tooltipText = `t = ${number(slice)} s · N(t) = ${present}`;
      if (event && slice >= event.t && slice < event.end) {
        tooltipText += `\nStrip ${strip} · ${number(event.t)}–${number(event.end)} s`;
      }
    } else if (cumulativeObservedSojourns && event && event.t <= time && cursor.x >= plot.left && cursor.x < plot.origin) {
      const duration = (plot.origin - cursor.x) / (plot.origin - plot.left) * view.sojournHorizon;
      if (duration <= event.cumulativeSojourn) {
        const observed = cumulativeObservedSojourns[strip - 1];
        tooltipText = `Strip ${strip} · ${number(event.cumulativeSojourn)} s cumulative sojourn` +
          `\n${number(observed)} s observed through this strip`;
      }
    }
  }
  if (cursor && tooltipText) {
    tooltip.textContent = tooltipText;
    tooltip.hidden = false;
    tooltip.style.left = `${Math.max(4, Math.min(cursor.x + 13, width - tooltip.offsetWidth - 6))}px`;
    tooltip.style.top = `${Math.max(4, Math.min(cursor.y + 13, height - tooltip.offsetHeight - 6))}px`;
  } else {
    tooltip.hidden = true;
  }
  dirty = false;
}

function tick(timestamp: number) {
  if (running && previousFrame !== undefined) {
    time += Math.min(0.1, (timestamp - previousFrame) / 1000) * Number(speed.value);
  }
  previousFrame = timestamp;
  let stepDropAge: number | undefined;
  if (stepDropStartedAt !== undefined) {
    const elapsed = Math.max(0, (timestamp - stepDropStartedAt) / 1000);
    if (elapsed < dropDuration) {
      stepDropAge = elapsed;
    } else {
      stepDropStartedAt = undefined;
      dirty = true;
    }
  }
  if (dirty || ((running || stepDropAge !== undefined) && timestamp - previousDraw >= 25)) {
    draw(stepDropAge);
    previousDraw = timestamp;
  }
  requestAnimationFrame(tick);
}

function notify(message: string) {
  notice.textContent = message;
  notice.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = window.setTimeout(() => { notice.hidden = true; }, 3300);
}

canvas.addEventListener("pointermove", event => {
  const rect = canvas.getBoundingClientRect();
  cursor = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  dirty = true;
});
canvas.addEventListener("pointerleave", () => { cursor = undefined; dirty = true; });
new ResizeObserver(() => { dirty = true; }).observe(canvas);
reducedMotion.addEventListener("change", () => {
  if (reducedMotion.matches) stepDropStartedAt = undefined;
  dirty = true;
});
function syncColorNotes() {
  element("block-colors").hidden = blockView !== "full" || !colorObserved.checked;
  const contourColors = element("contour-colors");
  contourColors.hidden = blockView !== "contour";
  contourColors.textContent = colorObserved.checked
    ? "Solid blue shows observed presence. Dashed green shows remaining duration."
    : "Solid strokes show observed presence. Dashed strokes show remaining duration.";
  dirty = true;
}

viewSelect.addEventListener("change", () => {
  const mode = viewSelect.value;
  blockView = mode === "contour" || mode === "observed" ? mode : "full";
  syncColorNotes();
});
colorObserved.addEventListener("change", syncColorNotes);
function syncPresenceControls() {
  showAveragePresence.disabled = !showPresence.checked;
  element("presence-scale").hidden = !showPresence.checked;
  element("presence-comparison").hidden = !showPresence.checked;
  dirty = true;
}
showPresence.addEventListener("change", syncPresenceControls);
showAveragePresence.addEventListener("change", () => {
  dirty = true;
});
matchStripScale.addEventListener("change", () => {
  dirty = true;
});
showSojourn.addEventListener("change", () => {
  element("sojourn-comparison").hidden = !showSojourn.checked;
  dirty = true;
});

play.addEventListener("click", () => {
  running = !running;
  if (running && time === process.completionTime) {
    reset();
    return;
  }
  stepDropStartedAt = undefined;
  syncControls();
});
next.addEventListener("click", () => {
  const arrival = process.nextAfter(time);
  if (!arrival) return;
  time = arrival.t;
  running = false;
  stepDropStartedAt = reducedMotion.matches ? undefined : performance.now();
  syncControls();
});
element("reset").addEventListener("click", reset);

document.querySelectorAll<HTMLButtonElement>("[data-close]").forEach(button => {
  button.addEventListener("click", () => element<HTMLDialogElement>(button.dataset.close!).close());
});

function showRate() {
  const text = `${rateInput.valueAsNumber} arrivals/s`;
  rateOutput.value = text;
  rateInput.setAttribute("aria-valuetext", text);
}

rateInput.addEventListener("input", () => {
  rateInput.value = rateInput.valueAsNumber.toFixed(2);
  showRate();
});

element("settings").addEventListener("click", () => {
  const settings = scenario.kind === "simulation" ? scenario : defaultScenario;
  element<HTMLInputElement>("seed").value = String(settings.seed);
  rateInput.value = String(settings.rate);
  showRate();
  element<HTMLInputElement>("duration").value = String(settings.mean);
  blockCountInput.value = settings.blockCount?.toString() ?? "";
  element("scenario-error").hidden = true;
  scenarioDialog.showModal();
});

function showScenarioError(error: unknown) {
  const output = element("scenario-error");
  output.textContent = error instanceof Error ? error.message : String(error);
  output.hidden = false;
}

element("scenario-form").addEventListener("submit", event => {
  event.preventDefault();
  try {
    useScenario(simulation(
      element<HTMLInputElement>("seed").valueAsNumber,
      rateInput.valueAsNumber,
      element<HTMLInputElement>("duration").valueAsNumber,
      blockCountInput.value === "" ? undefined : blockCountInput.valueAsNumber,
    ));
  } catch (error) {
    showScenarioError(error);
  }
});

element<HTMLInputElement>("stream-file").addEventListener("change", async event => {
  const field = event.currentTarget as HTMLInputElement;
  const file = field.files?.[0];
  if (!file) return;
  try {
    useScenario(fromJson(JSON.parse(await file.text())));
    notify(`Loaded ${file.name}.`);
  } catch (error) {
    showScenarioError(error);
  } finally {
    field.value = "";
  }
});

element("share").addEventListener("click", async () => {
  const url = new URL(location.href);
  url.hash = encodeURIComponent(JSON.stringify(scenario));
  if (url.href.length > 8000) {
    notify("This event stream is too large for a link. Share the JSON or an exported GIF.");
    return;
  }
  try {
    await navigator.clipboard.writeText(url.href);
    notify("Scenario link copied.");
  } catch {
    window.prompt("Copy this scenario link", url.href);
  }
});

const progress = element("gif-status");
const preview = element("gif-preview");
const download = element<HTMLAnchorElement>("download");
const makeGif = element<HTMLButtonElement>("make-gif");

function gifStatus(text: string) {
  progress.textContent = text;
  progress.hidden = !text;
}

function clearGif() {
  if (gifUrl) URL.revokeObjectURL(gifUrl);
  gifUrl = undefined;
  preview.hidden = true;
  download.hidden = true;
}

element("export").addEventListener("click", () => {
  const end = Math.max(8, Math.ceil(time));
  element<HTMLInputElement>("gif-start").value = String(Math.max(0, end - 12));
  element<HTMLInputElement>("gif-end").value = String(end);
  gifStatus("");
  exportDialog.showModal();
});

element("export-form").addEventListener("submit", event => {
  event.preventDefault();
  const start = element<HTMLInputElement>("gif-start").valueAsNumber;
  const end = element<HTMLInputElement>("gif-end").valueAsNumber;
  const width = Number(element<HTMLSelectElement>("gif-width").value);
  if (end - start < 0.1 || end - start > 30) {
    gifStatus("Choose an interval between 0.1 and 30 seconds.");
    return;
  }
  if (scenario.kind === "simulation" && end > time + 60) {
    gifStatus("Choose an end within 60 seconds of the live simulation time.");
    return;
  }

  clearGif();
  gifStatus("Rendering GIF…");
  makeGif.disabled = true;

  function finish() {
    worker?.terminate();
    worker = undefined;
    makeGif.disabled = false;
  }

  try {
    worker = new Worker(new URL("./export-worker.js", import.meta.url), { type: "module" });
    worker.onmessage = event => {
      const result = event.data;
      if (result.kind === "progress") {
        gifStatus(`Rendering GIF… ${result.percent}%`);
      } else if (result.kind === "error") {
        gifStatus(result.message);
        finish();
      } else {
        const gif = new Blob([result.gif], { type: "image/gif" });
        gifUrl = URL.createObjectURL(gif);
        element<HTMLImageElement>("gif-image").src = gifUrl;
        download.href = gifUrl;
        download.hidden = false;
        preview.hidden = false;
        gifStatus(`${width} px wide · ${number(gif.size / 1024)} KB`);
        finish();
      }
    };
    worker.onerror = () => {
      gifStatus("GIF rendering could not start in this browser.");
      finish();
    };
    worker.postMessage({
      scenario, start, end, width, animated: element<HTMLInputElement>("gif-animate").checked,
      mode: blockView, colorObserved: colorObserved.checked,
      showPresence: showPresence.checked, showAveragePresence: showAveragePresence.checked,
      showSojourn: showSojourn.checked, matchStripScale: matchStripScale.checked,
    });
  } catch {
    gifStatus("GIF rendering could not start in this browser.");
    finish();
  }
});

exportDialog.addEventListener("close", () => {
  worker?.terminate();
  worker = undefined;
  makeGif.disabled = false;
  clearGif();
});

syncControls();
syncScenarioTargets();
syncColorNotes();
syncPresenceControls();
requestAnimationFrame(tick);
