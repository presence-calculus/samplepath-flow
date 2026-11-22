export type MarkedPoint = { t: number; duration: number };
export type Arrival = MarkedPoint & { strip: number; end: number; cumulativeSojourn: number };
export type PresenceStep = { t: number; present: number };

export type Simulation = { kind: "simulation"; seed: number; rate: number; mean: number; blockCount?: number };
export type Scenario = Simulation | { kind: "file"; events: MarkedPoint[] };

export const defaultScenario: Simulation = { kind: "simulation", seed: 2718, rate: 1.15, mean: 4.5 };

export function simulation(seed: number, rate: number, mean: number, blockCount?: number): Simulation {
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295 ||
      !Number.isFinite(rate) || rate < 0.05 || rate > 20 ||
      !Number.isFinite(mean) || mean < 0.1 || mean > 300) {
    throw new Error("Use a valid seed, arrival rate, and mean duration.");
  }
  if (blockCount !== undefined && (!Number.isSafeInteger(blockCount) || blockCount < 1)) {
    throw new Error("Use a positive whole number of blocks or leave it blank for unlimited arrivals.");
  }
  return { kind: "simulation", seed, rate, mean, blockCount };
}

export function fromJson(data: unknown): Scenario {
  if (!Array.isArray(data)) throw new Error("The input must be an array of marked points.");
  let previousTime = 0;
  const events = data.map((point: unknown, index): MarkedPoint => {
    const value = point as Partial<MarkedPoint> | null;
    if (!value || typeof value.t !== "number" || typeof value.duration !== "number" ||
        !Number.isFinite(value.t) || !Number.isFinite(value.duration) || value.t < previousTime ||
        value.duration <= 0 || !Number.isFinite(value.t + value.duration) ||
        value.t + value.duration <= value.t) {
      throw new Error(`Point ${index + 1} needs a nonnegative, ordered arrival time and a positive duration.`);
    }
    previousTime = value.t;
    return { t: value.t, duration: value.duration };
  });
  return { kind: "file", events };
}

function randomGenerator(seed: number) {
  // Mulberry32 keeps the demonstration identical on reset and in shared links.
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let mixed = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed);
    return (((mixed ^ (mixed >>> 14)) >>> 0) + 0.5) / 4294967296;
  };
}

function observedDuration(point: MarkedPoint, t: number): number {
  return Math.min(t - point.t, point.duration);
}

export class PresenceProcess {
  readonly arrivals: Arrival[] = [];
  readonly scenario: Scenario;
  private readonly random: () => number;
  private lastDeparture = 0;
  private presenceHistory?: { arrivalCount: number; steps: PresenceStep[] };

  constructor(scenario: Scenario) {
    this.scenario = scenario;
    this.random = randomGenerator(scenario.kind === "simulation" ? scenario.seed : 0);
    if (scenario.kind === "file") {
      for (const event of scenario.events) this.append(event);
    } else {
      this.generateNext();
    }
  }

  private append(point: MarkedPoint) {
    const end = point.t + point.duration;
    const cumulativeSojourn = (this.arrivals.at(-1)?.cumulativeSojourn ?? 0) + point.duration;
    this.arrivals.push({ ...point, end, strip: this.arrivals.length + 1, cumulativeSojourn });
    this.lastDeparture = Math.max(this.lastDeparture, end);
  }

  private generateNext() {
    if (this.scenario.kind !== "simulation") return;
    const previous = this.arrivals.at(-1)?.t ?? 0;
    const t = previous - Math.log(this.random()) / this.scenario.rate;
    const duration = -Math.log(this.random()) * this.scenario.mean;
    this.append({ t, duration });
  }

  ensureAhead(t: number) {
    if (this.scenario.kind !== "simulation") return;
    const limit = this.scenario.blockCount ?? Infinity;
    while (this.arrivals.length < limit && this.arrivals.at(-1)!.t <= t) this.generateNext();
  }

  get completionTime(): number | undefined {
    if (this.scenario.kind === "simulation" && this.arrivals.length === this.scenario.blockCount) {
      return this.lastDeparture;
    }
    return undefined;
  }

  countAt(t: number): number {
    let low = 0;
    let high = this.arrivals.length;
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if (this.arrivals[mid].t <= t) low = mid + 1;
      else high = mid;
    }
    return low;
  }

  nextAfter(t: number): Arrival | undefined {
    this.ensureAhead(t);
    return this.arrivals[this.countAt(t)];
  }

  presenceSteps(arrivalCount: number): readonly PresenceStep[] {
    if (this.presenceHistory?.arrivalCount === arrivalCount) return this.presenceHistory.steps;

    const changes = new Map<number, number>();
    for (let i = 0; i < arrivalCount; i++) {
      const { t, end } = this.arrivals[i];
      changes.set(t, (changes.get(t) ?? 0) + 1);
      changes.set(end, (changes.get(end) ?? 0) - 1);
    }

    const steps: PresenceStep[] = [];
    let present = 0;
    for (const [t, change] of [...changes].sort(([a], [b]) => a - b)) {
      if (change === 0) continue;
      present += change;
      steps.push({ t, present });
    }
    this.presenceHistory = { arrivalCount, steps };
    return steps;
  }

  cumulativeObservedSojourns(t: number, count = this.countAt(t)): number[] {
    const totals: number[] = [];
    let observed = 0;
    for (let i = 0; i < count; i++) {
      observed += observedDuration(this.arrivals[i], t);
      totals.push(observed);
    }
    return totals;
  }

  measure(t: number, count = this.countAt(t)) {
    let present = 0;
    let mass = 0;
    for (let index = 0; index < count; index++) {
      const point = this.arrivals[index];
      if (t < point.end) present++;
      mass += observedDuration(point, t);
    }
    return { present, mass };
  }
}
