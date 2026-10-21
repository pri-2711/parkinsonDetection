export type CanvasPhase = 'down' | 'move' | 'up';

export interface CanvasSampleInput {
  x: number;
  y: number;
  timestampMs: number;
  phase: CanvasPhase;
  strokeId: number;
  pointerType: string;
  pressure: number | null;
  pressureAvailable: boolean;
  tiltX: number | null;
  tiltY: number | null;
  tiltAvailable: boolean;
}

export interface CanvasSample extends CanvasSampleInput {
  pointIndex: number;
  deltaX: number;
  deltaY: number;
  speedPxPerSecond: number | null;
  accelerationPxPerSecond2: number | null;
  jerkPxPerSecond3: number | null;
}

export interface CanvasTelemetrySummary {
  sampleCount: number;
  strokeCount: number;
  pathLengthPx: number;
  meanSpeedPxPerSecond: number | null;
  maxSpeedPxPerSecond: number | null;
  meanAccelerationPxPerSecond2: number | null;
  meanJerkPxPerSecond3: number | null;
  pausesMs: number[];
  pressureSamples: number;
  pressureCoverage: number;
  tiltSamples: number;
  durationMs: number;
}

export function appendCanvasSample(samples: CanvasSample[], input: CanvasSampleInput): CanvasSample;
export function summarizeCanvasTelemetry(samples: CanvasSample[], pauseThresholdMs?: number): CanvasTelemetrySummary;
