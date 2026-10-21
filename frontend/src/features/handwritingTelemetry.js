/**
 * Pure canvas telemetry helpers. These calculations describe browser canvas
 * coordinates; they are not the tablet sensor measurements used by the model.
 */

export function appendCanvasSample(samples, input) {
  let previous = null;
  for (let index = samples.length - 1; index >= 0; index -= 1) {
    if (samples[index].strokeId === input.strokeId) {
      previous = samples[index];
      break;
    }
  }
  const dtSeconds = previous ? (input.timestampMs - previous.timestampMs) / 1000 : 0;
  const deltaX = previous ? input.x - previous.x : 0;
  const deltaY = previous ? input.y - previous.y : 0;
  const speedPxPerSecond = previous && dtSeconds > 0
    ? Math.hypot(deltaX, deltaY) / dtSeconds
    : null;
  const previousMovement = previous?.speedPxPerSecond == null ? null : previous;
  const accelerationPxPerSecond2 = speedPxPerSecond != null && previousMovement && dtSeconds > 0
    ? (speedPxPerSecond - previousMovement.speedPxPerSecond) / dtSeconds
    : null;
  const jerkPxPerSecond3 = accelerationPxPerSecond2 != null && previousMovement?.accelerationPxPerSecond2 != null
    ? (accelerationPxPerSecond2 - previousMovement.accelerationPxPerSecond2) / dtSeconds
    : null;

  const sample = {
    pointIndex: samples.length,
    x: input.x,
    y: input.y,
    timestampMs: input.timestampMs,
    phase: input.phase,
    strokeId: input.strokeId,
    pointerType: input.pointerType,
    pressure: input.pressure,
    pressureAvailable: input.pressureAvailable,
    tiltX: input.tiltX,
    tiltY: input.tiltY,
    tiltAvailable: input.tiltAvailable,
    deltaX,
    deltaY,
    speedPxPerSecond,
    accelerationPxPerSecond2,
    jerkPxPerSecond3,
  };
  samples.push(sample);
  return sample;
}

export function summarizeCanvasTelemetry(samples, pauseThresholdMs = 250) {
  const strokes = new Set(samples.filter((sample) => sample.phase === 'down').map((sample) => sample.strokeId));
  const speeds = samples.map((sample) => sample.speedPxPerSecond).filter(Number.isFinite);
  const accelerations = samples.map((sample) => sample.accelerationPxPerSecond2).filter(Number.isFinite);
  const jerks = samples.map((sample) => sample.jerkPxPerSecond3).filter(Number.isFinite);
  const pauses = [];
  let pathLengthPx = 0;

  for (let index = 1; index < samples.length; index += 1) {
    const previous = samples[index - 1];
    const current = samples[index];
    const elapsed = current.timestampMs - previous.timestampMs;
    if (elapsed >= pauseThresholdMs) pauses.push(elapsed);
    if (current.strokeId === previous.strokeId) {
      pathLengthPx += Math.hypot(current.x - previous.x, current.y - previous.y);
    }
  }

  const measuredPressure = samples.filter((sample) => sample.pressureAvailable).map((sample) => sample.pressure);
  return {
    sampleCount: samples.length,
    strokeCount: strokes.size,
    pathLengthPx,
    meanSpeedPxPerSecond: mean(speeds),
    maxSpeedPxPerSecond: speeds.length ? speeds.reduce((maximum, value) => Math.max(maximum, value), -Infinity) : null,
    meanAccelerationPxPerSecond2: mean(accelerations),
    meanJerkPxPerSecond3: mean(jerks),
    pausesMs: pauses,
    pressureSamples: measuredPressure.length,
    pressureCoverage: samples.length ? measuredPressure.length / samples.length : 0,
    tiltSamples: samples.filter((sample) => sample.tiltAvailable).length,
    durationMs: samples.length ? samples[samples.length - 1].timestampMs - samples[0].timestampMs : 0,
  };
}

function mean(values) {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : null;
}
