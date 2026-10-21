import test from 'node:test';
import assert from 'node:assert/strict';
import { appendCanvasSample, summarizeCanvasTelemetry } from './handwritingTelemetry.js';

const sample = (overrides = {}) => ({
  x: 0,
  y: 0,
  timestampMs: 0,
  phase: 'move',
  strokeId: 0,
  pointerType: 'mouse',
  pressure: null,
  pressureAvailable: false,
  tiltX: null,
  tiltY: null,
  tiltAvailable: false,
  ...overrides,
});

test('captures time-ordered canvas kinematics without crossing strokes', () => {
  const points = [];
  appendCanvasSample(points, sample({ phase: 'down', timestampMs: 0 }));
  appendCanvasSample(points, sample({ x: 10, timestampMs: 100 }));
  appendCanvasSample(points, sample({ x: 30, timestampMs: 200 }));
  appendCanvasSample(points, sample({ x: 60, timestampMs: 300 }));
  appendCanvasSample(points, sample({ x: 100, timestampMs: 600, phase: 'up' }));
  appendCanvasSample(points, sample({ x: 200, timestampMs: 900, phase: 'down', strokeId: 1 }));

  assert.equal(points[3].speedPxPerSecond, 300);
  assert.equal(points[3].accelerationPxPerSecond2, 1000);
  assert.equal(points[3].jerkPxPerSecond3, 0);
  const result = summarizeCanvasTelemetry(points);
  assert.equal(result.strokeCount, 2);
  assert.equal(result.pathLengthPx, 100);
  assert.equal(result.durationMs, 900);
  assert.deepEqual(result.pausesMs, [300, 300]);
});

test('reports sensor availability only when source samples mark it present', () => {
  const points = [];
  appendCanvasSample(points, sample({ phase: 'down', pressure: 0.2, pressureAvailable: true, tiltX: 8, tiltY: 2, tiltAvailable: true }));
  appendCanvasSample(points, sample({ x: 10, timestampMs: 10, pressure: 0.8, pressureAvailable: true, tiltX: 12, tiltY: 3, tiltAvailable: true }));
  const result = summarizeCanvasTelemetry(points);
  assert.equal(result.pressureSamples, 2);
  assert.equal(result.pressureCoverage, 1);
  assert.equal(result.tiltSamples, 2);
});
