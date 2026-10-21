import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useParams } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';
import { appendCanvasSample, summarizeCanvasTelemetry } from '../features/handwritingTelemetry.js';
import type { CanvasSample, CanvasTelemetrySummary } from '../features/handwritingTelemetry.js';

export function Handwriting() {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const [pointCount, setPointCount] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [summary, setSummary] = useState<CanvasTelemetrySummary | null>(null);
  const [canvasSaving, setCanvasSaving] = useState(false);
  const [canvasSaved, setCanvasSaved] = useState(false);
  const [canvasError, setCanvasError] = useState('');
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const samplesRef = useRef<CanvasSample[]>([]);
  const startTimeRef = useRef<number | null>(null);
  const activeStrokeIdRef = useRef(-1);
  const isDrawingRef = useRef(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [annotatedUrl, setAnnotatedUrl] = useState<string | null>(null);
  const [videoAnalysis, setVideoAnalysis] = useState<any>(null);
  const [videoBusy, setVideoBusy] = useState(false);
  const [videoError, setVideoError] = useState('');

  useEffect(() => {
    return () => {
      if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
      if (annotatedUrl) URL.revokeObjectURL(annotatedUrl);
    };
  }, [videoPreviewUrl, annotatedUrl]);

  const selectVideo = (file: File | null) => {
    setVideoFile(file);
    setVideoError('');
    setVideoAnalysis(null);
    if (annotatedUrl) URL.revokeObjectURL(annotatedUrl);
    setAnnotatedUrl(null);
    if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    setVideoPreviewUrl(file ? URL.createObjectURL(file) : null);
  };

  const analyzeVideo = async () => {
    if (!id || !videoFile) return;
    setVideoBusy(true);
    setVideoError('');
    try {
      const response = await assessmentsApi.analyzeSpiralVideo(Number(id), videoFile);
      const blob = await assessmentsApi.getSpiralVideoBlob(Number(id), 'annotated');
      setAnnotatedUrl(URL.createObjectURL(blob));
      setVideoAnalysis(response.analysis);
    } catch (error) {
      setVideoError(error instanceof Error ? error.message : 'Video analysis failed. Please check the file and try again.');
    } finally {
      setVideoBusy(false);
    }
  };

  useEffect(() => {
    initializeCanvas();
  }, []);

  useEffect(() => {
    if (!id) return;
    let active = true;
    assessmentsApi.get(Number(id)).then(async (assessment) => {
      const outcome = assessment.steps.spiral.outcome;
      const savedCanvas = outcome?.canvas_telemetry;
      if (savedCanvas && active) {
        setSummary({
          sampleCount: savedCanvas.sample_count, strokeCount: savedCanvas.stroke_count,
          pathLengthPx: savedCanvas.path_length_px, meanSpeedPxPerSecond: savedCanvas.mean_speed_px_per_second,
          maxSpeedPxPerSecond: savedCanvas.max_speed_px_per_second,
          meanAccelerationPxPerSecond2: savedCanvas.mean_acceleration_px_per_second2,
          meanJerkPxPerSecond3: savedCanvas.mean_jerk_px_per_second3,
          pausesMs: savedCanvas.pause_durations_ms, pressureSamples: savedCanvas.pressure_samples,
          pressureCoverage: savedCanvas.pressure_coverage, tiltSamples: savedCanvas.tilt_samples,
          durationMs: savedCanvas.duration_ms,
        });
        setCanvasSaved(true);
      }
      const savedVideo = outcome?.video_analysis;
      if (savedVideo && active) setVideoAnalysis(savedVideo);
      if (savedVideo?.status === 'completed' && active) {
        try {
          const blob = await assessmentsApi.getSpiralVideoBlob(Number(id), 'annotated');
          if (active) setAnnotatedUrl(URL.createObjectURL(blob));
        } catch {
          if (active) setVideoError('The saved analysis is present, but its private annotated video could not be retrieved.');
        }
      }
    }).catch(() => undefined);
    return () => { active = false; };
  }, [id]);

  const initializeCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    samplesRef.current = [];
    startTimeRef.current = null;
    activeStrokeIdRef.current = -1;
    isDrawingRef.current = false;
    setIsDrawing(false);
    setPointCount(0);
    setSummary(null);
    setCanvasSaved(false);
    setCanvasError('');
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.strokeStyle = '#E0E5DC';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    ctx.beginPath();
    for (let theta = 0; theta < 6 * Math.PI; theta += 0.05) {
      const radius = 8 * theta;
      const x = centerX + radius * Math.cos(theta);
      const y = centerY + radius * Math.sin(theta);
      if (theta === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = '#788B6A';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 4, 0, 2 * Math.PI);
    ctx.fill();
  };

  const recordSample = (
    event: React.PointerEvent<HTMLCanvasElement>,
    phase: 'down' | 'move' | 'up',
    native = event.nativeEvent,
  ) => {
    const canvas = canvasRef.current;
    if (!canvas || startTimeRef.current === null) return null;
    const rect = canvas.getBoundingClientRect();
    const pressureAvailable = native.pointerType !== 'mouse' && Number.isFinite(native.pressure);
    const tiltAvailable = native.pointerType === 'pen'
      && Number.isFinite(native.tiltX)
      && Number.isFinite(native.tiltY)
      && (native.tiltX !== 0 || native.tiltY !== 0);
    const sample = appendCanvasSample(samplesRef.current, {
      x: Math.round((native.clientX - rect.left) * canvas.width / rect.width * 10) / 10,
      y: Math.round((native.clientY - rect.top) * canvas.height / rect.height * 10) / 10,
      timestampMs: performance.now() - startTimeRef.current,
      phase,
      strokeId: activeStrokeIdRef.current,
      pointerType: native.pointerType || 'unknown',
      pressure: pressureAvailable ? native.pressure : null,
      pressureAvailable,
      tiltX: tiltAvailable ? native.tiltX : null,
      tiltY: tiltAvailable ? native.tiltY : null,
      tiltAvailable,
    });
    setPointCount(samplesRef.current.length);
    setSummary(null);
    return sample;
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.setPointerCapture(event.pointerId);
    setCanvasSaved(false);
    if (startTimeRef.current === null) startTimeRef.current = performance.now();
    activeStrokeIdRef.current += 1;
    isDrawingRef.current = true;
    setIsDrawing(true);
    const point = recordSample(event, 'down');
    const ctx = canvas.getContext('2d');
    if (point && ctx) {
      ctx.strokeStyle = '#3E5438';
      ctx.lineWidth = point.pressureAvailable && point.pressure !== null ? 1.5 + point.pressure * 5 : 3;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(point.x, point.y);
    }
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) return;
    const ctx = canvasRef.current?.getContext('2d');
    const coalesced = event.nativeEvent.getCoalescedEvents?.() ?? [];
    const movementEvents = coalesced.length ? coalesced : [event.nativeEvent];
    for (const movementEvent of movementEvents) {
      const point = recordSample(event, 'move', movementEvent);
      if (point && ctx) {
        ctx.lineWidth = point.pressureAvailable && point.pressure !== null ? 1.5 + point.pressure * 5 : 3;
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
      }
    }
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) return;
    recordSample(event, 'up');
    isDrawingRef.current = false;
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (canvas?.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  };

  const reviewCapture = async () => {
    const canvas = canvasRef.current;
    if (!canvas || !id) return;
    setCanvasSaving(true);
    setCanvasError('');
    try {
      const response = await assessmentsApi.saveCanvasObservation(Number(id), samplesRef.current, canvas.width, canvas.height);
      const observed = response.observation;
      setSummary({
        sampleCount: observed.sample_count,
        strokeCount: observed.stroke_count,
        pathLengthPx: observed.path_length_px,
        meanSpeedPxPerSecond: observed.mean_speed_px_per_second,
        maxSpeedPxPerSecond: observed.max_speed_px_per_second,
        meanAccelerationPxPerSecond2: observed.mean_acceleration_px_per_second2,
        meanJerkPxPerSecond3: observed.mean_jerk_px_per_second3,
        pausesMs: observed.pause_durations_ms,
        pressureSamples: observed.pressure_samples,
        pressureCoverage: observed.pressure_coverage,
        tiltSamples: observed.tilt_samples,
        durationMs: observed.duration_ms,
      });
      setCanvasSaved(true);
    } catch (error) {
      setSummary(summarizeCanvasTelemetry(samplesRef.current));
      setCanvasError(error instanceof Error ? error.message : 'Could not save canvas observations to this assessment.');
    } finally {
      setCanvasSaving(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: '820px' }}>
      <button onClick={() => navigate('/dashboard')} className="btn btn-secondary" style={{ marginBottom: '1.25rem' }}>
        Back to Dashboard
      </button>
      <div className="card">
        <div style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '1rem', marginBottom: '1.5rem' }}>
          <h2 style={{ color: 'var(--sage-dark)', fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            Handwriting &amp; Spiral Test
          </h2>
          <p style={{ color: 'var(--text-light)', fontSize: '0.9rem', lineHeight: '1.5' }}>
            Draw a spiral to capture browser pointer telemetry. This proof of concept reports canvas kinematics and sensor availability;
            the measured canvas summary is saved to this assessment, but it is not sent to the digitizer-trained classifier and produces no prediction.
          </p>
        </div>

        <p style={{ color: 'var(--text-dark)', fontSize: '0.9rem', marginBottom: '0.75rem' }}>
          <strong>Instructions:</strong> Trace the guide from the center outward. Coordinates and elapsed time start at the first
          pointer-down. Use a pressure-capable pen to test optional browser pressure and tilt reporting.
        </p>
        <div className="canvas-container">
          <canvas
            ref={canvasRef}
            width={500}
            height={500}
            style={{ cursor: 'crosshair', display: 'block', touchAction: 'none' }}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
          />
        </div>
        <div className="canvas-toolbar">
          <span style={{ fontSize: '0.85rem', color: 'var(--text-light)', fontWeight: '500' }}>
            Captured pointer samples: <strong>{pointCount}</strong> {isDrawing ? '(drawing)' : canvasSaved ? '(summary saved)' : '(local capture)'}
          </span>
          <button type="button" className="btn btn-secondary" onClick={initializeCanvas}>
            Clear Canvas
          </button>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          style={{ width: '100%', marginTop: '1.25rem', padding: '0.85rem' }}
          onClick={reviewCapture}
          disabled={pointCount < 2 || isDrawing || canvasSaving}
        >
          {canvasSaving ? 'Saving Canvas Observations…' : 'Save & Review Canvas Telemetry'}
        </button>
        {canvasError && <p role="alert" style={{ color: '#9b2c2c' }}>{canvasError}</p>}
        {summary && (
          <section aria-live="polite" style={{ marginTop: '1rem', padding: '1rem', border: '1px solid var(--border-subtle)', borderRadius: '6px' }}>
            <h3 style={{ color: 'var(--sage-dark)', marginTop: 0 }}>Canvas telemetry observations {canvasSaved ? '(saved)' : '(local preview)'}</h3>
            <p>{summary.sampleCount} samples · {summary.strokeCount} stroke(s) · {summary.pathLengthPx.toFixed(1)} canvas px path · {summary.durationMs.toFixed(0)} ms</p>
            <p>
              Speed: {summary.meanSpeedPxPerSecond == null ? 'not enough samples' : `${summary.meanSpeedPxPerSecond.toFixed(1)} px/s mean, ${summary.maxSpeedPxPerSecond?.toFixed(1)} px/s max`} ·
              Acceleration samples: {samplesRef.current.filter((sample) => sample.accelerationPxPerSecond2 != null).length} ·
              Jerk samples: {samplesRef.current.filter((sample) => sample.jerkPxPerSecond3 != null).length}
            </p>
            <p>
              Pressure: {summary.pressureSamples ? `${summary.pressureSamples} browser-reported samples (${(summary.pressureCoverage * 100).toFixed(0)}% coverage; normalized, not tablet-calibrated)` : 'unavailable'} ·
              Tilt: {summary.tiltSamples ? `${summary.tiltSamples} nonzero pen-tilt samples` : 'unavailable'} ·
              Pauses ≥250 ms: {summary.pausesMs.length}
            </p>
            <details>
              <summary>Coverage against the 16-feature model schema</summary>
              <ul>
                <li><strong>Direct:</strong> n_points (browser event count only).</li>
                <li><strong>Approximate, canvas units:</strong> path_length, velocity, acceleration, jerk, movement_duration, spatial_std, bounding_box_diagonal.</li>
                <li><strong>Hardware-dependent and not schema-compatible:</strong> pressure statistics (normalized browser pressure, without tablet calibration).</li>
                <li><strong>Unavailable:</strong> grip_angle_std. Pen tilt is orientation and is not grip angle.</li>
                <li><strong>Not equivalent:</strong> pen-up fraction. Stroke boundaries are recorded, but are not the training digitizer's z measurement.</li>
              </ul>
              <p>Canvas-derived observations are stored separately. The existing classifier is not called and no disease prediction is generated.</p>
            </details>
          </section>
        )}

        <section style={{ marginTop: '2rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border-subtle)' }}>
          <h3 style={{ color: 'var(--sage-dark)' }}>Spiral test video analysis</h3>
          <p style={{ color: 'var(--text-light)', fontSize: '0.9rem', lineHeight: 1.5 }}>
            Upload a separate camera recording for an experimental OpenCV visual analysis. This does not call the handwriting classifier.
          </p>
          <label htmlFor="spiral-video">Upload Video (MP4, MOV, AVI, or WebM)</label>
          <input
            id="spiral-video"
            type="file"
            accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,.mp4,.mov,.avi,.webm"
            onChange={(event) => selectVideo(event.target.files?.[0] ?? null)}
            disabled={videoBusy}
            style={{ display: 'block', margin: '0.6rem 0' }}
          />
          {videoFile && <p aria-live="polite">Selected: <strong>{videoFile.name}</strong> ({(videoFile.size / 1024 / 1024).toFixed(1)} MB)</p>}
          {videoPreviewUrl && <video controls preload="metadata" src={videoPreviewUrl} style={{ display: 'block', width: '100%', maxHeight: 360, margin: '0.75rem 0', background: '#111' }}>Video preview is unavailable in this browser.</video>}
          <button type="button" className="btn btn-primary" onClick={analyzeVideo} disabled={!videoFile || videoBusy || !id}>
            {videoBusy ? 'Uploading and analysing…' : 'Analyse Video'}
          </button>
          {videoBusy && <p role="status">Uploading video and generating annotated output. This can take a little time.</p>}
          {videoError && <p role="alert" style={{ color: '#9b2c2c' }}>{videoError}</p>}
          {videoAnalysis && (
            <div aria-live="polite" style={{ marginTop: '1rem', padding: '1rem', border: '1px solid var(--border-subtle)', borderRadius: 6 }}>
              <h4 style={{ marginTop: 0 }}>Video analysis observations</h4>
              <p>Duration: {videoAnalysis.video.duration_seconds}s · {videoAnalysis.video.fps} fps · {videoAnalysis.video.width} × {videoAnalysis.video.height}</p>
              <p>Tracked candidate points: {videoAnalysis.trajectory.tracked_points}</p>
              {videoAnalysis.trajectory.path_length_px != null && <p>Visible-change path: {videoAnalysis.trajectory.path_length_px} px · mean speed: {videoAnalysis.trajectory.mean_speed_px_per_second} px/s · speed variation (CV): {videoAnalysis.trajectory.speed_coefficient_of_variation} · mean direction change: {videoAnalysis.trajectory.mean_direction_change_degrees}°</p>}
              <p>Red candidate movement annotations: {videoAnalysis.candidate_movement_events.length} · blue visible appearance observations: {videoAnalysis.visible_appearance_events.length}</p>
              <ul>{videoAnalysis.limitations.map((limitation: string) => <li key={limitation}>{limitation}</li>)}</ul>
              {annotatedUrl && <video controls preload="metadata" src={annotatedUrl} style={{ display: 'block', width: '100%', maxHeight: 420, background: '#111' }}>Annotated video playback is unavailable in this browser.</video>}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
