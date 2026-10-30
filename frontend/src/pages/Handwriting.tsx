import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useParams } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';
import { appendCanvasSample, summarizeCanvasTelemetry } from '../features/handwritingTelemetry.js';
import type { CanvasSample, CanvasTelemetrySummary } from '../features/handwritingTelemetry.js';
import { FileDropZone } from '../components/FileDropZone';

// Must match ANALYSIS_VERSION in app_backend/services/spiral_video_analysis.py.
const SPIRAL_ANALYSIS_VERSION = 'ink-hand-v2';
import { VideoObservations } from '../components/VideoObservations';

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
  const [videoAnalysis, setVideoAnalysis] = useState<any>(null);
  const [videoBusy, setVideoBusy] = useState(false);
  const [videoError, setVideoError] = useState('');
  const [tab, setTab] = useState<'draw' | 'video'>('draw');
  const [previewFailed, setPreviewFailed] = useState(false);

  useEffect(() => {
    return () => {
      if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    };
  }, [videoPreviewUrl]);

  const selectVideo = (file: File | null) => {
    setVideoFile(file);
    setPreviewFailed(false);
    setVideoError('');
    setVideoAnalysis(null);
    if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    setVideoPreviewUrl(file ? URL.createObjectURL(file) : null);
  };

  const analyzeVideo = async () => {
    if (!id || !videoFile || videoBusy) return;
    // Clear the previous result first, so a failed request never leaves an older analysis on screen.
    setVideoAnalysis(null);
    setVideoError('');
    setVideoBusy(true);
    try {
      const response = await assessmentsApi.analyzeSpiralVideo(Number(id), videoFile);
      const analysis = response.analysis;
      if (!analysis?.measurements || analysis.analysis_version !== SPIRAL_ANALYSIS_VERSION || analysis.annotated_video_format !== 'webm') {
        throw new Error(
          'The server answered with an analysis from an older version, so the result was not displayed. ' +
          'Stop the backend that is running on this address, start it again with python -m app_backend, and analyse the video again.'
        );
      }
      setVideoAnalysis(analysis);
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
        ← Back to Dashboard
      </button>
      <div className="card">
        <div style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '1rem', marginBottom: '1.5rem' }}>
          <h2 style={{ color: 'var(--sage-dark)', fontSize: '1.5rem', marginBottom: '0.4rem' }}>Handwriting &amp; Spiral Assessment</h2>
          <p style={{ color: 'var(--text-light)', fontSize: '0.9rem', margin: 0 }}>
            Draw a spiral on screen, or upload a video of one. Both produce research observations only.
          </p>
        </div>

        <div role="tablist" aria-label="Spiral input" style={{ display: 'flex', gap: '0.25rem', borderBottom: '1px solid var(--border-subtle)', marginBottom: '1.25rem' }}>
          {(['draw', 'video'] as const).map((key) => (
            <button key={key} type="button" role="tab" id={`tab-${key}`} aria-selected={tab === key} aria-controls={`panel-${key}`}
              onClick={() => setTab(key)}
              style={{ padding: '0.7rem 1.1rem', border: 'none', background: 'none', cursor: 'pointer', fontSize: '0.95rem', fontWeight: 600,
                color: tab === key ? 'var(--sage-dark)' : 'var(--text-light)',
                borderBottom: tab === key ? '3px solid var(--sage-green)' : '3px solid transparent', marginBottom: '-1px' }}>
              {key === 'draw' ? 'Draw Spiral' : 'Upload Video'}
            </button>
          ))}
        </div>

        {tab === 'draw' && (
          <section id="panel-draw" role="tabpanel" aria-labelledby="tab-draw">

          <h3 id="draw-heading" style={{ color: 'var(--sage-dark)', fontSize: '1.1rem', margin: '0 0 0.4rem' }}>1. Draw a spiral</h3>
          <p style={{ color: 'var(--text-light)', fontSize: '0.88rem', margin: '0 0 0.75rem' }}>
            Trace the guide from the centre outward, then save the drawing.
          </p>
          <div className="canvas-container">
            <canvas
              ref={canvasRef}
              width={500}
              height={500}
              style={{ cursor: 'crosshair', display: 'block', touchAction: 'none', maxWidth: '100%', height: 'auto' }}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
            />
          </div>
          <div className="canvas-toolbar">
            <span style={{ fontSize: '0.85rem', color: 'var(--text-light)' }}>
              {pointCount > 0 || isDrawing
                ? <>Points captured: <strong>{pointCount}</strong>{isDrawing ? ' (drawing)' : canvasSaved ? ' (saved)' : ''}</>
                : canvasSaved ? 'A saved drawing is on record' : 'No points captured yet'}
            </span>
            <button type="button" className="btn btn-secondary" onClick={initializeCanvas}>Clear</button>
          </div>
          <button
            type="button"
            className="btn btn-primary"
            style={{ width: '100%', marginTop: '0.75rem', padding: '0.8rem' }}
            onClick={reviewCapture}
            disabled={pointCount < 2 || isDrawing || canvasSaving}
          >
            {canvasSaving ? 'Saving drawing…' : 'Save drawing'}
          </button>
          {canvasError && <p role="alert" className="error" style={{ marginTop: '0.6rem' }}>{canvasError}</p>}
          {summary && (
            <div aria-live="polite" style={{ marginTop: '0.9rem', padding: '0.9rem 1rem', border: '1px solid var(--border-subtle)', borderRadius: 6, background: '#FAFAF7' }}>
              <strong style={{ color: 'var(--text-dark)' }}>Drawing summary {canvasSaved ? '(saved)' : '(not yet saved)'}</strong>
              <p style={{ margin: '0.4rem 0 0', fontSize: '0.9rem' }}>
                {summary.sampleCount} points · {summary.strokeCount} stroke(s) · {(summary.durationMs / 1000).toFixed(1)} s
              </p>
              <p style={{ margin: '0.25rem 0 0', fontSize: '0.9rem', color: 'var(--text-light)' }}>
                Screen estimate only: not calibrated to a tablet and not a clinical measurement.
              </p>
              <details style={{ marginTop: '0.6rem' }}>
                <summary>Measurement details</summary>
                <ul style={{ fontSize: '0.88rem' }}>
                  <li>Path length: {summary.pathLengthPx.toFixed(1)} canvas px</li>
                  <li>Mean speed: {summary.meanSpeedPxPerSecond == null ? 'not enough samples' : `${summary.meanSpeedPxPerSecond.toFixed(1)} px/s (max ${summary.maxSpeedPxPerSecond?.toFixed(1)})`}</li>
                  <li>Pauses of 250 ms or more: {summary.pausesMs.length}</li>
                  <li>Pressure: {summary.pressureSamples ? `${summary.pressureSamples} browser-reported samples (normalized, not tablet-calibrated)` : 'not reported by this device'}</li>
                  <li>Pen tilt: {summary.tiltSamples ? `${summary.tiltSamples} samples` : 'not reported by this device'}</li>
                </ul>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-light)' }}>
                  The handwriting classifier was trained on digitizer data with calibrated pressure and grip angle. Screen drawings do not provide those, so they are not classified.
                </p>
              </details>
            </div>
          )}
          </section>
        )}

        {tab === 'video' && (
          <section id="panel-video" role="tabpanel" aria-labelledby="tab-video">
            <p style={{ color: 'var(--text-light)', fontSize: '0.9rem', margin: '0 0 0.9rem' }}>
              Upload a recording of a spiral. The app detects the ink on the paper, ignores the hand and pen, and produces an annotated copy of the video.
            </p>
            <FileDropZone
              id="spiral-video"
              file={videoFile}
              accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,.mp4,.mov,.avi,.webm"
              acceptLabel="MP4, MOV, AVI or WebM"
              disabled={videoBusy}
              onFile={selectVideo}
            />
            {videoPreviewUrl && (
              <div style={{ marginTop: '0.9rem' }}>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-light)', margin: '0 0 0.3rem' }}>Original upload (unchanged)</p>
                <video controls preload="metadata" src={videoPreviewUrl} onError={() => setPreviewFailed(true)} style={{ display: 'block', width: '100%', maxHeight: 340, background: '#111', borderRadius: 8 }}>
                  Your browser cannot play this video.
                </video>
                {previewFailed && (
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-light)', margin: '0.4rem 0 0' }}>
                    This browser cannot preview the original file. The analysis can still run on it.
                  </p>
                )}
              </div>
            )}
            <button
              type="button"
              className="btn btn-primary"
              style={{ marginTop: '0.9rem' }}
              onClick={analyzeVideo}
              disabled={!videoFile || videoBusy || !id}
            >
              {videoBusy ? 'Analysing video…' : 'Analyse Video'}
            </button>
            {videoBusy && <p role="status" style={{ fontSize: '0.88rem', color: 'var(--text-light)' }}>Uploading and creating the annotated video. This can take a minute.</p>}
            {videoError && <p role="alert" className="error" style={{ marginTop: '0.6rem' }}>{videoError}</p>}
            {videoAnalysis && videoAnalysis.status === 'failed' && (
              <p role="alert" className="error" style={{ marginTop: '0.6rem' }}>
                Video analysis failed: {videoAnalysis.error?.message ?? 'no annotated output was created.'}
              </p>
            )}
            {videoAnalysis && videoAnalysis.status === 'completed' && videoAnalysis.video && (
              <VideoObservations analysis={videoAnalysis} assessmentId={Number(id)} />
            )}
          </section>
        )}

        <details style={{ marginTop: '1.5rem' }}>
          <summary style={{ color: 'var(--text-light)' }}>About these observations</summary>
          <ul style={{ fontSize: '0.88rem', color: 'var(--text-light)' }}>
            <li>Screen drawings record pointer positions, so their speed and timing values are estimates, not clinical measurements.</li>
            <li>Video analysis measures visible pixel changes. It does not measure tremor, pen pressure or grip.</li>
            <li>Neither input is passed to the handwriting classifier, and no disease prediction is produced from them.</li>
          </ul>
        </details>
      </div>
    </div>
  );
}
