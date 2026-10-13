import { useState, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';

interface TrajectoryPoint {
  point_index: number;
  x: number;
  y: number;
  z: number;
  pressure: number;
  grip_angle: number;
  timestamp: number;
  test_id: number;
}

export function Handwriting() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [activeMode, setActiveMode] = useState<'canvas' | 'csv' | 'video'>('canvas');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [pointCount, setPointCount] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pointsRef = useRef<TrajectoryPoint[]>([]);
  const startTimeRef = useRef<number | null>(null);

  // File upload state for CSV or video
  const [file, setFile] = useState<File | null>(null);

  useEffect(() => {
    if (activeMode === 'canvas') {
      initCanvas();
    }
  }, [activeMode]);

  const initCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Reset points
    pointsRef.current = [];
    startTimeRef.current = null;
    setPointCount(0);

    // Draw background & reference Archimedean spiral guide
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
      const r = 8 * theta;
      const x = centerX + r * Math.cos(theta);
      const y = centerY + r * Math.sin(theta);
      if (theta === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // Center start dot
    ctx.fillStyle = '#788B6A';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 4, 0, 2 * Math.PI);
    ctx.fill();
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.setPointerCapture(e.pointerId);
    setIsDrawing(true);

    if (!startTimeRef.current) {
      startTimeRef.current = Date.now();
    }

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const pressure = e.pressure && e.pressure > 0 ? e.pressure : 0.5;
    const timestamp = Date.now() - startTimeRef.current;

    const newPt: TrajectoryPoint = {
      point_index: pointsRef.current.length,
      x: Math.round(x * 10) / 10,
      y: Math.round(y * 10) / 10,
      z: 0.0, // pen down
      pressure: Math.round(pressure * 1000) / 1000,
      grip_angle: 45.0,
      timestamp,
      test_id: 1,
    };
    pointsRef.current.push(newPt);
    setPointCount(pointsRef.current.length);

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.strokeStyle = '#3E5438';
      ctx.lineWidth = Math.max(2, pressure * 4);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x, y);
    }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas || !startTimeRef.current) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const pressure = e.pressure && e.pressure > 0 ? e.pressure : 0.5;
    const timestamp = Date.now() - startTimeRef.current;

    const newPt: TrajectoryPoint = {
      point_index: pointsRef.current.length,
      x: Math.round(x * 10) / 10,
      y: Math.round(y * 10) / 10,
      z: 0.0,
      pressure: Math.round(pressure * 1000) / 1000,
      grip_angle: 45.0,
      timestamp,
      test_id: 1,
    };
    pointsRef.current.push(newPt);
    setPointCount(pointsRef.current.length);

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.lineTo(x, y);
      ctx.stroke();
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch (_) {}
  };

  const handleSubmitCanvas = async () => {
    if (!id) return;
    if (pointsRef.current.length < 15) {
      setError('Please draw a complete spiral trace (at least 15 trajectory points required).');
      return;
    }

    setSubmitting(true);
    setError('');

    try {
      await assessmentsApi.predictSpiral(parseInt(id, 10), pointsRef.current, 'canvas_drawing');
      navigate('/dashboard');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Handwriting classification failed');
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmitFile = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!id || !file) return;

    setSubmitting(true);
    setError('');

    try {
      if (activeMode === 'csv') {
        // Parse CSV file in browser and submit trajectory
        const text = await file.text();
        const lines = text.trim().split('\n');
        if (lines.length < 3) throw new Error('CSV file is empty or too short.');

        const headers = lines[0].split(',').map((h) => h.trim().toLowerCase());
        const xIdx = headers.indexOf('x');
        const yIdx = headers.indexOf('y');
        if (xIdx === -1 || yIdx === -1) throw new Error('CSV must contain x and y coordinate columns.');

        const pts: TrajectoryPoint[] = [];
        for (let i = 1; i < lines.length; i++) {
          const cols = lines[i].split(',').map((c) => parseFloat(c.trim()));
          if (cols.length > Math.max(xIdx, yIdx) && !isNaN(cols[xIdx]) && !isNaN(cols[yIdx])) {
            pts.push({
              point_index: pts.length,
              x: cols[xIdx],
              y: cols[yIdx],
              z: headers.indexOf('z') !== -1 ? cols[headers.indexOf('z')] : 0.0,
              pressure: headers.indexOf('pressure') !== -1 ? cols[headers.indexOf('pressure')] : 0.5,
              grip_angle: headers.indexOf('grip_angle') !== -1 ? cols[headers.indexOf('grip_angle')] : 45.0,
              timestamp: headers.indexOf('timestamp') !== -1 ? cols[headers.indexOf('timestamp')] : (i - 1) * 20,
              test_id: 1,
            });
          }
        }

        await assessmentsApi.predictSpiral(parseInt(id, 10), pts, 'tablet_csv_upload');
        navigate('/dashboard');
      } else {
        // Video upload: validates upload and explains mapping
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/spiral/predict', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.status === 501) {
          await assessmentsApi.updateSpiral(parseInt(id, 10), null, data.error?.message || 'Camera-to-tablet mapping unvalidated');
          navigate('/dashboard');
        } else {
          throw new Error(data.error?.message || 'Upload failed');
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Processing failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: '820px' }}>
      <button onClick={() => navigate('/dashboard')} className="btn btn-secondary" style={{ marginBottom: '1.25rem' }}>
        ← Back to Dashboard
      </button>

      <div className="card">
        <div style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '1rem', marginBottom: '1.5rem' }}>
          <h2 style={{ color: 'var(--sage-dark)', fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            Handwriting & Spiral Test
          </h2>
          <p style={{ color: 'var(--text-light)', fontSize: '0.9rem', lineHeight: '1.5' }}>
            The trained handwriting model (<code>handwriting_svm_tuned.joblib</code>) evaluates 16 fine-motor kinematic features
            (velocity, acceleration, jerk, path deviation, pen-up intervals) derived from digitizer trajectory telemetry.
          </p>
        </div>

        {/* Input Mode Tabs */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-subtle)' }}>
          <button
            type="button"
            className={`btn ${activeMode === 'canvas' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ borderRadius: '6px 6px 0 0', borderBottom: 'none' }}
            onClick={() => { setActiveMode('canvas'); setError(''); }}
          >
            ✏️ Interactive Spiral Canvas
          </button>
          <button
            type="button"
            className={`btn ${activeMode === 'csv' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ borderRadius: '6px 6px 0 0', borderBottom: 'none' }}
            onClick={() => { setActiveMode('csv'); setError(''); }}
          >
            📊 Upload Tablet CSV Telemetry
          </button>
          <button
            type="button"
            className={`btn ${activeMode === 'video' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ borderRadius: '6px 6px 0 0', borderBottom: 'none' }}
            onClick={() => { setActiveMode('video'); setError(''); }}
          >
            🎥 Video Visual Inspection
          </button>
        </div>

        {activeMode === 'canvas' && (
          <div>
            <p style={{ color: 'var(--text-dark)', fontSize: '0.9rem', marginBottom: '0.75rem' }}>
              <strong>Instructions:</strong> Use your mouse, stylus, or touchscreen to trace the Archimedean spiral starting from the center dot outward in one continuous motion. Telemetry coordinates, timing, and pressure are captured in real-time.
            </p>

            <div className="canvas-container">
              <canvas
                ref={canvasRef}
                width={500}
                height={500}
                style={{ cursor: 'crosshair', display: 'block' }}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
              />
            </div>

            <div className="canvas-toolbar">
              <span style={{ fontSize: '0.85rem', color: 'var(--text-light)', fontWeight: '500' }}>
                Captured Points: <strong>{pointCount}</strong> {pointCount >= 15 ? '✓ Ready' : '(draw spiral to continue)'}
              </span>
              <button type="button" className="btn btn-secondary" onClick={initCanvas}>
                Clear Canvas
              </button>
            </div>

            {error && (
              <div style={{ backgroundColor: 'var(--error-bg)', border: '1px solid var(--error)', padding: '0.75rem 1rem', borderRadius: '6px', margin: '1rem 0' }}>
                <p className="error" style={{ margin: 0 }}>{error}</p>
              </div>
            )}

            <button
              type="button"
              className="btn btn-primary"
              style={{ width: '100%', marginTop: '1.25rem', padding: '0.85rem' }}
              onClick={handleSubmitCanvas}
              disabled={submitting || pointCount < 15}
            >
              {submitting ? 'Extracting Kinematic Features & Classifying…' : 'Submit Drawn Spiral Telemetry'}
            </button>
          </div>
        )}

        {activeMode === 'csv' && (
          <form onSubmit={handleSubmitFile}>
            <p style={{ color: 'var(--text-dark)', fontSize: '0.9rem', marginBottom: '1rem' }}>
              Upload raw tablet telemetry collected from a digitizing tablet (PaHaW schema). The file should contain coordinate columns (<code>x</code>, <code>y</code>, <code>timestamp</code>, and optionally <code>pressure</code>, <code>grip_angle</code>).
            </p>

            <div style={{ padding: '1.5rem', backgroundColor: '#FAFAF7', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: '1.25rem' }}>
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setFile(e.target.files[0]);
                    setError('');
                  }
                }}
                style={{ width: '100%', padding: '0.75rem', border: '1px solid var(--border-subtle)', borderRadius: '6px', backgroundColor: '#FFFFFF' }}
              />
            </div>

            {error && (
              <div style={{ backgroundColor: 'var(--error-bg)', border: '1px solid var(--error)', padding: '0.75rem 1rem', borderRadius: '6px', marginBottom: '1rem' }}>
                <p className="error" style={{ margin: 0 }}>{error}</p>
              </div>
            )}

            <button
              type="submit"
              className="btn btn-primary"
              style={{ width: '100%', padding: '0.85rem' }}
              disabled={submitting || !file}
            >
              {submitting ? 'Processing Telemetry…' : 'Submit Tablet CSV Telemetry'}
            </button>
          </form>
        )}

        {activeMode === 'video' && (
          <form onSubmit={handleSubmitFile}>
            <div style={{ backgroundColor: 'var(--cream-dark)', padding: '1rem', borderRadius: '6px', marginBottom: '1.25rem' }}>
              <p style={{ color: 'var(--text-dark)', fontSize: '0.85rem', margin: 0 }}>
                <strong>Camera Video Notice:</strong> Standard video cannot directly measure tablet stylus pressure, grip angle, or sub-millimeter pen-up states. Video submissions are validated through an OpenCV computer vision pipeline for optical trace irregularity and visual oscillation analysis.
              </p>
            </div>

            <div style={{ padding: '1.5rem', backgroundColor: '#FAFAF7', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: '1.25rem' }}>
              <label htmlFor="spiral-video-file" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '600', color: 'var(--text-dark)' }}>
                Select Spiral Drawing Video (MP4, MOV, WebM, up to 30s)
              </label>
              <input
                id="spiral-video-file"
                type="file"
                accept="video/mp4,video/quicktime,video/webm,video/x-msvideo"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setFile(e.target.files[0]);
                    setError('');
                  }
                }}
                style={{ width: '100%', padding: '0.75rem', border: '1px solid var(--border-subtle)', borderRadius: '6px', backgroundColor: '#FFFFFF' }}
              />
            </div>

            {error && (
              <div style={{ backgroundColor: 'var(--error-bg)', border: '1px solid var(--error)', padding: '0.75rem 1rem', borderRadius: '6px', marginBottom: '1rem' }}>
                <p className="error" style={{ margin: 0 }}>{error}</p>
              </div>
            )}

            <button
              type="submit"
              className="btn btn-primary"
              style={{ width: '100%', padding: '0.85rem' }}
              disabled={submitting || !file}
            >
              {submitting ? 'Validating Video…' : 'Submit Video for Visual Analysis'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
