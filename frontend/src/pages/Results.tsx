import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';
import type { Assessment } from '../types/api';

export function Results() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [loading, setLoading] = useState(true);
  const [completing, setCompleting] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [loadingVideo, setLoadingVideo] = useState(false);
  const [annotatedVideoUrl, setAnnotatedVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState('');

  useEffect(() => () => {
    if (annotatedVideoUrl) URL.revokeObjectURL(annotatedVideoUrl);
  }, [annotatedVideoUrl]);

  useEffect(() => {
    loadResults();
  }, [id]);

  const loadResults = async () => {
    if (!id) return;
    setLoading(true);
    try {
      const data = await assessmentsApi.get(parseInt(id, 10));
      setAssessment(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load assessment results');
    } finally {
      setLoading(false);
    }
  };

  const handleComplete = async () => {
    if (!id) return;
    setCompleting(true);
    setError('');
    try {
      await assessmentsApi.complete(parseInt(id, 10));
      await loadResults();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to complete assessment');
    } finally {
      setCompleting(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (!id) return;
    setDownloadingPdf(true);
    setError('');
    try {
      const blob = await assessmentsApi.downloadReportPdf(parseInt(id, 10));
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Parkinson_Research_Report_${id}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate PDF report');
    } finally {
      setDownloadingPdf(false);
    }
  };

  const handleViewVideo = async () => {
    if (!id) return;
    setLoadingVideo(true);
    setError('');
    try {
      const blob = await assessmentsApi.getSpiralVideoBlob(Number(id), 'annotated');
      setAnnotatedVideoUrl(URL.createObjectURL(blob));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not retrieve the authenticated annotated video.');
    } finally {
      setLoadingVideo(false);
    }
  };

  if (loading) return <div className="loading"><p>Loading results…</p></div>;
  if (!assessment) return <div className="container"><p>Assessment not found</p></div>;

  const steps = assessment.steps;
  const fusion = assessment.results?.fusion;
  const isCompleted = assessment.status === 'completed';

  const hasVerifiedPrediction = (
    (steps.tremor.status === 'completed' && steps.tremor.outcome?.model_verified) ||
    (steps.spiral.status === 'completed' && steps.spiral.outcome?.model_verified) ||
    (steps.speech.status === 'completed' && steps.speech.outcome?.model_verified)
  );

  return (
    <div className="container" style={{ maxWidth: '820px' }}>
      <button onClick={() => navigate('/dashboard')} className="btn btn-secondary" style={{ marginBottom: '1.25rem' }}>
        ← Back to Dashboard
      </button>

      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '1rem', marginBottom: '1.5rem' }}>
          <div>
            <span style={{ fontSize: '0.8rem', fontWeight: '700', color: 'var(--sage-green)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Assessment #{assessment.id}
            </span>
            <h2 style={{ color: 'var(--sage-dark)', fontSize: '1.6rem', marginTop: '0.2rem' }}>
              Research Evaluation Results
            </h2>
          </div>
          <span className={`status-badge status-${assessment.status}`}>
            {assessment.status === 'completed' ? 'Finalized' : 'In Progress'}
          </span>
        </div>

        {/* Fused / Multimodal Outcome Banner */}
        {isCompleted && fusion ? (
          <div style={{ backgroundColor: 'var(--success-bg)', border: '1px solid var(--success)', padding: '1.5rem', borderRadius: '8px', marginBottom: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: '700', color: 'var(--success)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                {fusion.aggregation_type === 'single_modality_probability' ? 'Single Available Modality Result' : 'AUC-Weighted Decision-Level Aggregate'}
              </span>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-light)' }}>
                Weights: {fusion.available_modalities.map((m: string) => `${m}: ${(fusion.effective_weights[m] * 100).toFixed(0)}%`).join(', ')}
              </span>
            </div>
            <h3 style={{ fontSize: '1.35rem', color: 'var(--text-dark)', marginBottom: '0.5rem' }}>
              Model Prediction: <strong>{fusion.prediction === 1 ? "Parkinson's Class Indicator" : 'Control Class Indicator'}</strong>
            </h3>
            <p style={{ fontSize: '1.1rem', color: 'var(--sage-dark)', fontWeight: '600', marginBottom: '0.75rem' }}>
              {fusion.aggregation_type === 'single_modality_probability' ? 'Classifier probability' : 'Weighted score (not patient-level calibrated probability)'}: {(fusion.probability * 100).toFixed(1)}% (Threshold: {(fusion.threshold ?? 0.5).toFixed(2)})
            </p>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-light)', lineHeight: '1.4', margin: 0 }}>
              <em>Interpretation:</em> {fusion.score_interpretation || 'Research output only; patient-level multimodal performance has not been measured.'} Included modalities: {fusion.available_modalities.join(', ')}. Unavailable or omitted: {(fusion.missing_modalities || []).join(', ') || 'none'}.
            </p>
          </div>
        ) : (
          <div style={{ backgroundColor: '#FAFAF7', border: '1px solid var(--border-subtle)', padding: '1.25rem', borderRadius: '8px', marginBottom: '2rem' }}>
            <p style={{ color: 'var(--text-dark)', fontSize: '0.95rem', margin: 0 }}>
              {hasVerifiedPrediction
                ? 'Verified modality predictions are saved. Finalize to aggregate only those predictions; unavailable modalities will be omitted and their configured weights renormalized.'
                : 'No compatible trained-model prediction is saved yet. Available extracted features and experimental video observations remain visible below; they are not fused.'}
            </p>
          </div>
        )}

        {/* Individual Modality Breakdowns */}
        <div style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--sage-dark)', fontSize: '1.2rem', marginBottom: '1rem' }}>
            Modality Breakdown
          </h3>

          {/* Tremor Step Result */}
          <div style={{ backgroundColor: '#FAFAF7', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '1.25rem', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h4 style={{ color: 'var(--text-dark)', fontSize: '1rem' }}>1. Tremor Assessment</h4>
              <span className={`status-badge status-${steps.tremor.status}`}>
                {steps.tremor.status}
              </span>
            </div>
            {steps.tremor.status === 'completed' && steps.tremor.outcome && typeof steps.tremor.outcome.probability === 'number' ? (
              <div>
                <p style={{ color: 'var(--sage-dark)', fontWeight: '600', fontSize: '0.95rem', marginBottom: '0.25rem' }}>
                  Classifier Probability: {(steps.tremor.outcome.probability * 100).toFixed(1)}% ({steps.tremor.outcome.prediction === 1 ? "Parkinson's class" : 'Control class'})
                </p>
                <p style={{ color: 'var(--text-light)', fontSize: '0.85rem', margin: 0 }}>
                  Model: Calibrated RBF SVM (C=1.0, gamma=0.1) | Features: 7 UPDRS III items (Items 20 & 21) | Test AUC: 0.8997
                </p>
              </div>
            ) : steps.tremor.error ? (
              <p style={{ color: 'var(--warning)', fontSize: '0.85rem', margin: 0 }}>{steps.tremor.error}</p>
            ) : (
              <p style={{ color: 'var(--text-light)', fontSize: '0.85rem', margin: 0 }}>
                Not submitted yet. Enter 7 UPDRS items to run classifier.
              </p>
            )}
            {steps.tremor.outcome?.input_features && <details style={{ marginTop: '0.75rem' }}><summary>Saved UPDRS inputs</summary><dl>{Object.entries(steps.tremor.outcome.input_features).map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{String(value)}</dd></div>)}</dl></details>}
          </div>

          {/* Handwriting / Spiral Step Result */}
          <div style={{ backgroundColor: '#FAFAF7', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '1.25rem', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h4 style={{ color: 'var(--text-dark)', fontSize: '1rem' }}>2. Handwriting / Spiral Test</h4>
              <span className={`status-badge status-${steps.spiral.status}`}>
                {steps.spiral.status}
              </span>
            </div>
            {steps.spiral.status === 'completed' && steps.spiral.outcome && typeof steps.spiral.outcome.probability === 'number' ? (
              <div>
                <p style={{ color: 'var(--sage-dark)', fontWeight: '600', fontSize: '0.95rem', marginBottom: '0.25rem' }}>
                  Classifier Probability: {(steps.spiral.outcome.probability * 100).toFixed(1)}% ({steps.spiral.outcome.prediction === 1 ? "Parkinson's class" : 'Control class'})
                </p>
                <p style={{ color: 'var(--text-light)', fontSize: '0.85rem', margin: 0 }}>
                  Model: Calibrated RBF SVM (C=0.1, gamma=0.01) | Schema: 16 PaHaW kinematic features | Test AUC: 0.9322
                </p>
              </div>
            ) : steps.spiral.error ? (
              <p style={{ color: 'var(--warning)', fontSize: '0.85rem', margin: 0 }}>
                {steps.spiral.error}
              </p>
            ) : (
              <p style={{ color: 'var(--text-light)', fontSize: '0.85rem', margin: 0 }}>
                No compatible digitizer classifier prediction is saved. Canvas observations and video analysis are shown separately when available.
              </p>
            )}
            {steps.spiral.outcome?.canvas_telemetry && (() => {
              const canvas = steps.spiral.outcome.canvas_telemetry;
              return <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: '#F4F7F2', borderRadius: 6 }}>
                <strong>Saved browser-canvas observations (experimental, not classifier-compatible)</strong>
                <p>{canvas.sample_count} samples · {canvas.stroke_count} stroke(s) · {canvas.path_length_px} px path · {canvas.duration_ms} ms</p>
                <p>Mean speed: {canvas.mean_speed_px_per_second ?? 'unavailable'} px/s · acceleration: {canvas.mean_acceleration_px_per_second2 ?? 'unavailable'} · jerk: {canvas.mean_jerk_px_per_second3 ?? 'unavailable'} · browser pressure samples: {canvas.pressure_samples}</p>
                <details><summary>Feature coverage and limitations</summary><ul>{canvas.limitations.map((item: string) => <li key={item}>{item}</li>)}</ul></details>
              </div>;
            })()}
            {steps.spiral.outcome?.video_analysis && (() => {
              const video = steps.spiral.outcome.video_analysis;
              return <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: '#F1F5F8', borderRadius: 6 }}>
                <strong>Spiral video observations ({video.status})</strong>
                {video.trajectory && <p>Tracked candidates: {video.trajectory.tracked_points} · visible-change path: {video.trajectory.path_length_px ?? 'insufficient tracking'} px · mean speed: {video.trajectory.mean_speed_px_per_second ?? 'unavailable'} px/s</p>}
                <p>Red movement candidates: {video.candidate_movement_events?.length ?? 0} · blue visible-appearance observations: {video.visible_appearance_events?.length ?? 0}</p>
                {video.error?.message && <p role="alert">{video.error.message}</p>}
                {video.status === 'completed' && <button type="button" className="btn btn-secondary" onClick={handleViewVideo} disabled={loadingVideo}>{loadingVideo ? 'Loading annotated video…' : 'View Annotated Video'}</button>}
                {annotatedVideoUrl && <video controls preload="metadata" src={annotatedVideoUrl} style={{ display: 'block', width: '100%', maxHeight: 400, marginTop: '0.75rem', background: '#111' }}>Video playback is unavailable in this browser.</video>}
                <details><summary>Video-analysis limitations</summary><ul>{(video.limitations || []).map((item: string) => <li key={item}>{item}</li>)}</ul></details>
              </div>;
            })()}
          </div>

          {/* Speech Step Result */}
          <div style={{ backgroundColor: '#FAFAF7', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '1.25rem', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h4 style={{ color: 'var(--text-dark)', fontSize: '1rem' }}>3. Speech Assessment</h4>
              <span className={`status-badge status-${steps.speech.status}`}>
                {steps.speech.status}
              </span>
            </div>
            {steps.speech.status === 'completed' && steps.speech.outcome && typeof steps.speech.outcome.probability === 'number' ? (
              <div>
                <p style={{ color: 'var(--sage-dark)', fontWeight: '600', fontSize: '0.95rem', marginBottom: '0.25rem' }}>
                  Classifier Probability: {(steps.speech.outcome.probability * 100).toFixed(1)}%
                </p>
              </div>
            ) : steps.speech.error ? (
              <div>
                <p style={{ color: 'var(--warning)', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                  {steps.speech.error}
                </p>
              </div>
            ) : (
              <p style={{ color: 'var(--text-light)', fontSize: '0.85rem', margin: 0 }}>
                Not submitted yet.
              </p>
            )}
            {steps.speech.outcome?.feature_values && <div style={{ marginTop: '0.75rem' }}>
              <p>{steps.speech.outcome.feature_count} {steps.speech.outcome.feature_schema} features extracted from {steps.speech.outcome.recording_mode}. Audio was not retained.</p>
              {steps.speech.outcome.prediction_status === 'unavailable' && <p role="status">Prediction unavailable: no compatible trained model exists for this audio feature schema and recording mode. The legacy 45-feature SVM was not used.</p>}
              <details><summary>View saved speech features</summary><dl style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.4rem 1rem' }}>{Object.entries(steps.speech.outcome.feature_values).map(([name, value]) => <div key={name}><dt style={{ fontWeight: 600 }}>{name}</dt><dd style={{ margin: 0 }}>{typeof value === 'number' ? value.toPrecision(6) : String(value)}</dd></div>)}</dl></details>
            </div>}
          </div>
        </div>

        {error && (
          <div style={{ backgroundColor: 'var(--error-bg)', border: '1px solid var(--error)', padding: '0.75rem 1rem', borderRadius: '6px', marginBottom: '1.25rem' }}>
            <p className="error" style={{ margin: 0 }}>{error}</p>
          </div>
        )}

        {/* Action Controls */}
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
          {!isCompleted && hasVerifiedPrediction && (
            <button
              onClick={handleComplete}
              className="btn btn-primary"
              style={{ flex: 1, minWidth: '220px', padding: '0.85rem' }}
              disabled={completing}
            >
              {completing ? 'Calculating Fusion…' : 'Finalize Assessment & Run Fusion'}
            </button>
          )}

          <button
            onClick={handleDownloadPdf}
            className="btn btn-secondary"
            style={{ flex: 1, minWidth: '220px', padding: '0.85rem', borderColor: 'var(--sage-green)', color: 'var(--sage-dark)' }}
            disabled={downloadingPdf}
          >
            {downloadingPdf ? 'Generating PDF…' : '📄 Download Research PDF Report'}
          </button>
        </div>
      </div>
    </div>
  );
}
