import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';
import { VideoObservations } from '../components/VideoObservations';
import { Icon } from '../components/Icon';
import type { Assessment, AssessmentSummary } from '../types/api';
import { fusionDisplayMode, formatPercent, speechStepSummary } from '../features/fusionDisplay.js';

type Tone = 'success' | 'warning' | 'error' | 'muted';

const TONE_COLOR: Record<Tone, string> = {
  success: 'var(--sage-dark)',
  warning: 'var(--warning)',
  error: 'var(--error)',
  muted: 'var(--text-light)',
};

function StepBadge({ status }: { status: string }) {
  const labels: Record<string, string> = {
    pending: 'Not submitted', processing: 'Processing', completed: 'Completed', unavailable: 'Features only', failed: 'Failed',
  };
  return <span className={`status-badge status-${status}`}>{labels[status] ?? status}</span>;
}

function ModalityCard({ title, status, children }: { title: string; status: string; children: React.ReactNode }) {
  return (
    <section style={{ backgroundColor: '#FAFAF7', border: '1px solid var(--border-subtle)', borderRadius: 8, padding: '1rem 1.15rem', marginBottom: '0.9rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem', marginBottom: '0.45rem' }}>
        <h3 style={{ color: 'var(--text-dark)', fontSize: '1rem', margin: 0 }}>{title}</h3>
        <StepBadge status={status} />
      </div>
      {children}
    </section>
  );
}

export function Results() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [summary, setSummary] = useState<AssessmentSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [completing, setCompleting] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    loadResults();
  }, [id]);

  const loadResults = async () => {
    if (!id) return;
    setLoading(true);
    try {
      const [data, summaryData] = await Promise.all([
        assessmentsApi.get(parseInt(id, 10)),
        assessmentsApi.getSummary(parseInt(id, 10)).catch(() => null),
      ]);
      setAssessment(data);
      setSummary(summaryData);
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
      setError(err instanceof Error ? err.message : 'Failed to finalize assessment');
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

  if (loading) return <div className="loading"><p>Loading results…</p></div>;
  if (!assessment) return <div className="container"><p>Assessment not found</p></div>;

  const steps = assessment.steps;
  const fusion = assessment.results?.fusion;
  const fusionMode = fusionDisplayMode(fusion);
  const isCompleted = assessment.status === 'completed';
  const speech = speechStepSummary(steps.speech);
  const processed = (status: string) => status === 'completed' || status === 'unavailable';
  const allProcessed = processed(steps.tremor.status) && processed(steps.spiral.status) && processed(steps.speech.status);
  const hasVerifiedPrediction = (
    (steps.tremor.status === 'completed' && steps.tremor.outcome?.model_verified) ||
    (steps.spiral.status === 'completed' && steps.spiral.outcome?.model_verified) ||
    (steps.speech.status === 'completed' && steps.speech.outcome?.model_verified)
  );

  let statusLabel = 'In Progress';
  if (isCompleted) statusLabel = 'Finalized';
  else if (allProcessed && hasVerifiedPrediction) statusLabel = 'Ready to Finalize';

  const video = steps.spiral.outcome?.video_analysis;
  const canvas = steps.spiral.outcome?.canvas_telemetry;
  const speechFeatures = steps.speech.outcome?.feature_values as Record<string, number> | undefined;

  return (
    <div className="container" style={{ maxWidth: '820px' }}>
      <button onClick={() => navigate('/dashboard')} className="btn btn-secondary" style={{ marginBottom: '1.25rem' }}>
        ← Back to Dashboard
      </button>

      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '1rem', marginBottom: '1.25rem' }}>
          <div>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--sage-green)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Assessment #{assessment.id}
            </span>
            <h2 style={{ color: 'var(--sage-dark)', fontSize: '1.5rem', margin: '0.2rem 0 0' }}>Results</h2>
          </div>
          <span className={`status-badge ${isCompleted ? 'status-completed' : statusLabel === 'Ready to Finalize' ? 'status-processing' : 'status-pending'}`}>
            {statusLabel}
          </span>
        </div>

        {isCompleted && fusion && fusionMode === 'single' && (
          <div style={{ backgroundColor: 'var(--success-bg)', border: '1px solid var(--success)', padding: '1rem 1.2rem', borderRadius: 8, marginBottom: '1.25rem' }}>
            <strong style={{ color: 'var(--success)' }}>Final result: one contributing model</strong>
            <p style={{ margin: '0.35rem 0 0', fontSize: '1rem' }}>
              {fusion.available_modalities[0] === 'speech' ? 'Speech (experimental)' : fusion.available_modalities[0] === 'tremor' ? 'Tremor' : 'Handwriting'}:
              {' '}<strong>{formatPercent(fusion.probability)}</strong> model probability
            </p>
          </div>
        )}
        {isCompleted && fusion && fusionMode === 'hidden' && (
          <div style={{ backgroundColor: '#FAFAF7', border: '1px solid var(--border-subtle)', padding: '1rem 1.2rem', borderRadius: 8, marginBottom: '1.25rem' }}>
            <strong style={{ color: 'var(--sage-dark)' }}>Final result: {fusion.available_modalities.length} modalities contributed</strong>
            <p style={{ margin: '0.35rem 0 0', fontSize: '0.92rem', color: 'var(--text-light)' }}>
              A combined probability is not shown because the modality cohorts are unmatched. Each modality's own result is listed below.
            </p>
          </div>
        )}
        {!isCompleted && statusLabel === 'Ready to Finalize' && (
          <p style={{ margin: '0 0 1.25rem', fontSize: '0.92rem', color: 'var(--text-light)' }}>
            All modalities have been processed. Finalize to combine the verified results into the assessment record.
          </p>
        )}
        {!isCompleted && statusLabel === 'In Progress' && (
          <p style={{ margin: '0 0 1.25rem', fontSize: '0.92rem', color: 'var(--text-light)' }}>
            Complete at least one modality with a verified prediction to finalize this assessment.
          </p>
        )}

        <ModalityCard title="Tremor (UPDRS III items 20–21)" status={steps.tremor.status}>
          {steps.tremor.status === 'completed' && typeof steps.tremor.outcome?.probability === 'number' ? (
            <p style={{ margin: 0, color: TONE_COLOR.success, fontSize: '0.95rem' }}>
              Classifier probability {formatPercent(steps.tremor.outcome.probability)} · {steps.tremor.outcome.prediction === 1 ? "Parkinson's-class direction" : 'control-class direction'}
            </p>
          ) : (
            <p style={{ margin: 0, color: TONE_COLOR.muted, fontSize: '0.9rem' }}>{steps.tremor.error || 'Not submitted yet.'}</p>
          )}
          {steps.tremor.outcome?.input_features && (
            <details style={{ marginTop: '0.6rem' }}>
              <summary>Entered ratings and model details</summary>
              <dl style={{ fontSize: '0.88rem' }}>
                {Object.entries(steps.tremor.outcome.input_features).map(([name, value]) => (
                  <div key={name} style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
                    <dt>{name.replace('updrs_iii_motor_scale_specific_items_', '').replace(/_/g, ' ')}</dt>
                    <dd style={{ margin: 0 }}>{String(value)}</dd>
                  </div>
                ))}
              </dl>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-light)' }}>
                Calibrated RBF SVM trained on clinician ratings. Cross-validated ROC-AUC 0.90 (folds 2–5, 104 participant rows). Research output only.
              </p>
            </details>
          )}
        </ModalityCard>

        <ModalityCard title="Handwriting & Spiral" status={steps.spiral.status}>
          {steps.spiral.status === 'completed' && typeof steps.spiral.outcome?.probability === 'number' ? (
            <p style={{ margin: 0, color: TONE_COLOR.success, fontSize: '0.95rem' }}>
              Classifier probability {formatPercent(steps.spiral.outcome.probability)}
            </p>
          ) : (
            <p style={{ margin: 0, color: TONE_COLOR.muted, fontSize: '0.9rem' }}>
              {canvas || video?.status === 'completed'
                ? 'Observations saved. The handwriting classifier needs calibrated tablet data, so no prediction is made from them.'
                : 'Not submitted yet.'}
            </p>
          )}
          {canvas && (
            <p style={{ margin: '0.5rem 0 0', fontSize: '0.9rem' }}>
              Drawing: {canvas.sample_count} points, {canvas.stroke_count} strokes (screen estimate, not calibrated).
            </p>
          )}
          {video && video.status === 'completed' && video.video && (
            <div style={{ marginTop: '0.75rem' }}>
              <VideoObservations analysis={video} assessmentId={assessment.id} />
            </div>
          )}
          {video && video.status === 'failed' && (
            <p role="alert" className="error" style={{ margin: '0.5rem 0 0', fontSize: '0.9rem' }}>
              Video analysis failed: {video.error?.message ?? 'no annotated output was created.'}
            </p>
          )}
          {canvas && (
            <details style={{ marginTop: '0.6rem' }}>
              <summary>Drawing measurement details</summary>
              <ul style={{ fontSize: '0.88rem' }}>
                <li>Mean speed: {canvas.mean_speed_px_per_second ?? 'unavailable'} px/s (screen units)</li>
                <li>Browser pressure samples: {canvas.pressure_samples} (normalized, not tablet-calibrated)</li>
                {canvas.limitations?.map((item: string) => <li key={item}>{item}</li>)}
              </ul>
            </details>
          )}
        </ModalityCard>

        <ModalityCard title="Speech (reading passage)" status={steps.speech.status}>
          <p style={{ margin: 0, color: TONE_COLOR[speech.tone], fontSize: '0.9rem' }}>{speech.text}</p>
          {speechFeatures && (
            <details style={{ marginTop: '0.6rem' }}>
              <summary>Extracted features ({Object.keys(speechFeatures).length})</summary>
              <dl style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.3rem 1rem', fontSize: '0.85rem' }}>
                {Object.entries(speechFeatures).map(([name, value]) => (
                  <div key={name}>
                    <dt style={{ fontWeight: 600 }}>{name}</dt>
                    <dd style={{ margin: 0 }}>{typeof value === 'number' ? value.toPrecision(6) : String(value)}</dd>
                  </div>
                ))}
              </dl>
            </details>
          )}
        </ModalityCard>

        {error && <p role="alert" className="error" style={{ marginTop: '0.5rem' }}>{error}</p>}

        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '1rem' }}>
          {!isCompleted && hasVerifiedPrediction && (
            <button onClick={handleComplete} className="btn btn-primary" style={{ flex: 1, minWidth: 220, padding: '0.8rem' }} disabled={completing}>
              {completing ? 'Finalizing…' : 'Finalize Assessment'}
            </button>
          )}
          <button
            onClick={handleDownloadPdf}
            className="btn btn-secondary"
            style={{ flex: 1, minWidth: 220, padding: '0.8rem', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}
            disabled={downloadingPdf}
          >
            {downloadingPdf ? 'Generating PDF…' : <><Icon name="download" />Download PDF report</>}
          </button>
        </div>

        <details style={{ marginTop: '1.25rem' }}>
          <summary>Limitations and scope</summary>
          <ul style={{ fontSize: '0.88rem', color: 'var(--text-light)' }}>
            {(summary?.limitations ?? []).map((item) => <li key={item}>{item}</li>)}
          </ul>
        </details>
      </div>
    </div>
  );
}
