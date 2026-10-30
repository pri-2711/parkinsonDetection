import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';
import type { Assessment, AssessmentStep } from '../types/api';

type Modality = 'tremor' | 'spiral' | 'speech';

// What each step actually did. Saved observations and experimental results are not classifier predictions.
function stepLabel(modality: Modality, step: AssessmentStep | undefined): { text: string; tone: string } {
  const status = step?.status ?? 'pending';
  if (status === 'failed') return { text: 'Needs Attention', tone: 'failed' };
  if (status === 'processing') return { text: 'Processing', tone: 'processing' };
  if (status === 'completed') {
    if (modality === 'speech') return { text: 'Experimental Result', tone: 'completed' };
    if (modality === 'spiral' && step?.outcome?.probability == null) return { text: 'Observations Saved', tone: 'unavailable' };
    return { text: 'Completed', tone: 'completed' };
  }
  if (status === 'unavailable') return { text: 'Observations Saved', tone: 'unavailable' };
  return { text: 'Pending', tone: 'pending' };
}

interface ModalityCardProps {
  title: string;
  description: string;
  modality: Modality;
  step: AssessmentStep | undefined;
  onOpen: () => void;
}

function ModalityCard({ title, description, modality, step, onOpen }: ModalityCardProps) {
  const label = stepLabel(modality, step);
  return (
    <article
      className="card modality-card"
      role="button"
      tabIndex={0}
      aria-label={`${title}: ${label.text}. Open assessment.`}
      onClick={onOpen}
      onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onOpen(); } }}
      style={{ display: 'flex', flexDirection: 'column', gap: '0.7rem', padding: '1.4rem', minHeight: 230, height: '100%' }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.6rem' }}>
        <h2 style={{ color: 'var(--sage-dark)', fontSize: '1.1rem', margin: 0 }}>{title}</h2>
        <span className={`status-badge status-${label.tone}`}>{label.text}</span>
      </div>
      <p style={{ color: 'var(--text-light)', fontSize: '0.92rem', margin: 0, flex: 1 }}>{description}</p>
      <span style={{ color: 'var(--sage-dark)', fontWeight: 600, fontSize: '0.9rem' }}>Open assessment →</span>
    </article>
  );
}

export function Dashboard() {
  const navigate = useNavigate();
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    loadAssessment();
  }, []);

  const loadAssessment = async () => {
    try {
      const history = await assessmentsApi.listHistory(1);
      if (history.assessments.length > 0) {
        const latest = history.assessments[0];
        if (latest.status === 'in_progress') {
          setAssessment(latest);
          setLoading(false);
          return;
        }
      }
      const newAssessment = await assessmentsApi.create();
      setAssessment(newAssessment);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load or create an assessment.');
    } finally {
      setLoading(false);
    }
  };

  const handleStartNew = async () => {
    setCreating(true);
    try {
      const newAssessment = await assessmentsApi.create();
      setAssessment(newAssessment);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create a new assessment.');
    } finally {
      setCreating(false);
    }
  };

  if (loading) return <div className="loading"><p>Loading dashboard…</p></div>;

  const steps = assessment?.steps;
  const processed = (step: AssessmentStep | undefined) => step?.status === 'completed' || step?.status === 'unavailable';
  const anyPrediction = Object.values(steps ?? {}).some((step) => step?.status === 'completed' && step?.outcome?.model_verified);
  const allProcessed = !!steps && processed(steps.tremor) && processed(steps.spiral) && processed(steps.speech);
  const finalized = assessment?.status === 'completed';
  const readyToFinalize = !finalized && allProcessed && anyPrediction;

  return (
    <div className="container" style={{ maxWidth: 1040 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ maxWidth: 640 }}>
          <h1 style={{ color: 'var(--sage-dark)', fontSize: '1.7rem', margin: '0 0 0.35rem' }}>Assessment Dashboard</h1>
          <p style={{ color: 'var(--text-light)', fontSize: '0.95rem', margin: 0 }}>
            Record tremor ratings, a spiral drawing and a speech sample for one participant, then review the results together.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center', flexWrap: 'wrap' }}>
          {finalized && <span className="status-badge status-completed">Finalized</span>}
          {readyToFinalize && <span className="status-badge status-processing">Ready to Finalize</span>}
          <button className="btn btn-secondary" onClick={handleStartNew} disabled={creating} style={{ whiteSpace: 'nowrap' }}>
            {creating ? 'Starting…' : 'Start new assessment'}
          </button>
        </div>
      </div>

      {error && <div role="alert" className="error" style={{ marginBottom: '1.25rem' }}>{error}</div>}

      {assessment && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(270px, 1fr))', gap: '1.1rem', marginBottom: '1.75rem' }}>
          <ModalityCard
            title="Tremor Assessment"
            description="Seven UPDRS III ratings from items 20 and 21: rest tremor in the head and limbs, and action or postural tremor in the arms. A clinician provides the ratings, which the tremor classifier then uses."
            modality="tremor"
            step={steps?.tremor}
            onOpen={() => navigate(`/tremor/${assessment.id}`)}
          />
          <ModalityCard
            title="Handwriting & Spiral Assessment"
            description="Draw a spiral on screen or upload a video of spiral writing, and the app examines writing patterns and visual features. Only calibrated tablet measurements feed the handwriting classifier; screen and video results are experimental observations."
            modality="spiral"
            step={steps?.spiral}
            onOpen={() => navigate(`/handwriting/${assessment.id}`)}
          />
          <ModalityCard
            title="Speech Assessment"
            description="Record or upload a reading of the passage. The app analyses voice stability, pitch and speech variation. The reading-passage model is experimental and not validated for clinical use."
            modality="speech"
            step={steps?.speech}
            onOpen={() => navigate(`/speech/${assessment.id}`)}
          />
        </div>
      )}

      {assessment && (
        <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <button className="btn btn-primary" style={{ padding: '0.8rem 1.75rem' }} onClick={() => navigate(`/results/${assessment.id}`)}>
            {finalized ? 'View results' : readyToFinalize ? 'Review and finalize' : 'View results'}
          </button>
          <button className="btn btn-secondary" onClick={() => navigate('/research')} style={{ padding: '0.8rem 1.25rem' }}>
            Research &amp; Insights
          </button>
        </div>
      )}
    </div>
  );
}
