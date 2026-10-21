import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';
import { StatusCard } from '../components/StatusCard';
import type { Assessment } from '../types/api';

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

  const getStepStatus = (step: 'speech' | 'spiral' | 'tremor') => {
    if (!assessment) return 'pending';
    return assessment.steps[step].status;
  };

  const hasAnyActivity = assessment && Object.values(assessment.steps).some(
    (step) => step.status !== 'pending' || step.outcome !== null,
  );

  if (loading) return <div className="loading"><p>Loading dashboard…</p></div>;

  return (
    <div className="container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
        <div>
          <h1 style={{ color: 'var(--sage-dark)', fontSize: '1.85rem', marginBottom: '0.4rem' }}>
            Assessment Dashboard
          </h1>
          <p style={{ color: 'var(--text-light)', fontSize: '0.95rem' }}>
            Research evaluation suite. Complete assessments in any order. Outputs represent experimental classifier probabilities.
          </p>
        </div>
        <button
          className="btn btn-secondary"
          onClick={handleStartNew}
          disabled={creating}
          style={{ whiteSpace: 'nowrap' }}
        >
          {creating ? 'Starting…' : '+ Start New Assessment'}
        </button>
      </div>

      {error && <div role="alert" style={{ backgroundColor: 'var(--error-bg)', border: '1px solid var(--error)', padding: '0.75rem 1rem', borderRadius: '6px', marginBottom: '1.5rem' }}>{error}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        <StatusCard
          title="Tremor Assessment"
          status={getStepStatus('tremor')}
          description={
            assessment?.steps.tremor.status === 'completed'
              ? 'Completed — 7 UPDRS III motor items classified with calibrated RBF SVM.'
              : 'Enter seven clinician-rated UPDRS III tremor items (0–4) for calibrated model classification.'
          }
          onClick={() => assessment && navigate(`/tremor/${assessment.id}`)}
        />

        <StatusCard
          title="Handwriting / Spiral Test"
          status={getStepStatus('spiral')}
          description={
            assessment?.steps.spiral.status === 'completed'
              ? 'Completed — compatible tablet telemetry was classified with the calibrated RBF SVM.'
              : assessment?.steps.spiral.status === 'unavailable'
              ? 'Canvas observations and/or experimental video analysis are saved; these are not compatible with the digitizer classifier.'
              : 'Draw to save browser telemetry, or upload a separate video for experimental visual observations. Neither invents digitizer measurements.'
          }
          onClick={() => assessment && navigate(`/handwriting/${assessment.id}`)}
        />

        <StatusCard
          title="Speech Assessment"
          status={getStepStatus('speech')}
          description={
            assessment?.steps.speech.status === 'completed'
              ? 'Completed with a compatible trained classifier.'
              : assessment?.steps.speech.status === 'unavailable'
              ? `${assessment.steps.speech.outcome?.feature_count ?? 0} experimental speech features saved; no compatible classifier is trained for this mode.`
              : assessment?.steps.speech.status === 'failed'
              ? `Audio processing failed: ${assessment.steps.speech.error || 'review this step and try again.'}`
              : 'Record or upload a WAV. Features use speech_audio_v2; prediction needs a separately trained compatible model.'
          }
          onClick={() => assessment && navigate(`/speech/${assessment.id}`)}
        />
      </div>

      {assessment && (
        <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem' }}>
          <button
            className="btn btn-primary"
            style={{ padding: '0.85rem 2rem', fontSize: '1rem' }}
            onClick={() => navigate(`/results/${assessment.id}`)}
            disabled={!hasAnyActivity}
          >
            {hasAnyActivity ? 'View Assessment Results & Status' : 'Submit or review a modality to view results'}
          </button>
        </div>
      )}
    </div>
  );
}
