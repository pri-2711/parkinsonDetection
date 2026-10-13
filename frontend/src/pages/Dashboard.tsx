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
      console.error('Failed to load assessment:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleStartNew = async () => {
    setCreating(true);
    try {
      const newAssessment = await assessmentsApi.create();
      setAssessment(newAssessment);
    } catch (err) {
      console.error('Failed to create new assessment:', err);
    } finally {
      setCreating(false);
    }
  };

  const getStepStatus = (step: 'speech' | 'spiral' | 'tremor') => {
    if (!assessment) return 'pending';
    const stepData = assessment.steps[step];
    if (stepData.status === 'completed' && stepData.outcome) return 'completed';
    if (stepData.status === 'failed') return 'failed';
    return 'pending';
  };

  const hasAnyCompleted = assessment && (
    assessment.steps.speech.status === 'completed' ||
    assessment.steps.spiral.status === 'completed' ||
    assessment.steps.tremor.status === 'completed'
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
              ? 'Completed — Spiral kinematic telemetry extracted and classified with calibrated RBF SVM.'
              : 'Draw the Archimedean spiral on the interactive canvas, upload tablet CSV telemetry, or inspect video.'
          }
          onClick={() => assessment && navigate(`/handwriting/${assessment.id}`)}
        />

        <StatusCard
          title="Speech Assessment"
          status={getStepStatus('speech')}
          description={
            assessment?.steps.speech.status === 'completed'
              ? 'Completed — Acoustic features classified.'
              : assessment?.steps.speech.status === 'failed'
              ? 'Audio sample recorded & passage logged. (Precomputed 45-feature schema required for live prediction).'
              : 'Speak the standardized reading passage in-browser or upload WAV recording.'
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
            disabled={!hasAnyCompleted}
          >
            {hasAnyCompleted ? 'View & Finalize Results →' : 'Complete at least one test to view results'}
          </button>
        </div>
      )}
    </div>
  );
}
