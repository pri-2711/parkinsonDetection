import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';
import { StatusCard } from '../components/StatusCard';
import type { Assessment } from '../types/api';

export function Dashboard() {
  const navigate = useNavigate();
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [loading, setLoading] = useState(true);

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

  const getStepStatus = (step: 'speech' | 'spiral' | 'tremor') => {
    if (!assessment) return 'pending';
    const stepData = assessment.steps[step];
    if (stepData.status === 'completed' && stepData.outcome) return 'completed';
    if (stepData.status === 'failed') return 'failed';
    if (step === 'speech' || step === 'spiral') return 'unavailable';
    return 'pending';
  };

  if (loading) return <div className="loading"><p>Loading...</p></div>;

  return (
    <div className="container">
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ color: 'var(--sage-dark)', marginBottom: '0.5rem' }}>Assessment Dashboard</h1>
        <p style={{ color: 'var(--text-light)' }}>
          Complete the following tests in any order. This is a research prototype and not a clinical diagnostic tool.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
        <StatusCard
          title="Handwriting Test"
          status={getStepStatus('spiral')}
          description={
            getStepStatus('spiral') === 'unavailable'
              ? 'Camera video cannot provide tablet-specific features. This test is currently unavailable.'
              : assessment?.steps.spiral.status === 'completed'
              ? 'Completed'
              : 'Upload a spiral drawing video'
          }
          onClick={() => assessment && navigate(`/handwriting/${assessment.id}`)}
        />

        <StatusCard
          title="Speech Test"
          status={getStepStatus('speech')}
          description={
            getStepStatus('speech') === 'unavailable'
              ? 'Raw audio feature extraction is not yet implemented. This test is currently unavailable.'
              : assessment?.steps.speech.status === 'completed'
              ? 'Completed'
              : 'Record or upload audio'
          }
          onClick={() => assessment && navigate(`/speech/${assessment.id}`)}
        />

        <StatusCard
          title="Tremor Assessment"
          status={getStepStatus('tremor')}
          description={
            assessment?.steps.tremor.status === 'completed'
            ? 'Completed'
            : 'Report tremor location and intensity'
          }
          onClick={() => assessment && navigate(`/tremor/${assessment.id}`)}
        />
      </div>

      {assessment && assessment.status === 'in_progress' && (
        <div style={{ marginTop: '2rem', textAlign: 'center' }}>
          <button
            className="btn btn-primary"
            onClick={() => navigate(`/results/${assessment.id}`)}
            disabled={
              assessment.steps.speech.status !== 'completed' &&
              assessment.steps.spiral.status !== 'completed' &&
              assessment.steps.tremor.status !== 'completed'
            }
          >
            View Results
          </button>
        </div>
      )}
    </div>
  );
}
