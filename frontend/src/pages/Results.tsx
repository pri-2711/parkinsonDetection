import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';
import type { Assessment } from '../types/api';

export function Results() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadResults();
  }, [id]);

  const loadResults = async () => {
    if (!id) return;
    try {
      const data = await assessmentsApi.get(parseInt(id));
      setAssessment(data);
    } catch (err) {
      setError('Failed to load results');
    } finally {
      setLoading(false);
    }
  };

  const handleComplete = async () => {
    if (!id) return;
    try {
      await assessmentsApi.complete(parseInt(id));
      navigate('/history');
    } catch (err) {
      setError('Failed to complete assessment');
    }
  };

  if (loading) return <div className="loading"><p>Loading...</p></div>;

  if (!assessment) return <div className="container"><p>Assessment not found</p></div>;

  return (
    <div className="container" style={{ maxWidth: '800px' }}>
      <button onClick={() => navigate('/dashboard')} className="btn btn-secondary" style={{ marginBottom: '1rem' }}>
        ← Back to Dashboard
      </button>

      <div className="card">
        <h2 style={{ marginBottom: '1.5rem', color: 'var(--sage-dark)' }}>Assessment Results</h2>

        <div style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--sage-dark)', marginBottom: '1rem' }}>Modality Status</h3>

          <div style={{ marginBottom: '1rem' }}>
            <strong>Handwriting:</strong>{' '}
            <span className={`status-badge status-${assessment.steps.spiral.status}`}>
              {assessment.steps.spiral.status}
            </span>
            {assessment.steps.spiral.error && (
              <p className="warning" style={{ marginTop: '0.25rem' }}>{assessment.steps.spiral.error}</p>
            )}
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <strong>Speech:</strong>{' '}
            <span className={`status-badge status-${assessment.steps.speech.status}`}>
              {assessment.steps.speech.status}
            </span>
            {assessment.steps.speech.error && (
              <p className="warning" style={{ marginTop: '0.25rem' }}>{assessment.steps.speech.error}</p>
            )}
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <strong>Tremor:</strong>{' '}
            <span className={`status-badge status-${assessment.steps.tremor.status}`}>
              {assessment.steps.tremor.status}
            </span>
            {assessment.steps.tremor.outcome?.data && (
              <p style={{ marginTop: '0.25rem', color: 'var(--text-light)', fontSize: '0.875rem' }}>
                Location: {assessment.steps.tremor.outcome.data.location}, Intensity: {assessment.steps.tremor.outcome.data.intensity}/10
              </p>
            )}
          </div>
        </div>

        {assessment.status === 'completed' && assessment.results?.fusion ? (
          <div style={{ backgroundColor: 'var(--success)', padding: '1.5rem', borderRadius: '4px', marginBottom: '2rem' }}>
            <h3 style={{ color: 'white', marginBottom: '1rem' }}>Fused Classification</h3>
            <p style={{ color: 'white', fontSize: '1.25rem', marginBottom: '0.5rem' }}>
              Prediction: {assessment.results.fusion.prediction === 1 ? 'Parkinson\'s Indicated' : 'Control'}
            </p>
            <p style={{ color: 'white', fontSize: '0.875rem' }}>
              Probability: {(assessment.results.fusion.probability * 100).toFixed(1)}%
            </p>
            <p style={{ color: 'white', fontSize: '0.875rem', marginTop: '1rem' }}>
              Based on: {assessment.results.fusion.available_modalities.join(', ')}
            </p>
          </div>
        ) : (
          <div style={{ backgroundColor: 'var(--warning)', padding: '1.5rem', borderRadius: '4px', marginBottom: '2rem' }}>
            <p style={{ color: 'white' }}>
              Fused classification not available. At least one modality with a valid prediction is required.
            </p>
          </div>
        )}

        {error && <p className="error">{error}</p>}

        {assessment.status === 'in_progress' && (
          <button onClick={handleComplete} className="btn btn-primary" style={{ width: '100%' }}>
            Complete Assessment
          </button>
        )}
      </div>
    </div>
  );
}
