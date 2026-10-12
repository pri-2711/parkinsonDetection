import { useEffect, useState } from 'react';
import { assessmentsApi } from '../api/assessments';
import type { Assessment } from '../types/api';

export function History() {
  const [assessments, setAssessments] = useState<Assessment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const data = await assessmentsApi.listCompleted(20);
      setAssessments(data.assessments);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="loading"><p>Loading...</p></div>;

  return (
    <div className="container">
      <h1 style={{ color: 'var(--sage-dark)', marginBottom: '2rem' }}>Assessment History</h1>

      {assessments.length === 0 ? (
        <div className="card">
          <p style={{ color: 'var(--text-light)' }}>No completed assessments found.</p>
        </div>
      ) : (
        assessments.map((assessment) => (
          <div key={assessment.id} className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div>
                <strong>Assessment #{assessment.id}</strong>
                <p style={{ color: 'var(--text-light)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                  Completed: {new Date(assessment.completed_at || '').toLocaleDateString()}
                </p>
              </div>
              <span className={`status-badge status-${assessment.status}`}>
                {assessment.status}
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', fontSize: '0.875rem' }}>
              <div>
                <strong>Handwriting:</strong>{' '}
                <span className={`status-badge status-${assessment.steps.spiral.status}`}>
                  {assessment.steps.spiral.status}
                </span>
              </div>
              <div>
                <strong>Speech:</strong>{' '}
                <span className={`status-badge status-${assessment.steps.speech.status}`}>
                  {assessment.steps.speech.status}
                </span>
              </div>
              <div>
                <strong>Tremor:</strong>{' '}
                <span className={`status-badge status-${assessment.steps.tremor.status}`}>
                  {assessment.steps.tremor.status}
                </span>
              </div>
            </div>

            {assessment.results?.fusion && (
              <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--cream-dark)' }}>
                <strong>Fused Result:</strong>{' '}
                {assessment.results.fusion.prediction === 1 ? 'Parkinson\'s Indicated' : 'Control'}{' '}
                ({(assessment.results.fusion.probability * 100).toFixed(1)}%)
              </div>
            )}
          </div>
        ))
      )}
    </div>
  );
}
