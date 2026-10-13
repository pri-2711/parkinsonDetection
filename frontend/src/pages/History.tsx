import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';
import type { Assessment } from '../types/api';

export function History() {
  const navigate = useNavigate();
  const [assessments, setAssessments] = useState<Assessment[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloadingId, setDownloadingId] = useState<number | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const data = await assessmentsApi.listCompleted(50);
      setAssessments(data.assessments);
    } catch (err) {
      setError('Failed to load assessment history');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPdf = async (assessmentId: number) => {
    setDownloadingId(assessmentId);
    setError('');
    try {
      const blob = await assessmentsApi.downloadReportPdf(assessmentId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Parkinson_Research_Report_${assessmentId}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(`Failed to generate PDF for assessment #${assessmentId}`);
    } finally {
      setDownloadingId(null);
    }
  };

  if (loading) return <div className="loading"><p>Loading history…</p></div>;

  return (
    <div className="container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h1 style={{ color: 'var(--sage-dark)', fontSize: '1.85rem', marginBottom: '0.3rem' }}>
            Assessment History
          </h1>
          <p style={{ color: 'var(--text-light)', fontSize: '0.95rem' }}>
            Longitudinal log of completed assessments and downloadable research reports.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={() => navigate('/dashboard')}>
          Back to Dashboard
        </button>
      </div>

      {error && (
        <div style={{ backgroundColor: 'var(--error-bg)', border: '1px solid var(--error)', padding: '0.75rem 1rem', borderRadius: '6px', marginBottom: '1.5rem' }}>
          <p className="error" style={{ margin: 0 }}>{error}</p>
        </div>
      )}

      {assessments.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <p style={{ color: 'var(--text-light)', fontSize: '1rem', marginBottom: '1.25rem' }}>
            No completed assessments found in your account history.
          </p>
          <button className="btn btn-primary" onClick={() => navigate('/dashboard')}>
            Go to Dashboard & Start Assessment
          </button>
        </div>
      ) : (
        assessments.map((assessment) => (
          <div key={assessment.id} className="card" style={{ marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
              <div>
                <strong style={{ fontSize: '1.1rem', color: 'var(--text-dark)' }}>
                  Assessment #{assessment.id}
                </strong>
                <p style={{ color: 'var(--text-light)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
                  Completed: {assessment.completed_at ? new Date(assessment.completed_at).toLocaleString() : 'Recently'}
                </p>
              </div>
              <span className={`status-badge status-${assessment.status}`}>
                {assessment.status}
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', fontSize: '0.875rem', backgroundColor: '#FAFAF7', padding: '0.85rem 1rem', borderRadius: '6px', marginBottom: '1rem' }}>
              <div>
                <strong>Tremor:</strong>{' '}
                <span className={`status-badge status-${assessment.steps.tremor.status}`}>
                  {assessment.steps.tremor.status}
                </span>
              </div>
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
            </div>

            {assessment.results?.fusion && (
              <div style={{ marginBottom: '1rem', padding: '0.75rem 1rem', backgroundColor: 'var(--success-bg)', borderRadius: '6px' }}>
                <strong style={{ color: 'var(--sage-dark)' }}>Fused Model Result:</strong>{' '}
                <span>
                  {assessment.results.fusion.prediction === 1 ? "Parkinson's Class Indicator" : 'Control Class'}{' '}
                  (<strong>{(assessment.results.fusion.probability * 100).toFixed(1)}%</strong>)
                </span>
              </div>
            )}

            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end', paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)' }}>
              <button
                className="btn btn-secondary"
                style={{ padding: '0.5rem 1rem', fontSize: '0.875rem' }}
                onClick={() => navigate(`/results/${assessment.id}`)}
              >
                View Full Results →
              </button>
              <button
                className="btn btn-primary"
                style={{ padding: '0.5rem 1rem', fontSize: '0.875rem' }}
                onClick={() => handleDownloadPdf(assessment.id)}
                disabled={downloadingId === assessment.id}
              >
                {downloadingId === assessment.id ? 'Generating…' : '📄 Download PDF Report'}
              </button>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
