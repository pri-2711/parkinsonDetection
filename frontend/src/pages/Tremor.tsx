import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';

export function Tremor() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [location, setLocation] = useState('');
  const [intensity, setIntensity] = useState(5);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!id) return;

    setSubmitting(true);
    setError('');

    try {
      const response = await fetch('/api/tremor/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ location, intensity }),
      });

      const data = await response.json();

      if (response.ok) {
        await assessmentsApi.updateTremor(parseInt(id), {
          prototype: true,
          classification: null,
          updrs_estimate: null,
          data: { location, intensity },
        }, null);
        navigate('/dashboard');
      } else {
        throw new Error(data.error?.message || 'Submission failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Submission failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: '600px' }}>
      <button onClick={() => navigate('/dashboard')} className="btn btn-secondary" style={{ marginBottom: '1rem' }}>
        ← Back to Dashboard
      </button>

      <div className="card">
        <h2 style={{ marginBottom: '1.5rem', color: 'var(--sage-dark)' }}>Tremor Assessment</h2>

        <div style={{ backgroundColor: 'var(--cream-dark)', padding: '1rem', borderRadius: '4px', marginBottom: '1.5rem' }}>
          <p style={{ color: 'var(--text-light)', fontSize: '0.875rem' }}>
            <strong>Note:</strong> This is a self-reported assessment. The trained tremor model expects UPDRS items 20/21,
            not body location and perceived intensity. Classification is not available for this input.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-light)' }}>
              Body Location
            </label>
            <select
              className="input"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              required
            >
              <option value="">Select location...</option>
              <option value="left_hand">Left Hand</option>
              <option value="right_hand">Right Hand</option>
              <option value="left_leg">Left Leg</option>
              <option value="right_leg">Right Leg</option>
              <option value="jaw">Jaw</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-light)' }}>
              Perceived Intensity: {intensity}/10
            </label>
            <input
              type="range"
              min="0"
              max="10"
              value={intensity}
              onChange={(e) => setIntensity(parseInt(e.target.value))}
              style={{ width: '100%', marginBottom: '1rem' }}
            />
          </div>

          {error && <p className="error">{error}</p>}

          <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={submitting}>
            {submitting ? 'Submitting...' : 'Submit Assessment'}
          </button>
        </form>
      </div>
    </div>
  );
}
