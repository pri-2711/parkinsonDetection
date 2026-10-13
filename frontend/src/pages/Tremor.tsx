import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';

interface TremorField {
  key: string;
  itemNumber: string;
  location: string;
  description: string;
}

const fields: TremorField[] = [
  {
    key: 'updrs_iii_motor_scale_specific_items_20_tremor_at_rest_head',
    itemNumber: 'UPDRS III Item 20',
    location: 'Head (face, lips, chin)',
    description: 'Rest tremor observed in head, chin, lips, or facial musculature.',
  },
  {
    key: 'updrs_iii_motor_scale_specific_items_20_tremor_at_rest_rue',
    itemNumber: 'UPDRS III Item 20',
    location: 'Right Upper Extremity (RUE)',
    description: 'Rest tremor observed in right arm, hand, or digits.',
  },
  {
    key: 'updrs_iii_motor_scale_specific_items_20_tremor_at_rest_lue',
    itemNumber: 'UPDRS III Item 20',
    location: 'Left Upper Extremity (LUE)',
    description: 'Rest tremor observed in left arm, hand, or digits.',
  },
  {
    key: 'updrs_iii_motor_scale_specific_items_20_tremor_at_rest_rle',
    itemNumber: 'UPDRS III Item 20',
    location: 'Right Lower Extremity (RLE)',
    description: 'Rest tremor observed in right leg, foot, or toes.',
  },
  {
    key: 'updrs_iii_motor_scale_specific_items_20_tremor_at_rest_lle',
    itemNumber: 'UPDRS III Item 20',
    location: 'Left Lower Extremity (LLE)',
    description: 'Rest tremor observed in left leg, foot, or toes.',
  },
  {
    key: 'updrs_iii_motor_scale_specific_items_21_action_or_postural_tremor_rue',
    itemNumber: 'UPDRS III Item 21',
    location: 'Action / Postural Tremor: Right Arm (RUE)',
    description: 'Tremor during sustained posture (arms outstretched) or action in right arm.',
  },
  {
    key: 'updrs_iii_motor_scale_specific_items_21_action_or_postural_tremor_lue',
    itemNumber: 'UPDRS III Item 21',
    location: 'Action / Postural Tremor: Left Arm (LUE)',
    description: 'Tremor during sustained posture (arms outstretched) or action in left arm.',
  },
];

const ratingDescriptions: Record<number, string> = {
  0: '0 — None: No tremor present (normal)',
  1: '1 — Slight: Slight and infrequently present',
  2: '2 — Mild: Mild amplitude & persistent, or moderate & intermittent',
  3: '3 — Moderate: Moderate amplitude and present most of the time',
  4: '4 — Marked: Marked amplitude and present most of the time',
};

export function Tremor() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [ratings, setRatings] = useState<Record<string, number | null>>({});
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleSelectRating = (key: string, value: number) => {
    setRatings((prev) => ({ ...prev, [key]: value }));
    setError('');
  };

  const allCompleted = fields.every((f) => ratings[f.key] !== undefined && ratings[f.key] !== null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!id) return;

    // Validate that all 7 fields have valid 0-4 values
    const missingFields = fields.filter((f) => ratings[f.key] === undefined || ratings[f.key] === null);
    if (missingFields.length > 0) {
      setError(`Please complete all 7 items. Missing: ${missingFields.map((f) => f.location).join(', ')}`);
      return;
    }

    setSubmitting(true);
    setError('');

    try {
      const payload: Record<string, number> = {};
      for (const f of fields) {
        const val = ratings[f.key];
        if (val === null || val === undefined || val < 0 || val > 4 || !Number.isInteger(val)) {
          throw new Error(`Rating for ${f.location} must be an integer between 0 and 4.`);
        }
        payload[f.key] = val;
      }

      await assessmentsApi.predictTremor(parseInt(id, 10), payload);
      navigate(`/dashboard`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Tremor classification failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: '780px' }}>
      <button onClick={() => navigate('/dashboard')} className="btn btn-secondary" style={{ marginBottom: '1.25rem' }}>
        ← Back to Dashboard
      </button>

      <div className="card">
        <div style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '1rem', marginBottom: '1.5rem' }}>
          <h2 style={{ color: 'var(--sage-dark)', fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            Tremor Assessment (UPDRS III Motor Scale)
          </h2>
          <p style={{ color: 'var(--text-light)', fontSize: '0.9rem', lineHeight: '1.5' }}>
            This module evaluates tremor patterns using the trained research classifier (<code>tremor_svm_tuned.joblib</code>).
            Enter the <strong>seven standardized clinician ratings</strong> from UPDRS III items 20 and 21.
            Each rating must be a discrete integer score from <strong>0 (normal/none)</strong> to <strong>4 (marked severity)</strong>.
          </p>
        </div>

        <div style={{ backgroundColor: 'var(--cream-dark)', padding: '0.85rem 1rem', borderRadius: '6px', marginBottom: '1.5rem' }}>
          <p style={{ color: 'var(--text-dark)', fontSize: '0.85rem', margin: 0 }}>
            <strong>Protocol Notice:</strong> In clinical research workflows, UPDRS ratings are recorded by a certified movement disorder specialist.
            These values serve as inputs to a trained research classifier and return an estimated class probability, not a definitive diagnosis.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          {fields.map((field) => {
            const currentValue = ratings[field.key];
            const isSelected = currentValue !== undefined && currentValue !== null;

            return (
              <div key={field.key} className="rating-item">
                <div className="rating-header">
                  <div>
                    <span style={{ fontSize: '0.8rem', fontWeight: '700', color: 'var(--sage-green)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      {field.itemNumber}
                    </span>
                    <h4 className="rating-label">{field.location}</h4>
                    <p style={{ color: 'var(--text-light)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
                      {field.description}
                    </p>
                  </div>
                  <div className={`rating-badge ${isSelected ? '' : 'empty'}`} title="Selected Rating">
                    {isSelected ? currentValue : '—'}
                  </div>
                </div>

                <div className="rating-segmented-group" role="radiogroup" aria-label={field.location}>
                  {[0, 1, 2, 3, 4].map((score) => {
                    const active = currentValue === score;
                    return (
                      <button
                        key={score}
                        type="button"
                        role="radio"
                        aria-checked={active}
                        className={`rating-button ${active ? 'selected' : ''}`}
                        onClick={() => handleSelectRating(field.key, score)}
                      >
                        <div>
                          <span style={{ fontSize: '1.1rem', display: 'block', fontWeight: '700' }}>{score}</span>
                          <span style={{ fontSize: '0.72rem', opacity: 0.85, display: 'block' }}>
                            {score === 0 ? 'None' : score === 1 ? 'Slight' : score === 2 ? 'Mild' : score === 3 ? 'Mod' : 'Marked'}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>

                {isSelected && (
                  <p style={{ color: 'var(--sage-dark)', fontSize: '0.8rem', marginTop: '0.5rem', fontWeight: '500' }}>
                    {ratingDescriptions[currentValue!]}
                  </p>
                )}
              </div>
            );
          })}

          {error && (
            <div style={{ backgroundColor: 'var(--error-bg)', border: '1px solid var(--error)', padding: '0.75rem 1rem', borderRadius: '6px', marginBottom: '1.25rem' }}>
              <p className="error" style={{ margin: 0 }}>{error}</p>
            </div>
          )}

          <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
            <button
              type="submit"
              className="btn btn-primary"
              style={{ flex: 1, padding: '0.85rem' }}
              disabled={submitting || !allCompleted}
            >
              {submitting ? 'Running Classifier…' : allCompleted ? 'Submit Ratings & Run Classifier' : 'Select all 7 ratings to submit'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
