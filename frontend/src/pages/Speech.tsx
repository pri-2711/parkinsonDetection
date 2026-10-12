import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';

export function Speech() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError('');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !id) return;

    setUploading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`/api/speech/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.status === 501) {
        await assessmentsApi.updateSpeech(parseInt(id), null, data.error?.message || 'Raw audio feature extractor unavailable');
      } else if (response.ok) {
        await assessmentsApi.updateSpeech(parseInt(id), data, null);
      } else {
        throw new Error(data.error?.message || 'Upload failed');
      }

      navigate(`/dashboard`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: '600px' }}>
      <button onClick={() => navigate('/dashboard')} className="btn btn-secondary" style={{ marginBottom: '1rem' }}>
        ← Back to Dashboard
      </button>

      <div className="card">
        <h2 style={{ marginBottom: '1.5rem', color: 'var(--sage-dark)' }}>Speech Test</h2>

        <div style={{ backgroundColor: 'var(--cream-dark)', padding: '1rem', borderRadius: '4px', marginBottom: '1.5rem' }}>
          <p style={{ color: 'var(--text-light)', fontSize: '0.875rem' }}>
            <strong>Important:</strong> The saved speech model expects precomputed acoustic features with specific MFCC, Delta, jitter, shimmer, and gender definitions.
            No verified raw-audio feature extractor matching the training contract is available. This test is currently unavailable for prediction.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-light)' }}>
              Upload audio file (WAV, 1-30 seconds)
            </label>
            <input
              type="file"
              accept="audio/wav"
              onChange={handleFileChange}
              disabled
              style={{ width: '100%', padding: '0.75rem', border: '1px solid var(--sage-light)', borderRadius: '4px', backgroundColor: 'var(--cream-dark)' }}
            />
          </div>

          {error && <p className="error">{error}</p>}

          <button type="submit" className="btn btn-primary" disabled style={{ width: '100%' }}>
            Upload (Currently Unavailable)
          </button>
        </form>
      </div>
    </div>
  );
}
