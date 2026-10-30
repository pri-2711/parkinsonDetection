import { useEffect, useState } from 'react';
import { assessmentsApi } from '../api/assessments';
import { Icon } from './Icon';

interface Props {
  assessmentId: number;
  // Identifies the analysis run. A new analysis has a new id, so the player reloads its file.
  videoId?: string;
  // Only a completed analysis has an annotated file; the server returns 404 otherwise.
  analysisStatus: string | undefined;
}

// Loads the private annotated video through the authenticated API and offers playback and download.
// If the browser cannot decode the file, it says so instead of showing an empty player.
export function AnnotatedVideo({ assessmentId, analysisStatus, videoId }: Props) {
  const [url, setUrl] = useState<string | null>(null);
  const [extension, setExtension] = useState('webm');
  const [error, setError] = useState('');
  const [playable, setPlayable] = useState<boolean | null>(null);

  useEffect(() => {
    if (analysisStatus !== 'completed') return;
    let active = true;
    let created: string | null = null;
    assessmentsApi.getSpiralVideoBlob(assessmentId, 'annotated')
      .then((data) => {
        if (!active) return;
        created = URL.createObjectURL(data);
        setExtension(data.type.includes('mp4') ? 'mp4' : 'webm');
        setUrl(created);
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : 'The annotated video could not be retrieved.');
      });
    return () => {
      active = false;
      if (created) URL.revokeObjectURL(created);
    };
  }, [assessmentId, analysisStatus, videoId]);

  if (analysisStatus !== 'completed') return null;
  if (error) return <p role="alert" className="error" style={{ margin: '0.5rem 0' }}>{error}</p>;
  if (!url) return <p role="status" style={{ color: 'var(--text-light)', fontSize: '0.9rem' }}>Loading annotated video…</p>;

  return (
    <div style={{ marginTop: '0.75rem' }}>
      <video
        controls
        preload="metadata"
        src={url}
        onLoadedMetadata={(event) => setPlayable(Number.isFinite(event.currentTarget.duration) && event.currentTarget.duration > 0)}
        onError={() => setPlayable(false)}
        style={{ display: 'block', width: '100%', maxHeight: 420, background: '#111', borderRadius: 6 }}
      >
        Your browser cannot play this annotated video.
      </video>
      {playable === false && (
        <p role="alert" className="error" style={{ margin: '0.5rem 0' }}>
          This browser could not decode the annotated video. Download it to view it in another player.
        </p>
      )}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', marginTop: '0.5rem' }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-light)' }}>Annotated output (visual observations only)</span>
        <a
          className="btn btn-secondary"
          href={url}
          download={`spiral_annotated_assessment_${assessmentId}.${extension}`}
          style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center' }}
        >
          <Icon name="download" />
          Download annotated video
        </a>
      </div>
    </div>
  );
}
