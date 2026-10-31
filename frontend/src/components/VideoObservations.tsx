import { AnnotatedVideo } from './AnnotatedVideo';

interface Props {
  analysis: any;
  assessmentId: number;
}

interface Item {
  label: string;
  value: string | number;
  unit: string;
  explanation: string;
  kind: 'measured' | 'estimated';
  reliability: 'high' | 'moderate' | 'low';
}

interface Unavailable {
  label: string;
  reason: string;
}

const SECTION_ORDER: [string, string][] = [
  ['ink_and_strokes', 'Ink & Stroke Analysis'],
  ['hand_and_grip', 'Hand & Grip Analysis'],
  ['spiral_geometry', 'Spiral Geometry Analysis'],
];

const RELIABILITY_LABEL: Record<string, string> = { high: 'High', moderate: 'Moderate', low: 'Low' };

function formatValue(value: string | number): string {
  if (typeof value === 'number') return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return value;
}

function Swatch({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: 'var(--text-dark)' }}>
      <span style={{ width: 14, height: 14, borderRadius: 3, background: color, display: 'inline-block' }} />
      {label}
    </span>
  );
}

function MeasureCard({ item }: { item: Item }) {
  const isText = typeof item.value === 'string';
  return (
    <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 8, background: '#fff', padding: '0.7rem 0.8rem', display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem', alignItems: 'baseline', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.82rem', color: 'var(--text-light)' }}>{item.label}</span>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-light)' }}>
          {item.kind === 'measured' ? 'Measured' : 'Estimated'} · {RELIABILITY_LABEL[item.reliability] ?? item.reliability}
        </span>
      </div>
      <div style={{ fontSize: isText ? '0.95rem' : '1.25rem', fontWeight: 600, color: 'var(--text-dark)', overflowWrap: 'anywhere' }}>
        {formatValue(item.value)}
        {item.unit && <span style={{ fontSize: '0.8rem', fontWeight: 400, color: 'var(--text-light)', marginLeft: '0.35rem' }}>{item.unit}</span>}
      </div>
      <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-light)' }}>{item.explanation}</p>
    </div>
  );
}

// Ink, hand and spiral values come from separate analyses. Each shows what was measured, what is estimated, and what could not be measured.
export function VideoObservations({ analysis, assessmentId }: Props) {
  const sections = analysis.sections;
  const legacy = !sections;
  // Analyses saved before the WebM change store an MP4 (MPEG-4 Part 2) file that Chromium cannot decode.
  const unsupportedFile = analysis.annotated_video_format !== undefined && analysis.annotated_video_format !== 'webm';

  return (
    <div aria-live="polite" style={{ marginTop: '1rem', padding: '1rem 1.1rem', border: '1px solid var(--border-subtle)', borderRadius: 10, background: '#FAFAF7' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: '0.5rem' }}>
        <strong style={{ color: 'var(--text-dark)' }}>Spiral video observations</strong>
        <span style={{ fontSize: '0.8rem', color: 'var(--warning)' }}>Experimental</span>
      </div>
      <p style={{ margin: '0.3rem 0 0.7rem', fontSize: '0.85rem', color: 'var(--text-light)' }}>
        {analysis.video.duration_seconds} s · {analysis.video.fps} fps · {analysis.video.width} × {analysis.video.height} px
      </p>

      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '0.8rem' }}>
        <Swatch color="#1E78FF" label="Newly drawn spiral ink" />
        <Swatch color="#3CC83C" label="Hand landmarks" />
        <Swatch color="#E62828" label="Hand outline" />
      </div>

      {legacy && (
        <p role="status" style={{ fontSize: '0.9rem' }}>
          This analysis was made by an earlier version and has no measurements. Analyse the video again to see them.
        </p>
      )}

      {!legacy && SECTION_ORDER.map(([key, fallbackTitle]) => {
        const section = sections[key] ?? { title: fallbackTitle, items: [], unavailable: [] };
        const items: Item[] = section.items ?? [];
        const unavailable: Unavailable[] = section.unavailable ?? [];
        return (
          <section key={key} style={{ marginTop: '1rem' }}>
            <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', color: 'var(--sage-dark)' }}>{section.title ?? fallbackTitle}</h4>
            {items.length > 0 ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.6rem' }}>
                {items.map((item) => <MeasureCard key={item.label} item={item} />)}
              </div>
            ) : (
              <p style={{ margin: 0, fontSize: '0.88rem', color: 'var(--text-light)' }}>No values could be measured in this section.</p>
            )}
            {unavailable.length > 0 && (
              <details style={{ marginTop: '0.5rem' }}>
                <summary style={{ fontSize: '0.85rem' }}>Not measured ({unavailable.length})</summary>
                <ul style={{ fontSize: '0.85rem', color: 'var(--text-light)', margin: '0.4rem 0 0' }}>
                  {unavailable.map((entry) => (
                    <li key={entry.label}><strong>{entry.label}:</strong> {entry.reason}</li>
                  ))}
                </ul>
              </details>
            )}
          </section>
        );
      })}

      {unsupportedFile ? (
        <p role="alert" className="error" style={{ margin: '0.8rem 0 0' }}>
          The annotated file from this analysis uses a format this browser cannot play. Analyse the video again to create a playable copy. The original upload is unchanged.
        </p>
      ) : (
        <div style={{ marginTop: '1rem' }}>
          <AnnotatedVideo assessmentId={assessmentId} analysisStatus={analysis.status} videoId={analysis.video_id} />
        </div>
      )}

      <details style={{ marginTop: '0.9rem' }}>
        <summary>Limitations</summary>
        <ul style={{ fontSize: '0.88rem', color: 'var(--text-light)' }}>
          {(analysis.limitations ?? []).map((item: string) => <li key={item}>{item}</li>)}
        </ul>
      </details>
    </div>
  );
}
