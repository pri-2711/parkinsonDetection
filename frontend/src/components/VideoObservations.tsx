import { AnnotatedVideo } from './AnnotatedVideo';

interface Props {
  analysis: any;
  assessmentId: number;
}

const UNAVAILABLE_LABELS: Record<string, string> = {
  finger_landmarks: 'Individual finger positions',
  grip_posture: 'Pen-holding grip posture',
  pen_tip_tracking: 'Pen-tip tracking',
  pressure: 'Pen pressure',
  grip_force: 'Grip force',
  grip_angle: 'Grip angle',
  fitted_spiral_deviation: 'Deviation from a fitted ideal spiral',
  hand_region: 'Hand region',
};

function pct(value: unknown): string {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'not measured';
}

function num(value: unknown, suffix = ''): string {
  return typeof value === 'number' ? `${value}${suffix}` : 'not measured';
}

function MeasureRows({ rows }: { rows: [string, string][] }) {
  return (
    <dl style={{ margin: 0, display: 'grid', gridTemplateColumns: 'minmax(170px, 1fr) auto', gap: '0.3rem 1rem', fontSize: '0.9rem' }}>
      {rows.map(([label, value]) => (
        <div key={label} style={{ display: 'contents' }}>
          <dt style={{ color: 'var(--text-light)' }}>{label}</dt>
          <dd style={{ margin: 0, fontWeight: 600, color: 'var(--text-dark)', textAlign: 'right' }}>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function Swatch({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: 'var(--text-dark)' }}>
      <span style={{ width: 14, height: 14, borderRadius: 3, background: color, display: 'inline-block' }} />
      {label}
    </span>
  );
}

// Two separate analyses: blue ink and strokes, red hand region. Measured, estimated and unavailable values are kept apart.
export function VideoObservations({ analysis, assessmentId }: Props) {
  const measurements = analysis.measurements;
  const measured = measurements?.directly_measured_image_features ?? {};
  const proxies = measurements?.estimated_visual_proxies ?? {};
  const unavailable: Record<string, string> = measurements?.unavailable_measurements ?? {};
  const legacy = !measurements;
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
        <Swatch color="#1E78FF" label="Pen ink / strokes" />
        <Swatch color="#E62828" label="Hand region" />
      </div>

      {legacy && (
        <p role="status" style={{ fontSize: '0.9rem' }}>
          This analysis was made by an earlier version and has no measurements. Analyse the video again to see them.
        </p>
      )}

      {!legacy && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
          <div>
            <h4 style={{ margin: '0 0 0.4rem', fontSize: '0.9rem', color: '#1E5FCC' }}>Ink and strokes (blue)</h4>
            {proxies.ink_status ? (
              <p style={{ margin: 0, fontSize: '0.88rem', color: 'var(--text-light)' }}>
                Too little blue ink was detected to measure the strokes. Check that the pen ink is blue or violet and the page is well lit.
              </p>
            ) : (
              <MeasureRows rows={[
                ['Ink coverage of frame', pct(measured.ink_coverage_of_frame)],
                ['Median stroke width', num(measured.stroke_width_px_median, ' px')],
                ['Stroke width variation', num(proxies.stroke_width_variation)],
                ['Continuity (largest piece share)', pct(proxies.continuity_largest_piece_share)],
                ['Separate ink pieces', num(measured.ink_components)],
              ]} />
            )}
          </div>
          <div>
            <h4 style={{ margin: '0 0 0.4rem', fontSize: '0.9rem', color: '#B81E1E' }}>Hand (red)</h4>
            {proxies.hand_status ? (
              <p style={{ margin: 0, fontSize: '0.88rem', color: 'var(--text-light)' }}>
                The hand was not detected reliably enough to report its position.
              </p>
            ) : (
              <MeasureRows rows={[
                ['Frames with hand region', pct(measured.frames_with_hand_region)],
                ['Hand region size (median)', pct(proxies.hand_region_area_median)],
                ['Hand stability (median step)', num(proxies.hand_stability_median_step)],
                ['Visible fingertips (estimate)', num(proxies.fingertips_visible_estimate_median)],
              ]} />
            )}
          </div>
        </div>
      )}

      {unsupportedFile ? (
        <p role="alert" className="error" style={{ margin: '0.8rem 0 0' }}>
          The annotated file from this analysis uses a format this browser cannot play. Analyse the video again to create a playable copy. The original upload is unchanged.
        </p>
      ) : (
        <AnnotatedVideo assessmentId={assessmentId} analysisStatus={analysis.status} videoId={analysis.video_id} />
      )}

      {Object.keys(unavailable).length > 0 && (
        <details style={{ marginTop: '0.9rem' }}>
          <summary>Not measured, and why</summary>
          <ul style={{ fontSize: '0.88rem', color: 'var(--text-light)' }}>
            {Object.entries(unavailable).map(([key, reason]) => (
              <li key={key}><strong>{UNAVAILABLE_LABELS[key] ?? key}:</strong> {reason}</li>
            ))}
          </ul>
        </details>
      )}

      <details style={{ marginTop: '0.6rem' }}>
        <summary>Limitations</summary>
        <ul style={{ fontSize: '0.88rem', color: 'var(--text-light)' }}>
          {(analysis.limitations ?? []).map((item: string) => <li key={item}>{item}</li>)}
        </ul>
      </details>
    </div>
  );
}
