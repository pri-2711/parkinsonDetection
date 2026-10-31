import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { researchApi } from '../api/research';
import type { ResearchInsights } from '../types/api';

const METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] as const;
const METRIC_LABELS: Record<(typeof METRICS)[number], string> = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1: 'F1',
  roc_auc: 'ROC-AUC',
};
const MODEL_LABELS: Record<string, string> = {
  baseline_svm: 'Baseline SVM (archived)',
  tuned_svm: 'Tuned SVM (calibrated)',
  baseline_random_forest: 'Baseline RF (archived)',
  tuned_random_forest: 'Tuned RF (calibrated)',
  nested_selected: 'Nested-selected SVM/RF',
};

const STATUS_STYLES = {
  implemented: { label: 'Implemented prediction', background: '#EBF3EA', color: '#3E5438' },
  observation: { label: 'Experimental observation', background: '#EEF3F8', color: '#2C4A6B' },
  unavailable: { label: 'Unavailable', background: '#FEF3C7', color: '#7A4A0B' },
  baseline: { label: 'Baseline only (not used live)', background: '#F1F1EE', color: '#4A4A46' },
  experimental: { label: 'Experimental prediction', background: '#FDF4E3', color: '#7A4A0B' },
} as const;

type StatusKind = keyof typeof STATUS_STYLES;

const MODALITY_STATUS: { name: string; kind: StatusKind; detail: string }[] = [
  { name: 'Tremor (7 UPDRS III items)', kind: 'implemented', detail: 'Calibrated SVM on exact integer ratings 0–4. Saved probability per assessment.' },
  { name: 'Handwriting / spiral (tablet telemetry)', kind: 'implemented', detail: 'Calibrated SVM only for complete measured tablet data (x, y, z, pressure, grip angle, timestamp).' },
  { name: 'Spiral canvas telemetry', kind: 'observation', detail: 'Browser pointer coordinates and timing. Not classifier-compatible; shown for inspection.' },
  { name: 'Spiral video analysis', kind: 'observation', detail: 'Newly drawn spiral ink (blue), found by aligning each frame to the first frame so printed text and ruling are not counted. Hand outline (red) and MediaPipe landmarks (green), with grip estimates only where the fingers are visible. Pen tip and pressure are not measured; not used by any classifier.' },
  { name: 'Speech, reading passage (speech_audio_v2)', kind: 'experimental', detail: 'Reading-mode model trained on 37 participants. Experimental: its cross-validated accuracy is near the majority-class baseline. Sustained vowels have no model.' },
  { name: 'Speech legacy SVM (45 precomputed features)', kind: 'baseline', detail: 'Archived validation of precomputed UCI descriptors. Never applied to speech_audio_v2 features.' },
];

function pct(value: number | null | undefined, digits = 3) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function meanSd(summary: { mean: number | null; sd: number | null } | undefined) {
  if (!summary || summary.mean === null) return '—';
  return summary.sd === null ? pct(summary.mean) : `${pct(summary.mean)} ± ${pct(summary.sd)}`;
}

function StatusChip({ kind }: { kind: StatusKind }) {
  const style = STATUS_STYLES[kind];
  return (
    <span style={{ display: 'inline-block', padding: '0.2rem 0.6rem', borderRadius: 999, fontSize: '0.78rem', fontWeight: 600, background: style.background, color: style.color }}>
      {style.label}
    </span>
  );
}

const cellStyle = { padding: '0.45rem 0.6rem', borderBottom: '1px solid var(--border-subtle)', textAlign: 'right' as const, whiteSpace: 'nowrap' as const };
const headStyle = { ...cellStyle, textAlign: 'right' as const, fontWeight: 600, color: 'var(--text-dark)', background: '#F4F7F2' };

export function Research() {
  const navigate = useNavigate();
  const [insights, setInsights] = useState<ResearchInsights | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    researchApi.getInsights()
      .then(setInsights)
      .catch((err) => setError(err instanceof Error ? err.message : 'Research outputs could not be loaded.'));
  }, []);

  if (error) return (
    <div className="container">
      <button type="button" className="btn btn-secondary" onClick={() => navigate('/dashboard')} style={{ marginBottom: '1.25rem' }}>
        ← Back to Dashboard
      </button>
      <p role="alert" className="error">{error}</p>
    </div>
  );
  if (!insights) return <div className="loading"><p>Loading research outputs…</p></div>;

  const { fusion, speech_audio_v2: speechV2 } = insights;

  return (
    <div className="container" style={{ maxWidth: '1000px' }}>
      <button type="button" className="btn btn-secondary" onClick={() => navigate('/dashboard')} style={{ marginBottom: '1.25rem' }}>
        ← Back to Dashboard
      </button>
      <h1 style={{ color: 'var(--sage-dark)', fontSize: '1.85rem', marginBottom: '0.4rem' }}>Research &amp; Model Insights</h1>
      <p style={{ color: 'var(--text-light)', marginTop: 0 }}>{insights.disclaimer}</p>

      <section className="card" style={{ marginBottom: '1.5rem' }} aria-labelledby="status-heading">
        <h2 id="status-heading" style={{ fontSize: '1.2rem', color: 'var(--sage-dark)', marginTop: 0 }}>What is implemented, observed or unavailable</h2>
        <p style={{ color: 'var(--text-light)', fontSize: '0.9rem' }}>
          Only the first category produces a probability. Observations are saved for review and are never fused. Unavailable items are not predicted.
        </p>
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          {MODALITY_STATUS.map((item) => (
            <li key={item.name} style={{ display: 'grid', gridTemplateColumns: 'minmax(200px, 1fr) 210px', gap: '0.5rem 1rem', padding: '0.6rem 0', borderBottom: '1px solid var(--border-subtle)' }}>
              <div>
                <strong style={{ color: 'var(--text-dark)' }}>{item.name}</strong>
                <div style={{ color: 'var(--text-light)', fontSize: '0.88rem' }}>{item.detail}</div>
              </div>
              <div><StatusChip kind={item.kind} /></div>
            </li>
          ))}
        </ul>
      </section>

      {insights.modalities.map((modality) => (
        <section key={modality.modality} className="card" style={{ marginBottom: '1.5rem' }} aria-labelledby={`${modality.modality}-heading`}>
          <h2 id={`${modality.modality}-heading`} style={{ fontSize: '1.2rem', color: 'var(--sage-dark)', marginTop: 0 }}>{modality.label}</h2>
          <p style={{ color: 'var(--text-light)', fontSize: '0.9rem', margin: '0 0 0.5rem' }}>
            <strong>Dataset:</strong> {modality.dataset} {modality.rows} rows from {modality.participant_groups} participant groups; {modality.feature_count} features.
            {modality.reported_folds.length > 0 && <> Reported folds {modality.reported_folds.join(', ')} ({modality.reported_rows} rows).</>}
          </p>
          <p style={{ color: 'var(--text-light)', fontSize: '0.9rem', margin: '0 0 0.9rem' }}>
            <strong>Live use:</strong> {modality.live_use}
          </p>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
              <caption style={{ textAlign: 'left', fontWeight: 600, color: 'var(--text-dark)', padding: '0 0 0.4rem' }}>Pooled cross-validated metrics (folds {modality.reported_folds.join('–')})</caption>
              <thead>
                <tr>
                  <th scope="col" style={{ ...headStyle, textAlign: 'left' }}>Model</th>
                  {METRICS.map((metric) => <th key={metric} scope="col" style={headStyle}>{METRIC_LABELS[metric]}</th>)}
                </tr>
              </thead>
              <tbody>
                {Object.keys(MODEL_LABELS).filter((key) => modality.pooled_metrics[key]).map((key) => (
                  <tr key={key}>
                    <th scope="row" style={{ ...cellStyle, textAlign: 'left', fontWeight: key.startsWith('tuned_svm') || key === 'nested_selected' ? 700 : 400 }}>{MODEL_LABELS[key]}</th>
                    {METRICS.map((metric) => <td key={metric} style={cellStyle}>{pct(modality.pooled_metrics[key][metric])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
              <caption style={{ textAlign: 'left', fontWeight: 600, color: 'var(--text-dark)', padding: '0 0 0.4rem' }}>Per-fold mean ± sample SD</caption>
              <thead>
                <tr>
                  <th scope="col" style={{ ...headStyle, textAlign: 'left' }}>Model</th>
                  {METRICS.map((metric) => <th key={metric} scope="col" style={headStyle}>{METRIC_LABELS[metric]}</th>)}
                </tr>
              </thead>
              <tbody>
                {Object.keys(MODEL_LABELS).filter((key) => modality.fold_metrics[key]).map((key) => (
                  <tr key={key}>
                    <th scope="row" style={{ ...cellStyle, textAlign: 'left', fontWeight: 400 }}>{MODEL_LABELS[key]}</th>
                    {METRICS.map((metric) => <td key={metric} style={cellStyle}>{meanSd(modality.fold_metrics[key][metric])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <dl style={{ display: 'grid', gridTemplateColumns: 'max-content 1fr', gap: '0.35rem 1rem', fontSize: '0.9rem', margin: '1rem 0 0' }}>
            <dt style={{ fontWeight: 600 }}>Deployed family</dt>
            <dd style={{ margin: 0 }}>{modality.deployed_family?.toUpperCase() ?? '—'} (sigmoid-calibrated)</dd>
            <dt style={{ fontWeight: 600 }}>Deployed parameters</dt>
            <dd style={{ margin: 0 }}>{modality.deployed_params ? Object.entries(modality.deployed_params).map(([k, v]) => `${k.replace('model__', '')}=${String(v)}`).join(', ') : '—'}</dd>
            <dt style={{ fontWeight: 600 }}>Inner-CV ROC-AUC (SVM vs RF)</dt>
            <dd style={{ margin: 0 }}>SVM {pct(modality.inner_cv_roc_auc.svm)} · RF {pct(modality.inner_cv_roc_auc.random_forest)}</dd>
          </dl>
        </section>
      ))}

      <section className="card" style={{ marginBottom: '1.5rem' }} aria-labelledby="fusion-heading">
        <h2 id="fusion-heading" style={{ fontSize: '1.2rem', color: 'var(--sage-dark)', marginTop: 0 }}>Late-fusion weights</h2>
        <p style={{ color: 'var(--text-light)', fontSize: '0.9rem' }}>
          {fusion.method}. Weights come from fold {fusion.validation_fold} only; reported test folds do not influence them.
          Patient-level multimodal performance: <strong>not measured</strong>, because the speech, handwriting and tremor cohorts are unmatched.
        </p>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
            <thead>
              <tr>
                <th scope="col" style={{ ...headStyle, textAlign: 'left' }}>Modality</th>
                <th scope="col" style={headStyle}>Validation ROC-AUC</th>
                <th scope="col" style={headStyle}>Normalized weight</th>
                <th scope="col" style={headStyle}>Archived V1 weight (audit only)</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(fusion.weights).map((name) => (
                <tr key={name}>
                  <th scope="row" style={{ ...cellStyle, textAlign: 'left', fontWeight: 400 }}>{name}</th>
                  <td style={cellStyle}>{pct(fusion.validation_auc[name], 4)}</td>
                  <td style={cellStyle}>{pct(fusion.weights[name], 4)}</td>
                  <td style={cellStyle}>{pct(fusion.baseline_v1_weights_for_audit[name], 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p style={{ color: 'var(--text-light)', fontSize: '0.85rem', marginBottom: 0 }}>{fusion.baseline_v1_weights_note}</p>
      </section>

      <section className="card" style={{ marginBottom: '1.5rem' }} aria-labelledby="speech-v2-heading">
        <h2 id="speech-v2-heading" style={{ fontSize: '1.2rem', color: 'var(--sage-dark)', marginTop: 0 }}>Speech (speech_audio_v2) status</h2>
        <p><StatusChip kind={speechV2.status === 'experimental_prediction' ? 'experimental' : 'unavailable'} /></p>
        {speechV2.blocker && <p style={{ fontSize: '0.92rem' }}><strong>Blocker:</strong> {speechV2.blocker}</p>}
        {Object.entries(speechV2.modes).map(([mode, info]) => (
          <div key={mode} style={{ fontSize: '0.92rem', marginBottom: '0.6rem' }}>
            <strong>{mode === 'reading' ? 'Reading passage' : 'Sustained vowel'}:</strong>{' '}
            {info.model_trained && info.outer_fold_roc_auc_mean !== undefined ? (
              <>
                {info.subjects} participants ({info.control_subjects} control, {info.parkinson_subjects} PD), {info.recordings} segments.
                {' '}Subject-disjoint 5-fold ROC-AUC {pct(info.outer_fold_roc_auc_mean)} ± {pct(info.outer_fold_roc_auc_sd)};
                {' '}majority-class accuracy {pct(info.majority_class_accuracy)}.
              </>
            ) : 'no trained model.'}
          </div>
        ))}
        <p style={{ fontSize: '0.88rem', color: 'var(--text-light)' }}><strong>Validation:</strong> {speechV2.validation_protocol}</p>
        <ul style={{ fontSize: '0.88rem', color: 'var(--text-dark)' }}>
          {speechV2.limitations.map((item) => <li key={item} style={{ marginBottom: '0.3rem' }}>{item}</li>)}
        </ul>
        <details>
          <summary>{speechV2.feature_count} model input features</summary>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-light)', wordBreak: 'break-word' }}>{speechV2.features.join(', ')}</p>
        </details>
      </section>

      <section className="card" style={{ marginBottom: '1.5rem' }} aria-labelledby="architecture-heading">
        <h2 id="architecture-heading" style={{ fontSize: '1.2rem', color: 'var(--sage-dark)', marginTop: 0 }}>Architecture</h2>
        <ul style={{ fontSize: '0.92rem', color: 'var(--text-dark)' }}>
          {insights.architecture.map((item) => <li key={item} style={{ marginBottom: '0.35rem' }}>{item}</li>)}
        </ul>
      </section>

      <section className="card" style={{ marginBottom: '1.5rem' }} aria-labelledby="validation-heading">
        <h2 id="validation-heading" style={{ fontSize: '1.2rem', color: 'var(--sage-dark)', marginTop: 0 }}>Validation protocol</h2>
        <dl style={{ display: 'grid', gridTemplateColumns: 'max-content 1fr', gap: '0.4rem 1rem', fontSize: '0.9rem', margin: 0 }}>
          {Object.entries(insights.validation).map(([key, value]) => (
            <div key={key} style={{ display: 'contents' }}>
              <dt style={{ fontWeight: 600 }}>{key.replace(/_/g, ' ')}</dt>
              <dd style={{ margin: 0 }}>{String(value)}</dd>
            </div>
          ))}
        </dl>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-light)', marginBottom: 0 }}>{insights.datasets_note}</p>
      </section>
    </div>
  );
}
