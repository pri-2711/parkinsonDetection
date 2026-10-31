// Speech features exactly as the backend returned them (speech_audio_v2). Values are shown as measured:
// no thresholds, scores or interpretations are added here.

interface FeatureDef {
  key: string;
  label: string;
  unit?: string;
}

interface FeatureGroup {
  title: string;
  explanation: string;
  features: FeatureDef[];
}

const HNR_BANDS: FeatureDef[] = [
  { key: 'audio_v2_hnr_500_db', label: 'HNR, 60–500 Hz band', unit: 'dB' },
  { key: 'audio_v2_hnr_1500_db', label: 'HNR, 60–1500 Hz band', unit: 'dB' },
  { key: 'audio_v2_hnr_2500_db', label: 'HNR, 60–2500 Hz band', unit: 'dB' },
  { key: 'audio_v2_hnr_3500_db', label: 'HNR, 60–3500 Hz band', unit: 'dB' },
  { key: 'audio_v2_hnr_3800_db', label: 'HNR, 60–3800 Hz band', unit: 'dB' },
];

const MFCC_KEYS: FeatureDef[] = Array.from({ length: 13 }, (_, i) => ({ key: `audio_v2_mfcc${i}_mean`, label: `MFCC ${i}` }));
const DELTA_KEYS: FeatureDef[] = Array.from({ length: 13 }, (_, i) => ({ key: `audio_v2_delta${i}_mean`, label: `Delta ${i}` }));

export const SPEECH_FEATURE_GROUPS: FeatureGroup[] = [
  {
    title: 'Voice Stability',
    explanation: 'Cycle-to-cycle variation in the timing of each vocal-fold vibration (jitter).',
    features: [
      { key: 'audio_v2_jitter_abs_s', label: 'Absolute jitter', unit: 's' },
      { key: 'audio_v2_jitter_local', label: 'Local jitter' },
      { key: 'audio_v2_jitter_rap', label: 'RAP jitter' },
      { key: 'audio_v2_jitter_ppq5', label: 'PPQ5 jitter' },
    ],
  },
  {
    title: 'Voice Quality',
    explanation: 'Cycle-to-cycle variation in the loudness of the voice (shimmer).',
    features: [
      { key: 'audio_v2_shimmer_local', label: 'Local shimmer' },
      { key: 'audio_v2_shimmer_db', label: 'Shimmer in dB', unit: 'dB' },
      { key: 'audio_v2_shimmer_apq3', label: 'APQ3 shimmer' },
      { key: 'audio_v2_shimmer_apq5', label: 'APQ5 shimmer' },
      { key: 'audio_v2_shimmer_apq11', label: 'APQ11 shimmer' },
    ],
  },
  {
    title: 'Pitch & Rhythm',
    explanation: 'How irregular the pitch period and the voice signal are over time.',
    features: [
      { key: 'audio_v2_pitch_period_entropy', label: 'Pitch period entropy' },
      { key: 'audio_v2_rpde_return_entropy', label: 'RPDE (return entropy)' },
      { key: 'audio_v2_dfa_alpha', label: 'DFA alpha' },
    ],
  },
  {
    title: 'Harmonic Quality',
    explanation: 'Ratio of harmonic energy to noise energy (HNR) in each frequency band, and how well the voice envelope matches across bands (GNE).',
    features: [...HNR_BANDS, { key: 'audio_v2_gne_max_envelope_corr', label: 'GNE (max envelope correlation)' }],
  },
  {
    title: 'Acoustic Characteristics',
    explanation: 'Spectral shape of the voice (MFCC) and how that shape changes from frame to frame (delta).',
    features: [...MFCC_KEYS, ...DELTA_KEYS],
  },
];

const RECORDING_FEATURE: FeatureDef = { key: 'audio_v2_gender_code', label: 'Sex code entered for this recording (0 = male, 1 = female)' };

// Four significant figures keeps values readable. The full value is kept in the tooltip.
export function formatMeasure(value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'not returned';
  return Number(value.toPrecision(4)).toString();
}

function MeasureTile({ def, value }: { def: FeatureDef; value: unknown }) {
  const text = formatMeasure(value);
  return (
    <div
      title={typeof value === 'number' ? String(value) : undefined}
      style={{ border: '1px solid var(--border-subtle)', borderRadius: 6, background: '#fff', padding: '0.5rem 0.65rem', minWidth: 0 }}
    >
      <div style={{ fontSize: '0.78rem', color: 'var(--text-light)' }}>{def.label}</div>
      <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-dark)', overflowWrap: 'anywhere' }}>
        {text}
        {def.unit && text !== 'not returned' && <span style={{ fontSize: '0.78rem', fontWeight: 400, color: 'var(--text-light)', marginLeft: '0.3rem' }}>{def.unit}</span>}
      </div>
    </div>
  );
}

interface Props {
  values: Record<string, unknown>;
}

// Every extracted feature is shown on the page, grouped, with no collapsed sections.
export function SpeechFeatureGroups({ values }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
      {SPEECH_FEATURE_GROUPS.map((group) => (
        <section key={group.title} aria-labelledby={`speech-group-${group.title.replace(/\W+/g, '-').toLowerCase()}`}>
          <h4 id={`speech-group-${group.title.replace(/\W+/g, '-').toLowerCase()}`} style={{ margin: '0 0 0.2rem', fontSize: '0.98rem', color: 'var(--sage-dark)' }}>
            {group.title}
          </h4>
          <p style={{ margin: '0 0 0.5rem', fontSize: '0.85rem', color: 'var(--text-light)' }}>{group.explanation}</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '0.5rem' }}>
            {group.features.map((def) => <MeasureTile key={def.key} def={def} value={values[def.key]} />)}
          </div>
        </section>
      ))}
      <section aria-labelledby="speech-group-recording">
        <h4 id="speech-group-recording" style={{ margin: '0 0 0.2rem', fontSize: '0.98rem', color: 'var(--sage-dark)' }}>Recording information</h4>
        <p style={{ margin: '0 0 0.5rem', fontSize: '0.85rem', color: 'var(--text-light)' }}>
          Entered with the recording. No audio is kept.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '0.5rem' }}>
          <MeasureTile def={RECORDING_FEATURE} value={values[RECORDING_FEATURE.key]} />
        </div>
      </section>
    </div>
  );
}
