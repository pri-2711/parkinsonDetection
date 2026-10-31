import { useState, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';
import { SpeechFeatureGroups } from '../components/SpeechFeatureGroups';
import { Icon } from '../components/Icon';
import { formatPercent, speechStatusLabel, speechStepSummary } from '../features/fusionDisplay.js';

const READING_PASSAGE = 'Today is a pleasant day. I am speaking at my normal pace and volume. I will read each sentence clearly and naturally, without rushing. After finishing the passage, I will pause briefly before ending the recording.';
const PASSAGE_VERSION = '1.0';

async function toPcmWav(blob: Blob): Promise<Blob> {
  const AudioContextClass = window.AudioContext;
  if (!AudioContextClass) throw new Error('This browser cannot decode the recorded audio for WAV upload.');
  const context = new AudioContextClass();
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer());
    const samples = new Float32Array(decoded.length);
    for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
      const input = decoded.getChannelData(channel);
      for (let i = 0; i < input.length; i++) samples[i] += input[i] / decoded.numberOfChannels;
    }
    const pcm = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(pcm);
    const write = (offset: number, value: string) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
    write(0, 'RIFF'); view.setUint32(4, pcm.byteLength - 8, true); write(8, 'WAVE');
    write(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, decoded.sampleRate, true);
    view.setUint32(28, decoded.sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    write(36, 'data'); view.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      const value = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, value < 0 ? value * 32768 : value * 32767, true);
    }
    return new Blob([pcm], { type: 'audio/wav' });
  } finally {
    await context.close();
  }
}

export function Speech() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [activeTab, setActiveTab] = useState<'record' | 'upload'>('record');
  const [recordingMode, setRecordingMode] = useState<'reading' | 'sustained_vowel'>('reading');
  const [genderCode, setGenderCode] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [savedSpeechStep, setSavedSpeechStep] = useState<{ status?: string; outcome?: any; error?: string | null } | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  useEffect(() => {
    if (!id) return;
    assessmentsApi.get(Number(id)).then((assessment) => {
      setSavedSpeechStep(assessment.steps.speech);
    }).catch(() => undefined);
  }, [id]);

  const startRecording = async () => {
    setError('');
    setRecordedBlob(null);
    setAudioUrl(null);
    setRecordingDuration(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType || 'application/octet-stream' });
        setRecordedBlob(blob);
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setRecording(true);

      timerRef.current = window.setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setError('Microphone access denied or unavailable. Please enable permissions or use file upload.');
    }

  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      if (!selected.name.toLowerCase().endsWith('.wav')) {
        setError('Only WAV audio format is supported by the validation pipeline.');
        setFile(null);
        return;
      }
      setFile(selected);
      setError('');
      setAudioUrl(URL.createObjectURL(selected));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!id) return;

    setSubmitting(true);
    setError('');

    try {
      const source = activeTab === 'record' ? recordedBlob : file;
      if (!source) throw new Error('Record or select a WAV sample first.');
      if (genderCode === '') throw new Error('Select the dataset gender category required by the training schema.');
      const wav = source.type === 'audio/wav' || (activeTab === 'upload' && file?.name.toLowerCase().endsWith('.wav'))
        ? source : await toPcmWav(source);
      const response = await assessmentsApi.predictSpeech(parseInt(id, 10), wav, 'speech-sample.wav', {
        mode: recordingMode,
        gender_code: genderCode,
        ...(recordingMode === 'reading' ? { passage_version: PASSAGE_VERSION, passage_text: READING_PASSAGE } : {}),
      });
      // The result is shown on this page straight away. Final Results is opened only on request.
      setSavedSpeechStep({ status: response.status, outcome: response.outcome, error: null });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Speech processing error');
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
          <h2 style={{ color: 'var(--sage-dark)', fontSize: '1.5rem', marginBottom: '0.75rem' }}>
            Speech Assessment
          </h2>
          <p style={{ color: 'var(--text-light)', fontSize: '0.92rem', lineHeight: '1.5', margin: 0 }}>
            Record or upload a WAV recording. The audio is analysed and then discarded. The result is an experimental research score, not a diagnosis.
          </p>
          <details style={{ marginTop: '0.75rem' }}>
            <summary>About the experimental model</summary>
            <ul style={{ fontSize: '0.88rem', color: 'var(--text-light)' }}>
              <li>This is a research prototype. Results are experimental and are not a diagnosis.</li>
              <li>Sustained-vowel recordings have no trained model, so they show measurements only.</li>
              <li>Audio is not stored. Only the extracted measurements and the result are saved to the assessment.</li>
            </ul>
          </details>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <label htmlFor="speech-mode" style={{ display: 'block', fontWeight: 600, marginBottom: '0.4rem' }}>Recording task</label>
          <select id="speech-mode" value={recordingMode} onChange={(event) => { setRecordingMode(event.target.value as 'reading' | 'sustained_vowel'); setRecordedBlob(null); setFile(null); setAudioUrl(null); setError(''); }}>
            <option value="reading">Read the passage</option>
            <option value="sustained_vowel">Sustained "ah" (features only)</option>
          </select>
        </div>
        <div style={{ marginBottom: '1rem' }}>
          <label htmlFor="speech-gender" style={{ display: 'block', fontWeight: 600, marginBottom: '0.4rem' }}>Sex (required)</label>
          <select id="speech-gender" value={genderCode} onChange={(event) => setGenderCode(event.target.value)}>
            <option value="">Select…</option>
            <option value="0">Male</option>
            <option value="1">Female</option>
          </select>
        </div>
        {recordingMode === 'reading' && <div style={{ backgroundColor: '#F4F7F2', border: '1px solid var(--sage-light)', padding: '1.25rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: '700', color: 'var(--sage-dark)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Standardized Reading Passage (v{PASSAGE_VERSION})
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-light)' }}>Approx. 15–20 seconds</span>
          </div>
          <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--text-dark)', fontStyle: 'italic', margin: 0 }}>
            "{READING_PASSAGE}"
          </p>
        </div>}
        {recordingMode === 'sustained_vowel' && <div style={{ backgroundColor: '#F4F7F2', border: '1px solid var(--sage-light)', padding: '1rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
          Hold a steady "ah" for at least five seconds. This task produces features only; no prediction is available for it.
        </div>}

        {/* Input Method Selector Tabs */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-subtle)' }}>
          <button
            type="button"
            className={`btn ${activeTab === 'record' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ borderRadius: '6px 6px 0 0', borderBottom: 'none' }}
            onClick={() => { setActiveTab('record'); setError(''); }}
          >
            🎙️ Record in Browser
          </button>
          <button
            type="button"
            className={`btn ${activeTab === 'upload' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ borderRadius: '6px 6px 0 0', borderBottom: 'none' }}
            onClick={() => { setActiveTab('upload'); setError(''); }}
          >
            <Icon name="upload" />Upload WAV File
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          {activeTab === 'record' ? (
            <div style={{ textAlign: 'center', padding: '1.5rem', backgroundColor: '#FAFAF7', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: '1.5rem' }}>
              <div style={{ marginBottom: '1rem' }}>
                <span style={{ fontSize: '2.5rem' }}>{recording ? <span aria-label="Recording" style={{ display: 'inline-block', width: 18, height: 18, borderRadius: '50%', background: '#B93838' }} /> : '🎙️'}</span>
                <p style={{ fontWeight: '600', color: 'var(--text-dark)', marginTop: '0.5rem' }}>
                  {recording ? `Recording in progress (${recordingDuration}s)… ${recordingMode === 'reading' ? 'Speak the passage clearly.' : 'Sustain a steady "ah" vowel.'}` : recordedBlob ? `Recording captured (${recordingDuration}s)` : 'Click to begin microphone recording'}
                </p>
              </div>

              <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginBottom: '1rem' }}>
                {!recording ? (
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={startRecording}
                  >
                    Start Recording
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn btn-primary"
                    style={{ backgroundColor: 'var(--error)' }}
                    onClick={stopRecording}
                  >
                    Stop Recording
                  </button>
                )}
              </div>

              {audioUrl && (
                <div style={{ marginTop: '1rem' }}>
                  <audio controls src={audioUrl} style={{ width: '100%', maxWidth: '400px' }} />
                </div>
              )}
            </div>
          ) : (
            <div style={{ padding: '1.5rem', backgroundColor: '#FAFAF7', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: '1.5rem' }}>
              <label htmlFor="speech-audio-file" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '600', color: 'var(--text-dark)' }}>
                Select Audio Recording (WAV format, {recordingMode === 'sustained_vowel' ? '5' : '1'}–30 seconds)
              </label>
              <input
                id="speech-audio-file"
                type="file"
                accept=".wav,audio/wav"
                onChange={handleFileChange}
                style={{ width: '100%', padding: '0.75rem', border: '1px solid var(--border-subtle)', borderRadius: '6px', backgroundColor: '#FFFFFF' }}
              />
              {audioUrl && (
                <div style={{ marginTop: '1rem' }}>
                  <audio controls src={audioUrl} style={{ width: '100%', maxWidth: '400px' }} />
                </div>
              )}
            </div>
          )}

          {/* Model Feature Contract Notice */}
          <div style={{ backgroundColor: '#F4F7F2', border: '1px solid var(--border-subtle)', padding: '0.85rem 1rem', borderRadius: '8px', fontSize: '0.88rem', color: 'var(--text-light)' }}>
            Audio is analysed and not kept. Only the extracted features and the score are saved to this assessment.
          </div>

          {error && (
            <div style={{ backgroundColor: 'var(--error-bg)', border: '1px solid var(--error)', padding: '0.75rem 1rem', borderRadius: '6px', marginBottom: '1.25rem' }}>
              <p className="error" style={{ margin: 0 }}>{error}</p>
            </div>
          )}

          <button
            type="submit"
            className="btn btn-primary"
            style={{ width: '100%', padding: '0.85rem' }}
            disabled={submitting || (activeTab === 'record' && !recordedBlob) || (activeTab === 'upload' && !file)}
          >
            {submitting ? 'Validating Audio & Processing…' : 'Submit Speech Sample'}
          </button>
        </form>
        {savedSpeechStep && (() => {
          const view = speechStepSummary(savedSpeechStep);
          const values = (savedSpeechStep.outcome?.feature_values ?? null) as Record<string, unknown> | null;
          const probability = savedSpeechStep.outcome?.probability;
          return (
            <section aria-live="polite" style={{ marginTop: '1.25rem', padding: '1rem', background: '#FAFAF7', border: '1px solid var(--border-subtle)', borderRadius: 6 }}>
              <h3 style={{ marginTop: 0, color: 'var(--sage-dark)' }}>
                {view.title}: {speechStatusLabel(savedSpeechStep.status)}
              </h3>

              {view.predictionState === 'completed' && typeof probability === 'number' && (
                <div style={{ padding: '0.8rem 0.95rem', background: 'var(--success-bg)', border: '1px solid var(--border-subtle)', borderRadius: 6, marginBottom: '0.6rem' }}>
                  <div style={{ fontSize: '0.82rem', color: 'var(--text-light)' }}>Prediction probability</div>
                  <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--success)' }}>{formatPercent(probability)}</div>
                  <p style={{ margin: '0.25rem 0 0', fontSize: '0.88rem', color: 'var(--text-dark)' }}>{view.predictionText}</p>
                  <p style={{ margin: '0.35rem 0 0', fontSize: '0.8rem', color: 'var(--text-light)' }}>Research output only. Not a diagnosis.</p>
                </div>
              )}
              {view.predictionState === 'unavailable' && (
                <div role="status" style={{ padding: '0.8rem 0.95rem', background: 'var(--warning-bg)', border: '1px solid var(--border-subtle)', borderRadius: 6, marginBottom: '0.6rem' }}>
                  <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--warning)' }}>Prediction unavailable</div>
                  <p style={{ margin: '0.25rem 0 0', fontSize: '0.88rem', color: 'var(--text-dark)' }}>
                    The model for this recording mode is not trained, so no prediction was made. The measurements below are extracted features only.
                  </p>
                </div>
              )}
              {view.predictionState === 'not_run' && view.text && (
                <p style={{ margin: '0 0 0.6rem' }}>{view.text}</p>
              )}
              {view.extractionText && <p style={{ margin: '0 0 0.6rem', fontSize: '0.92rem' }}>{view.extractionText}</p>}

              {values && (
                <>
                  <h3 style={{ margin: '1.1rem 0 0.25rem', color: 'var(--sage-dark)', fontSize: '1.05rem' }}>Extracted acoustic measurements</h3>
                  <p style={{ margin: '0 0 0.8rem', fontSize: '0.85rem', color: 'var(--text-light)' }}>
                    Values exactly as extracted from this recording. They are not clinical thresholds.
                  </p>
                  <SpeechFeatureGroups values={values} />
                </>
              )}
            </section>
          );
        })()}
      </div>
    </div>
  );
}
