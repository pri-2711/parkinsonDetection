import { useState, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';

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
  const [savedSpeech, setSavedSpeech] = useState<any>(null);
  const [savedSpeechStatus, setSavedSpeechStatus] = useState<string>('pending');

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
      setSavedSpeech(assessment.steps.speech.outcome);
      setSavedSpeechStatus(assessment.steps.speech.status);
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
      setSavedSpeech(response.outcome);
      setSavedSpeechStatus(response.status);
      navigate(`/results/${id}`);
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
          <p style={{ color: 'var(--text-dark)', fontSize: '0.95rem', lineHeight: '1.6', backgroundColor: '#FAFAF7', padding: '1rem', borderRadius: '6px', borderLeft: '4px solid var(--sage-green)' }}>
            <strong>Speech Assessment:</strong> WAV recordings are processed with the separate speech_audio_v2 schema. No compatible mode-specific model is currently available, so a successful extraction is saved as unavailable for prediction. The legacy SVM is not used with these features. This research tool is not a medical diagnosis.
          </p>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <label htmlFor="speech-mode" style={{ display: 'block', fontWeight: 600, marginBottom: '0.4rem' }}>Recording task</label>
          <select id="speech-mode" value={recordingMode} onChange={(event) => { setRecordingMode(event.target.value as 'reading' | 'sustained_vowel'); setRecordedBlob(null); setFile(null); setAudioUrl(null); setError(''); }}>
            <option value="reading">Read the passage</option>
            <option value="sustained_vowel">Sustain “ah” for at least 5 seconds</option>
          </select>
        </div>
        <div style={{ marginBottom: '1rem' }}>
          <label htmlFor="speech-gender" style={{ display: 'block', fontWeight: 600, marginBottom: '0.4rem' }}>Dataset gender category (required)</label>
          <select id="speech-gender" value={genderCode} onChange={(event) => setGenderCode(event.target.value)}>
            <option value="">Choose…</option><option value="0">Man (code 0)</option><option value="1">Woman (code 1)</option>
          </select>
          <small>The source dataset contains only codes 0 and 1; this value is never inferred from audio.</small>
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
          Hold a steady “ah” vowel on one breath for at least five seconds. This is a separate task from reading; the experimental v2 extractor does not treat them as interchangeable.
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
            📁 Upload WAV File
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          {activeTab === 'record' ? (
            <div style={{ textAlign: 'center', padding: '1.5rem', backgroundColor: '#FAFAF7', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: '1.5rem' }}>
              <div style={{ marginBottom: '1rem' }}>
                <span style={{ fontSize: '2.5rem' }}>{recording ? '🔴' : '🎙️'}</span>
                <p style={{ fontWeight: '600', color: 'var(--text-dark)', marginTop: '0.5rem' }}>
                  {recording ? `Recording in progress (${recordingDuration}s)… ${recordingMode === 'reading' ? 'Speak the passage clearly.' : 'Sustain a steady “ah” vowel.'}` : recordedBlob ? `Recording captured (${recordingDuration}s)` : 'Click to begin microphone recording'}
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
          <div style={{ backgroundColor: 'var(--cream-dark)', padding: '0.85rem 1rem', borderRadius: '6px', marginBottom: '1.5rem' }}>
            <p style={{ color: 'var(--text-dark)', fontSize: '0.85rem', margin: 0 }}>
              <strong>Experimental extractor:</strong> The separate schema uses Praat voice measures, band-limited HNR, custom recurrence/DFA/pitch-period entropy/GNE, and documented MFCC/delta settings. Its features are not the saved SVM inputs. Audio is uploaded for processing and not retained. Extracted feature values and their schema are saved to this assessment; predictions remain unavailable until a compatible model is trained.
            </p>
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
        {savedSpeech && (
          <section aria-live="polite" style={{ marginTop: '1.25rem', padding: '1rem', background: '#FAFAF7', border: '1px solid var(--border-subtle)', borderRadius: 6 }}>
            <h3 style={{ marginTop: 0, color: 'var(--sage-dark)' }}>Saved speech processing result ({savedSpeechStatus})</h3>
            <p>{savedSpeech.feature_extraction === 'completed' ? `${savedSpeech.feature_count} ${savedSpeech.feature_schema} features extracted.` : savedSpeech.error || 'No feature extraction result is saved.'}</p>
            {savedSpeech.feature_values && <details><summary>View extracted feature values</summary><dl style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.4rem 1rem' }}>{Object.entries(savedSpeech.feature_values).map(([name, value]) => <div key={name}><dt style={{ fontWeight: 600 }}>{name}</dt><dd style={{ margin: 0 }}>{typeof value === 'number' ? value.toPrecision(6) : String(value)}</dd></div>)}</dl></details>}
            {typeof savedSpeech.probability === 'number' && <p>Model probability: {(savedSpeech.probability * 100).toFixed(1)}% — research output, not a diagnosis.</p>}
          </section>
        )}
      </div>
    </div>
  );
}
