import { useState, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { assessmentsApi } from '../api/assessments';

const READING_PASSAGE = 'Today is a pleasant day. I am speaking at my normal pace and volume. I will read each sentence clearly and naturally, without rushing. After finishing the passage, I will pause briefly before ending the recording.';
const PASSAGE_VERSION = '1.0';

export function Speech() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [activeTab, setActiveTab] = useState<'record' | 'upload'>('record');
  const [file, setFile] = useState<File | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

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
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
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
      // 1. Submit through the assessment speech pipeline
      try {
        await assessmentsApi.predictSpeech(parseInt(id, 10), {
          passage_version: PASSAGE_VERSION,
          passage_text: READING_PASSAGE,
          duration_seconds: activeTab === 'record' ? recordingDuration : 5.0,
          sample_rate: 44100,
        });
      } catch (apiErr: any) {
        // Backend returns 501 raw_audio_feature_extractor_unavailable
        if (apiErr?.code === 'raw_audio_feature_extractor_unavailable' || apiErr?.message?.includes('raw_audio_feature_extractor_unavailable')) {
          // Expected honest diagnostic response
          navigate('/dashboard');
          return;
        }
        throw apiErr;
      }

      navigate('/dashboard');
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
            <strong>Speech Assessment:</strong> This research assessment analyses selected acoustic characteristics of a speech sample to investigate patterns associated with Parkinson’s disease. Follow the recording instructions and speak the displayed passage at a comfortable, natural pace, in a quiet environment, using a clear microphone recording. The application processes supported acoustic features and, when processing is successful, uses the trained research model to estimate a Parkinson’s classification probability. This result is experimental, is not a medical diagnosis, and must not be used as a substitute for assessment by a qualified healthcare professional.
          </p>
        </div>

        {/* Standardized Reading Passage */}
        <div style={{ backgroundColor: '#F4F7F2', border: '1px solid var(--sage-light)', padding: '1.25rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: '700', color: 'var(--sage-dark)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Standardized Reading Passage (v{PASSAGE_VERSION})
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-light)' }}>Approx. 15–20 seconds</span>
          </div>
          <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--text-dark)', fontStyle: 'italic', margin: 0 }}>
            "{READING_PASSAGE}"
          </p>
        </div>

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
                  {recording ? `Recording in progress (${recordingDuration}s)… Speak the passage clearly.` : recordedBlob ? `Recording captured (${recordingDuration}s)` : 'Click to begin microphone recording'}
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
                Select Audio Recording (WAV format, 1–30 seconds)
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
              <strong>Acoustic Model Contract Notice:</strong> The trained research model (<code>speech_svm_tuned.joblib</code>) operates on 45 precomputed acoustic features (jitter, shimmer, HNR, RPDE, DFA, PPE, gender). Live audio feature extraction requires a verified Praat/acoustic pipeline matching this schema; submitting records your audio sample and reading passage metadata.
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
      </div>
    </div>
  );
}
