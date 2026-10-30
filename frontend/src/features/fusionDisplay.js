// Display rules shared by the Results, Speech and Assessment summary views.
// A combined probability is shown only when exactly one model contributed.

export function fusionDisplayMode(fusion) {
  if (!fusion) return 'none';
  const modalities = Array.isArray(fusion.available_modalities) ? fusion.available_modalities : [];
  if (modalities.length === 1) return 'single';
  if (modalities.length === 0) return 'none';
  return 'hidden';
}

export function formatPercent(value, digits = 1) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'unavailable';
  return `${(value * 100).toFixed(digits)}%`;
}

export const SPEECH_BLOCKER_MESSAGES = {
  speech_audio_v2_model_not_trained:
    'No trained speech_audio_v2 model exists for this recording mode, so no prediction was made.',
};

// Returns the user-facing state of a saved speech step. The error lives on the
// step, while feature details live on the outcome, so both are read here.
export function speechStepSummary(step) {
  if (!step || typeof step !== 'object') {
    return { tone: 'muted', text: 'No speech sample has been submitted yet.' };
  }
  const outcome = step.outcome || {};
  const featureCount = outcome.feature_count ?? 0;
  const schema = outcome.feature_schema || 'speech_audio_v2';
  const mode = outcome.recording_mode || 'recording';

  if (step.status === 'completed' && outcome.model_verified === true && typeof outcome.probability === 'number') {
    return {
      tone: 'success',
      text: `Experimental prediction: ${formatPercent(outcome.probability)} model probability (reading-passage model, 37-participant validation). Research output only; not a diagnosis.`,
    };
  }
  if (step.status === 'unavailable') {
    const blocker = SPEECH_BLOCKER_MESSAGES[outcome.blocker_code]
      || 'Prediction is unavailable for this recording mode.';
    return {
      tone: 'warning',
      text: `${featureCount} ${schema} features extracted from a ${mode} recording (audio not retained). ${blocker}`,
    };
  }
  if (step.status === 'failed') {
    return {
      tone: 'error',
      text: `Audio processing failed${step.error ? `: ${step.error}` : '.'} Submit a new recording to try again.`,
    };
  }
  if (step.status === 'processing') {
    return { tone: 'muted', text: 'Processing the submitted recording…' };
  }
  return { tone: 'muted', text: 'No speech sample has been submitted yet.' };
}

export function speechStatusLabel(status) {
  switch (status) {
    case 'completed': return 'Prediction saved';
    case 'unavailable': return 'Features only';
    case 'failed': return 'Failed';
    case 'processing': return 'Processing';
    default: return 'Not submitted';
  }
}
