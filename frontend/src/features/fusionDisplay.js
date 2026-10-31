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
  speech_audio_v2_model_invalid:
    'The trained model file for this recording mode failed validation, so no prediction was made.',
};

export const SPEECH_MODE_LABELS = {
  reading: 'Reading passage',
  sustained_vowel: 'Sustained "ah" vowel',
};

export function speechModeLabel(mode) {
  return SPEECH_MODE_LABELS[mode] ?? null;
}

// Returns the user-facing state of a saved speech step. Feature extraction and prediction are reported as
// separate states, so features are never shown as a classification result. The title follows the recording
// mode saved with the step, not a fixed label.
export function speechStepSummary(step) {
  if (!step || typeof step !== 'object') {
    return {
      tone: 'muted', title: 'Speech', modeLabel: null,
      text: 'No speech sample has been submitted yet.',
      extractionText: null, predictionText: null, predictionState: 'not_run',
    };
  }
  const outcome = step.outcome || {};
  const modeLabel = speechModeLabel(outcome.recording_mode);
  const title = modeLabel ? `Speech (${modeLabel})` : 'Speech';
  const featureCount = outcome.feature_count ?? 0;
  const schema = outcome.feature_schema || 'speech_audio_v2';
  // An unavailable step only exists after features were extracted, so it counts as extraction completed.
  const extracted = outcome.feature_extraction === 'completed' || step.status === 'unavailable';

  let extractionText = 'Feature extraction not run.';
  if (extracted) {
    extractionText = `Feature extraction completed: ${featureCount} ${schema} features (audio not retained).`;
  } else if (step.status === 'failed') {
    extractionText = 'Feature extraction failed.';
  }

  if (step.status === 'completed' && outcome.model_verified === true && typeof outcome.probability === 'number') {
    const modelName = modeLabel ? `${modeLabel.toLowerCase()} model` : 'speech model';
    return {
      tone: 'success', title, modeLabel, extractionText,
      predictionState: 'completed',
      predictionText: `Prediction completed: ${formatPercent(outcome.probability)} probability from the ${modelName}, a 37-participant validation model. Research output only; not a diagnosis.`,
      text: `Experimental prediction: ${formatPercent(outcome.probability)} model probability. Research output only; not a diagnosis.`,
    };
  }
  if (step.status === 'unavailable') {
    const blocker = SPEECH_BLOCKER_MESSAGES[outcome.blocker_code]
      || 'Prediction is unavailable for this recording mode.';
    return {
      tone: 'warning', title, modeLabel, extractionText,
      predictionState: 'unavailable',
      predictionText: `Prediction unavailable: ${blocker}`,
      text: `${extractionText} ${blocker}`,
    };
  }
  if (step.status === 'failed') {
    const message = `Audio processing failed${step.error ? `: ${step.error}` : '.'} Submit a new recording to try again.`;
    return {
      tone: 'error', title, modeLabel, extractionText: null,
      predictionState: 'not_run', predictionText: 'Prediction not run.',
      text: message,
    };
  }
  if (step.status === 'processing') {
    return {
      tone: 'muted', title, modeLabel, extractionText: null,
      predictionState: 'not_run', predictionText: null,
      text: 'Processing the submitted recording…',
    };
  }
  return {
    tone: 'muted', title, modeLabel, extractionText: null,
    predictionState: 'not_run', predictionText: null,
    text: 'No speech sample has been submitted yet.',
  };
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
