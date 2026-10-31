import test from 'node:test';
import assert from 'node:assert/strict';
import { fusionDisplayMode, formatPercent, speechStepSummary, speechStatusLabel, speechModeLabel } from './fusionDisplay.js';

test('fused probability is shown only for a single contributing model', () => {
  assert.equal(fusionDisplayMode({ available_modalities: ['speech'] }), 'single');
  assert.equal(fusionDisplayMode({ available_modalities: ['speech', 'tremor'] }), 'hidden');
  assert.equal(fusionDisplayMode({ available_modalities: [] }), 'none');
  assert.equal(fusionDisplayMode(null), 'none');
  assert.equal(fusionDisplayMode(undefined), 'none');
});

test('percentages never show NaN or undefined', () => {
  assert.equal(formatPercent(0.4567), '45.7%');
  assert.equal(formatPercent(undefined), 'unavailable');
  assert.equal(formatPercent(Number.NaN), 'unavailable');
});

test('unavailable speech is reported as extracted features without a prediction', () => {
  const summary = speechStepSummary({
    status: 'unavailable',
    error: 'legacy text that must not drive the message',
    outcome: {
      feature_count: 45,
      feature_schema: 'speech_audio_v2',
      recording_mode: 'sustained_vowel',
      blocker_code: 'speech_audio_v2_model_not_trained',
    },
  });
  assert.equal(summary.tone, 'warning');
  assert.equal(summary.title, 'Speech (Sustained "ah" vowel)');
  assert.match(summary.extractionText, /Feature extraction completed: 45 speech_audio_v2 features/);
  assert.match(summary.predictionText, /^Prediction unavailable: No trained speech_audio_v2 model exists/);
  assert.match(summary.text, /No trained speech_audio_v2 model exists/);
  assert.doesNotMatch(summary.text, /legacy text/);
});

test('failed speech shows the step error rather than a missing-result message', () => {
  const summary = speechStepSummary({ status: 'failed', error: 'Audio is silent.', outcome: {} });
  assert.equal(summary.tone, 'error');
  assert.match(summary.text, /Audio processing failed: Audio is silent\./);
  assert.doesNotMatch(summary.text, /No feature extraction/);
});

test('a probability is shown as a prediction only when it was verified', () => {
  const verified = speechStepSummary({ status: 'completed', outcome: { probability: 0.25, model_verified: true } });
  assert.equal(verified.tone, 'success');
  assert.match(verified.text, /25\.0% model probability/);
  assert.match(verified.text, /Experimental prediction/);
  const unverified = speechStepSummary({ status: 'completed', outcome: { probability: 0.25, model_verified: false } });
  assert.doesNotMatch(unverified.text, /Prediction saved/);
});

test('empty and processing speech states have plain messages', () => {
  assert.match(speechStepSummary(null).text, /No speech sample/);
  assert.match(speechStepSummary({ status: 'processing' }).text, /Processing/);
  assert.equal(speechStatusLabel('unavailable'), 'Features only');
  assert.equal(speechStatusLabel(undefined), 'Not submitted');
});

test('the speech title follows the saved recording mode', () => {
  const reading = speechStepSummary({ status: 'completed', outcome: { recording_mode: 'reading', feature_extraction: 'completed', feature_count: 45, probability: 0.31, model_verified: true } });
  assert.equal(reading.title, 'Speech (Reading passage)');
  assert.equal(reading.predictionState, 'completed');
  assert.match(reading.predictionText, /^Prediction completed: 31\.0% probability from the reading passage model/);

  const vowel = speechStepSummary({ status: 'unavailable', outcome: { recording_mode: 'sustained_vowel', feature_extraction: 'completed', feature_count: 45, blocker_code: 'speech_audio_v2_model_not_trained' } });
  assert.equal(vowel.title, 'Speech (Sustained "ah" vowel)');
  assert.doesNotMatch(vowel.title, /reading/i);
  assert.equal(vowel.predictionState, 'unavailable');
  assert.doesNotMatch(vowel.extractionText, /Prediction/);
});

test('feature extraction and prediction are never merged into one line for a sustained vowel', () => {
  const vowel = speechStepSummary({ status: 'unavailable', outcome: { recording_mode: 'sustained_vowel', feature_extraction: 'completed', feature_count: 45, blocker_code: 'speech_audio_v2_model_not_trained' } });
  assert.notEqual(vowel.extractionText, vowel.predictionText);
  assert.doesNotMatch(vowel.predictionText, /features/);
});

test('speech mode labels are only known modes', () => {
  assert.equal(speechModeLabel('reading'), 'Reading passage');
  assert.equal(speechModeLabel('Reading'), null);
  assert.equal(speechModeLabel(undefined), null);
});
