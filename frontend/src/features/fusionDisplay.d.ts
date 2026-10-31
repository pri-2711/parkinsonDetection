export type FusionDisplayMode = 'none' | 'single' | 'hidden';

export interface FusionLike {
  available_modalities?: string[];
}

export function fusionDisplayMode(fusion: FusionLike | null | undefined): FusionDisplayMode;

export function formatPercent(value: unknown, digits?: number): string;

export const SPEECH_BLOCKER_MESSAGES: Record<string, string>;

export interface SpeechStepLike {
  status?: string;
  error?: string | null;
  outcome?: Record<string, any> | null;
}

export interface SpeechSummary {
  tone: 'success' | 'warning' | 'error' | 'muted';
  title: string;
  modeLabel: string | null;
  text: string;
  // Feature extraction and prediction are separate lines; null when the state has nothing to report.
  extractionText: string | null;
  predictionText: string | null;
  predictionState: 'completed' | 'unavailable' | 'not_run';
}

export const SPEECH_MODE_LABELS: Record<string, string>;

export function speechModeLabel(mode: string | undefined | null): string | null;

export function speechStepSummary(step: SpeechStepLike | null | undefined): SpeechSummary;

export function speechStatusLabel(status: string | undefined): string;
