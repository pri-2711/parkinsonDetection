export interface User {
  id: number;
  username: string;
  email: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface AssessmentStep {
  status: 'pending' | 'processing' | 'completed' | 'unavailable' | 'failed';
  outcome: {
    prediction?: number;
    probability?: number;
    [key: string]: any;
  } | null;
  error: string | null;
  updated_at: string | null;
}

export interface Assessment {
  id: number;
  user_id: number;
  status: 'in_progress' | 'completed';
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  steps: {
    speech: AssessmentStep;
    spiral: AssessmentStep;
    tremor: AssessmentStep;
  };
  results: {
    assessment_id: number;
    completed_at: string;
    modalities: {
      speech: AssessmentStep;
      spiral: AssessmentStep;
      tremor: AssessmentStep;
    };
    fusion?: {
      probability: number;
      prediction: number;
      available_modalities: string[];
      missing_modalities?: string[];
      effective_weights: Record<string, number>;
      aggregation_type?: string;
      score_interpretation?: string;
      threshold?: number;
    };
  } | null;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface TremorRequest {
  location: string;
  intensity: number;
  session_id?: string;
}

export interface SummaryModality {
  modality: 'tremor' | 'spiral' | 'speech';
  label: string;
  step_status: string;
  processed: boolean;
  prediction_available: boolean;
  probability: number | null;
  prediction_note: string;
  observations: string[];
  contributes_to_fusion: boolean;
  error: string | null;
}

export interface AssessmentSummary {
  assessment_id: number;
  status: 'in_progress' | 'completed';
  generated_at: string;
  modalities: SummaryModality[];
  completed_modalities: string[];
  predictions_available: string[];
  experimental_observations: { modality: string; text: string }[];
  fusion: {
    state: 'not_finalized' | 'no_contributions' | 'single_modality' | 'multi_modality_aggregate';
    contributing_modalities: string[];
    fused_probability_displayed: boolean;
    single_modality_probability: number | null;
    note: string;
  };
  limitations: string[];
}

export interface MetricSummary {
  mean: number | null;
  sd: number | null;
  n: number;
}

export interface ResearchInsights {
  disclaimer: string;
  modalities: {
    modality: 'speech' | 'handwriting' | 'tremor';
    label: string;
    dataset: string;
    live_use: string;
    rows: number;
    participant_groups: number;
    feature_count: number;
    reported_folds: number[];
    reported_rows: number | null;
    pooled_metrics: Record<string, Record<'accuracy' | 'precision' | 'recall' | 'f1' | 'roc_auc', number>>;
    fold_metrics: Record<string, Record<'accuracy' | 'precision' | 'recall' | 'f1' | 'roc_auc', MetricSummary>>;
    deployed_family: string | null;
    deployed_params: Record<string, unknown> | null;
    inner_cv_roc_auc: Record<string, number | null>;
    fusion_validation: Record<string, unknown> | null;
    source: string;
  }[];
  fusion: {
    method: string;
    validation_auc: Record<string, number>;
    weights: Record<string, number>;
    weights_sum: number;
    validation_fold: number;
    test_fold_labels_used: boolean;
    patient_level_performance: null;
    baseline_v1_weights_for_audit: Record<string, number>;
    baseline_v1_weights_note: string;
  };
  speech_audio_v2: {
    schema: string;
    feature_count: number;
    features: string[];
    modes: Record<string, {
      model_trained: boolean;
      recordings?: number;
      subjects?: number;
      control_subjects?: number;
      parkinson_subjects?: number;
      majority_class_accuracy?: number;
      deployed_family?: string;
      outer_fold_roc_auc_mean?: number;
      outer_fold_roc_auc_sd?: number;
      skipped_recordings?: number;
      validation?: string;
    }>;
    status: 'experimental_prediction' | 'extraction_only';
    blocker: string | null;
    limitations: string[];
    validation_protocol: string;
  };
  architecture: string[];
  validation: Record<string, string | number | null>;
  datasets_note: string;
}
