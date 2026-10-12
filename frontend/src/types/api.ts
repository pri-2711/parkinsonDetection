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
  status: 'pending' | 'completed' | 'failed';
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
      effective_weights: Record<string, number>;
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
