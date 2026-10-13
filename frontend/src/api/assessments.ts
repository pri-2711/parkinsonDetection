import { apiClient } from './client';
import type { Assessment } from '../types/api';

export const assessmentsApi = {
  async create(): Promise<Assessment> {
    return apiClient.post<Assessment>('/api/assessments', {});
  },

  async get(id: number): Promise<Assessment> {
    return apiClient.get<Assessment>(`/api/assessments/${id}`);
  },

  async complete(id: number): Promise<Assessment> {
    return apiClient.post<Assessment>(`/api/assessments/${id}/complete`, {});
  },

  async getResults(id: number): Promise<Assessment['results']> {
    return apiClient.get<Assessment['results']>(`/api/assessments/${id}/results`);
  },

  async updateSpeech(id: number, outcome: any, error: string | null = null): Promise<Assessment> {
    return apiClient.post<Assessment>(`/api/assessments/${id}/speech`, { outcome, error });
  },

  async updateSpiral(id: number, outcome: any, error: string | null = null): Promise<Assessment> {
    return apiClient.post<Assessment>(`/api/assessments/${id}/spiral`, { outcome, error });
  },

  async updateTremor(id: number, outcome: any, error: string | null = null): Promise<Assessment> {
    return apiClient.post<Assessment>(`/api/assessments/${id}/tremor`, { outcome, error });
  },

  async predictTremor(id: number, features: Record<string, number>): Promise<{ outcome: any }> {
    return apiClient.post<{ outcome: any }>(`/api/assessments/${id}/tremor/predict`, features);
  },

  async predictSpiral(id: number, trajectory: any[], inputMode: string = 'canvas_drawing'): Promise<{ outcome: any }> {
    return apiClient.post<{ outcome: any }>(`/api/assessments/${id}/spiral/predict`, { trajectory, input_mode: inputMode });
  },

  async predictSpeech(id: number, metadata: Record<string, any> = {}): Promise<any> {
    return apiClient.post(`/api/assessments/${id}/speech/predict`, metadata);
  },

  async downloadReportPdf(id: number): Promise<Blob> {
    const token = apiClient.getToken();
    const response = await fetch(`/api/assessments/${id}/report.pdf`, {
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    });
    if (!response.ok) {
      throw new Error('Failed to download report PDF');
    }
    return response.blob();
  },

  async listHistory(limit: number = 50): Promise<{ assessments: Assessment[] }> {
    return apiClient.get<{ assessments: Assessment[] }>(`/api/history/assessments?limit=${limit}`);
  },

  async listCompleted(limit: number = 100): Promise<{ assessments: Assessment[] }> {
    return apiClient.get<{ assessments: Assessment[] }>(`/api/history/completed?limit=${limit}`);
  },
};

