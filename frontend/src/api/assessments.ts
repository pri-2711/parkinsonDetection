import { apiClient } from './client';
import type { Assessment, TremorRequest } from '../types/api';

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

  async listHistory(limit: number = 50): Promise<{ assessments: Assessment[] }> {
    return apiClient.get<{ assessments: Assessment[] }>(`/api/history/assessments?limit=${limit}`);
  },

  async listCompleted(limit: number = 100): Promise<{ assessments: Assessment[] }> {
    return apiClient.get<{ assessments: Assessment[] }>(`/api/history/completed?limit=${limit}`);
  },
};
