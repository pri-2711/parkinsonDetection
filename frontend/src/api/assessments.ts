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

  async saveCanvasObservation(id: number, samples: any[], canvasWidth: number, canvasHeight: number): Promise<{ assessment: Assessment; observation: any }> {
    return apiClient.post(`/api/assessments/${id}/spiral/canvas/observe`, {
      samples,
      canvas_width: canvasWidth,
      canvas_height: canvasHeight,
    });
  },

  async predictSpeech(id: number, audio: Blob, filename: string, metadata: Record<string, string> = {}): Promise<any> {
    const form = new FormData();
    form.append('file', audio, filename);
    Object.entries(metadata).forEach(([key, value]) => form.append(key, value));
    return apiClient.postForm(`/api/assessments/${id}/speech/predict`, form);
  },

  async analyzeSpiralVideo(id: number, video: File): Promise<{ analysis: any }> {
    const form = new FormData();
    form.append('file', video, video.name);
    return apiClient.postForm(`/api/assessments/${id}/spiral/video/analyze`, form);
  },

  async getSpiralVideoBlob(id: number, variant: 'annotated' | 'original' = 'annotated'): Promise<Blob> {
    return apiClient.getBlob(`/api/assessments/${id}/spiral/video/${variant}`);
  },

  async downloadReportPdf(id: number): Promise<Blob> {
    return apiClient.getBlob(`/api/assessments/${id}/report.pdf`);
  },

  async listHistory(limit: number = 50): Promise<{ assessments: Assessment[] }> {
    return apiClient.get<{ assessments: Assessment[] }>(`/api/history/assessments?limit=${limit}`);
  },

  async listCompleted(limit: number = 100): Promise<{ assessments: Assessment[] }> {
    return apiClient.get<{ assessments: Assessment[] }>(`/api/history/completed?limit=${limit}`);
  },
};

