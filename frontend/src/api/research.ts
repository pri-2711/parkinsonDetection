import { apiClient } from './client';
import type { ResearchInsights } from '../types/api';

export const researchApi = {
  async getInsights(): Promise<ResearchInsights> {
    return apiClient.get<ResearchInsights>('/api/research/insights');
  },
};
