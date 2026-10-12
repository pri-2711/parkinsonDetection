import { apiClient } from './client';
import type { AuthResponse, RegisterRequest, LoginRequest, User } from '../types/api';

export const authApi = {
  async register(data: RegisterRequest): Promise<User> {
    return apiClient.post<User>('/api/auth/register', data);
  },

  async login(data: LoginRequest): Promise<AuthResponse> {
    const response = await apiClient.post<AuthResponse>('/api/auth/login', data);
    apiClient.setToken(response.token);
    return response;
  },

  async getCurrentUser(): Promise<User> {
    return apiClient.get<User>('/api/auth/me');
  },

  logout() {
    apiClient.clearToken();
  },
};
