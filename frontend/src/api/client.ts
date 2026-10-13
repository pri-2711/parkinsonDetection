const API_URL = import.meta.env.VITE_API_URL !== undefined
  ? import.meta.env.VITE_API_URL
  : (typeof window !== 'undefined' && window.location.port === '3000' ? '' : 'http://127.0.0.1:5000');

export interface ApiError extends Error {
  code?: string;
  status?: number;
}

class ApiClient {
  private token: string | null = null;

  setToken(token: string) {
    this.token = token;
    localStorage.setItem('auth_token', token);
  }

  getToken(): string | null {
    if (!this.token && typeof localStorage !== 'undefined') {
      this.token = localStorage.getItem('auth_token');
    }
    return this.token;
  }

  clearToken() {
    this.token = null;
    if (typeof localStorage !== 'undefined') {
      localStorage.removeItem('auth_token');
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const headers = new Headers(options.headers);
    if (options.body && !headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }

    const token = this.getToken();
    if (token) {
      headers.set('Authorization', `Bearer ${token}`);
    }

    let response: Response;
    try {
      response = await fetch(`${API_URL}${endpoint}`, {
        ...options,
        headers,
      });
    } catch (fetchErr) {
      const err = new Error(
        'Unable to connect to the backend server. Please ensure the Flask API is running on http://127.0.0.1:5000.'
      ) as ApiError;
      err.code = 'network_error';
      throw err;
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        error: { message: `Request failed with HTTP status ${response.status}`, code: `http_${response.status}` },
      }));
      const errorMsg = errorData.error?.message || `Request failed with HTTP status ${response.status}`;
      const err = new Error(errorMsg) as ApiError;
      err.code = errorData.error?.code;
      err.status = response.status;
      throw err;
    }

    return response.json();
  }

  async post<T>(endpoint: string, data: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'GET',
    });
  }
}

export const apiClient = new ApiClient();
