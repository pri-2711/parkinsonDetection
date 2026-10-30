// In `npm run dev`, requests are relative and Vite proxies /api to the backend (see vite.config.ts),
// so the page can run on any port. VITE_API_URL, when set, overrides this.
const API_URL = import.meta.env.VITE_API_URL !== undefined
  ? import.meta.env.VITE_API_URL
  : (import.meta.env.DEV ? '' : 'http://127.0.0.1:5000');

export interface ApiError extends Error {
  code?: string;
  status?: number;
}

const TOKEN_KEY = 'auth_token';
const LEGACY_TOKEN_KEY = 'auth_token';

// Decode the JWT expiry locally. The server still verifies every request; this only avoids sending expired tokens.
function tokenExpiry(token: string): number | null {
  try {
    const payload = token.split('.')[1];
    const json = JSON.parse(atob(payload.replace(/-/g, '+').replace(/_/g, '/')));
    return typeof json.exp === 'number' ? json.exp : null;
  } catch {
    return null;
  }
}

class ApiClient {
  private token: string | null = null;

  constructor() {
    // Tokens older builds kept in localStorage are removed so they do not persist on shared machines.
    try {
      if (typeof localStorage !== 'undefined') localStorage.removeItem(LEGACY_TOKEN_KEY);
    } catch {
      /* storage unavailable: nothing to clean */
    }
  }

  setToken(token: string) {
    this.token = token;
    try {
      sessionStorage.setItem(TOKEN_KEY, token);
    } catch {
      /* storage unavailable: the token lives only for this page load */
    }
  }

  getToken(): string | null {
    if (!this.token) {
      try {
        this.token = sessionStorage.getItem(TOKEN_KEY);
      } catch {
        this.token = null;
      }
    }
    if (this.token) {
      const expiry = tokenExpiry(this.token);
      if (expiry !== null && expiry * 1000 <= Date.now()) {
        this.clearToken();
      }
    }
    return this.token;
  }

  clearToken() {
    this.token = null;
    try {
      sessionStorage.removeItem(TOKEN_KEY);
    } catch {
      /* nothing stored */
    }
  }

  // Renews the session when it expires within an hour. The server refuses renewal past its absolute limit.
  async refreshIfNeeded(): Promise<void> {
    const token = this.getToken();
    if (!token) return;
    const expiry = tokenExpiry(token);
    if (expiry === null || expiry * 1000 - Date.now() > 60 * 60 * 1000) return;
    try {
      const response = await this.post<{ token: string }>('/api/auth/refresh', {});
      this.setToken(response.token);
    } catch {
      this.clearToken();
      window.location.assign('/login');
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const headers = new Headers(options.headers);
    if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
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
      const target = API_URL || (typeof window !== 'undefined' ? window.location.origin : 'the current page origin');
      const err = new Error(
        `Could not reach the backend at ${target}. Check that Flask is running and that VITE_API_URL points to this backend.`
      ) as ApiError;
      err.code = 'network_error';
      throw err;
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        error: { message: `Request failed with HTTP status ${response.status}`, code: `http_${response.status}` },
      }));
      const sessionCodes = ['missing_token', 'invalid_token', 'token_expired', 'session_expired', 'user_not_found'];
      if (response.status === 401 && sessionCodes.includes(errorData.error?.code) && !endpoint.startsWith('/api/auth/')) {
        this.clearToken();
        window.location.assign('/login');
      }
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

  async postForm<T>(endpoint: string, data: FormData): Promise<T> {
    return this.request<T>(endpoint, { method: 'POST', body: data });
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'GET',
    });
  }

  async getBlob(endpoint: string): Promise<Blob> {
    const headers = new Headers();
    const token = this.getToken();
    if (token) headers.set('Authorization', `Bearer ${token}`);
    let response: Response;
    try {
      response = await fetch(`${API_URL}${endpoint}`, { headers });
    } catch {
      const target = API_URL || (typeof window !== 'undefined' ? window.location.origin : 'the current page origin');
      const err = new Error(`Could not reach the backend at ${target}. Check the Flask server and VITE_API_URL.`) as ApiError;
      err.code = 'network_error';
      throw err;
    }
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: {} }));
      const err = new Error(errorData.error?.message || `PDF request failed with HTTP status ${response.status}`) as ApiError;
      err.code = errorData.error?.code;
      err.status = response.status;
      throw err;
    }
    return response.blob();
  }
}

export const apiClient = new ApiClient();
