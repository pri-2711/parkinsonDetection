import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { authApi } from '../api/auth';
import type { LoginRequest } from '../types/api';

export function Login() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState<LoginRequest>({ username: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await authApi.login(formData);
      navigate('/dashboard');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: '400px', marginTop: '4rem' }}>
      <div className="card">
        <h2 style={{ marginBottom: '1.5rem', color: 'var(--sage-dark)' }}>Login</h2>
        <form onSubmit={handleSubmit}>
          <div>
            <label htmlFor="login-username" style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-light)' }}>
              Username
            </label>
            <input
              type="text"
              id="login-username"
              className="input"
              autoComplete="username"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              required
            />
          </div>
          <div>
            <label htmlFor="login-password" style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-light)' }}>
              Password
            </label>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <input
                type={showPassword ? 'text' : 'password'}
                id="login-password"
                className="input"
                autoComplete="current-password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                required
              />
              <button type="button" className="btn btn-secondary" aria-pressed={showPassword} aria-label={showPassword ? 'Hide password' : 'Show password'} onClick={() => setShowPassword((visible) => !visible)}>
                {showPassword ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>
          {error && <p className="error">{error}</p>}
          <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={loading}>
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>
        <p style={{ marginTop: '1rem', textAlign: 'center', color: 'var(--text-light)' }}>
          Don't have an account? <a href="/register" style={{ color: 'var(--sage-green)' }}>Register</a>
        </p>
      </div>
    </div>
  );
}
