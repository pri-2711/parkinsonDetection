import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { authApi } from '../api/auth';
import type { RegisterRequest } from '../types/api';

export function Register() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState<RegisterRequest>({ username: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await authApi.register(formData);
      navigate('/login');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: '400px', marginTop: '4rem' }}>
      <div className="card">
        <h2 style={{ marginBottom: '1.5rem', color: 'var(--sage-dark)' }}>Register</h2>
        <form onSubmit={handleSubmit}>
          <div>
            <label htmlFor="register-username" style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-light)' }}>
              Username
            </label>
            <input
              type="text"
              id="register-username"
              className="input"
              autoComplete="username"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              required
            />
          </div>
          <div>
            <label htmlFor="register-email" style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-light)' }}>
              Email
            </label>
            <input
              type="email"
              id="register-email"
              className="input"
              autoComplete="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              required
            />
          </div>
          <div>
            <label htmlFor="register-password" style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-light)' }}>
              Password (min 8 characters)
            </label>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <input
                type={showPassword ? 'text' : 'password'}
                id="register-password"
                className="input"
                autoComplete="new-password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                required
                minLength={8}
              />
              <button type="button" className="btn btn-secondary" aria-pressed={showPassword} aria-label={showPassword ? 'Hide password' : 'Show password'} onClick={() => setShowPassword((visible) => !visible)}>
                {showPassword ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>
          {error && <p className="error">{error}</p>}
          <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={loading}>
            {loading ? 'Registering...' : 'Register'}
          </button>
        </form>
        <p style={{ marginTop: '1rem', textAlign: 'center', color: 'var(--text-light)' }}>
          Already have an account? <a href="/login" style={{ color: 'var(--sage-green)' }}>Login</a>
        </p>
      </div>
    </div>
  );
}
