import { Outlet, Link, useNavigate } from 'react-router-dom';
import { authApi } from '../api/auth';

export function Layout() {
  const navigate = useNavigate();

  const handleLogout = () => {
    authApi.logout();
    navigate('/login');
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ backgroundColor: 'var(--sage-dark)', padding: '1rem 2rem', color: '#FFFFFF', boxShadow: '0 2px 6px rgba(0,0,0,0.1)' }}>
        <div style={{ maxWidth: '1100px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Link to="/dashboard" style={{ color: '#FFFFFF', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontSize: '1.4rem' }}>🌿</span>
              <span style={{ fontSize: '1.25rem', fontWeight: '700', letterSpacing: '-0.01em' }}>Parkinson's Research Lab</span>
            </Link>
          </div>
          <nav style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <Link to="/dashboard" style={{ color: '#FFFFFF', textDecoration: 'none', fontWeight: '500', fontSize: '0.95rem', opacity: 0.9 }}>
              Dashboard
            </Link>
            <Link to="/history" style={{ color: '#FFFFFF', textDecoration: 'none', fontWeight: '500', fontSize: '0.95rem', opacity: 0.9 }}>
              History
            </Link>
            <button
              onClick={handleLogout}
              style={{
                background: 'rgba(255,255,255,0.15)',
                border: '1px solid rgba(255,255,255,0.3)',
                color: '#FFFFFF',
                borderRadius: '4px',
                padding: '0.4rem 0.85rem',
                cursor: 'pointer',
                fontSize: '0.9rem',
                fontWeight: '600',
                transition: 'background 0.2s',
              }}
              onMouseOver={(e) => (e.currentTarget.style.background = 'rgba(255,255,255,0.25)')}
              onMouseOut={(e) => (e.currentTarget.style.background = 'rgba(255,255,255,0.15)')}
            >
              Logout
            </button>
          </nav>
        </div>
      </header>

      <main style={{ flex: 1 }}>
        <Outlet />
      </main>

      <footer style={{ backgroundColor: 'var(--cream-dark)', borderTop: '1px solid var(--border-subtle)', padding: '1.25rem 2rem', textAlign: 'center', color: 'var(--text-light)', fontSize: '0.85rem' }}>
        <p>Algorithmic research demonstration only. Not a clinically validated diagnostic tool.</p>
      </footer>
    </div>
  );
}
