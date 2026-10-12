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
      <header style={{ backgroundColor: 'var(--sage-green)', padding: '1rem 2rem', color: 'white' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: '600' }}>Parkinson's Detection</h1>
          <nav>
            <Link to="/dashboard" style={{ color: 'white', textDecoration: 'none', marginRight: '1.5rem' }}>
              Dashboard
            </Link>
            <Link to="/history" style={{ color: 'white', textDecoration: 'none', marginRight: '1.5rem' }}>
              History
            </Link>
            <button onClick={handleLogout} style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer', fontSize: '1rem' }}>
              Logout
            </button>
          </nav>
        </div>
      </header>
      <main style={{ flex: 1 }}>
        <Outlet />
      </main>
      <footer style={{ backgroundColor: 'var(--cream-dark)', padding: '1rem 2rem', textAlign: 'center', color: 'var(--text-light)', fontSize: '0.875rem' }}>
        <p>This is a research prototype and not a clinically validated diagnostic tool.</p>
      </footer>
    </div>
  );
}
