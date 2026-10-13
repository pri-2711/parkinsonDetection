interface StatusCardProps {
  title: string;
  status: 'pending' | 'completed' | 'failed' | 'unavailable';
  description?: string;
  onClick?: () => void;
}

export function StatusCard({ title, status, description, onClick }: StatusCardProps) {
  const statusLabels = {
    pending: 'Pending',
    completed: 'Completed',
    failed: 'Action Required / Blocked',
    unavailable: 'Protocol Notice',
  };

  return (
    <div
      className={`card ${onClick ? 'card-interactive' : ''}`}
      onClick={onClick}
      style={{ cursor: onClick ? 'pointer' : 'default', height: '100%', display: 'flex', flexDirection: 'column' }}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={(e) => {
        if (onClick && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          onClick();
        }
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <h3 style={{ color: 'var(--sage-dark)', fontSize: '1.15rem', fontWeight: '700' }}>{title}</h3>
        <span className={`status-badge status-${status}`}>
          {statusLabels[status]}
        </span>
      </div>

      {description && (
        <p style={{ color: 'var(--text-light)', fontSize: '0.9rem', lineHeight: '1.5', flex: 1 }}>
          {description}
        </p>
      )}

      {onClick && (
        <div style={{ marginTop: '1.25rem', display: 'flex', alignItems: 'center', color: 'var(--sage-green)', fontWeight: '600', fontSize: '0.9rem' }}>
          <span>Open Assessment →</span>
        </div>
      )}
    </div>
  );
}
