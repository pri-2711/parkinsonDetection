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
    failed: 'Failed',
    unavailable: 'Unavailable',
  };

  return (
    <div className="card" onClick={onClick} style={{ cursor: onClick ? 'pointer' : 'default' }}>
      <h3 style={{ marginBottom: '0.5rem', color: 'var(--sage-dark)' }}>{title}</h3>
      <span className={`status-badge status-${status}`}>
        {statusLabels[status]}
      </span>
      {description && (
        <p style={{ marginTop: '1rem', color: 'var(--text-light)', fontSize: '0.875rem' }}>
          {description}
        </p>
      )}
    </div>
  );
}
