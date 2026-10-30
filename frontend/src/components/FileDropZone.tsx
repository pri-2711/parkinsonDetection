import { useRef, useState } from 'react';
import { Icon } from './Icon';

interface Props {
  id: string;
  file: File | null;
  accept: string;
  acceptLabel: string;
  disabled?: boolean;
  error?: string;
  onFile: (file: File | null) => void;
}

const ACCEPTED_EXTENSIONS = ['.mp4', '.mov', '.avi', '.webm'];

export function isAcceptedVideo(file: File): boolean {
  const name = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext));
}

// Themed drag-and-drop file input. Click or press Enter to browse; drop a file to select it.
export function FileDropZone({ id, file, accept, acceptLabel, disabled = false, error, onFile }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [localError, setLocalError] = useState('');

  const choose = (candidate: File | null) => {
    if (!candidate) return;
    if (!isAcceptedVideo(candidate)) {
      setLocalError(`“${candidate.name}” is not a supported format. Use ${acceptLabel}.`);
      return;
    }
    setLocalError('');
    onFile(candidate);
  };

  const remove = () => {
    setLocalError('');
    if (inputRef.current) inputRef.current.value = '';
    onFile(null);
  };

  const message = localError || error;

  if (file) {
    return (
      <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 10, padding: '0.9rem 1rem', background: 'var(--white)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <span style={{ color: 'var(--sage-green)', display: 'inline-flex' }}><Icon name="video" size={22} /></span>
          <div style={{ flex: 1, minWidth: 180 }}>
            <strong style={{ color: 'var(--text-dark)', wordBreak: 'break-all' }}>{file.name}</strong>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-light)' }}>{(file.size / 1024 / 1024).toFixed(1)} MB</div>
          </div>
          <button type="button" className="btn btn-secondary" disabled={disabled} onClick={() => inputRef.current?.click()}>Replace</button>
          <button type="button" className="btn btn-secondary" disabled={disabled} onClick={remove}>Remove</button>
        </div>
        <input ref={inputRef} id={id} type="file" accept={accept} hidden disabled={disabled}
          onChange={(event) => choose(event.target.files?.[0] ?? null)} />
        {message && <p role="alert" className="error" style={{ margin: '0.6rem 0 0' }}>{message}</p>}
      </div>
    );
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Choose a video file to upload"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(event) => { if ((event.key === 'Enter' || event.key === ' ') && !disabled) inputRef.current?.click(); }}
        onDragOver={(event) => { event.preventDefault(); if (!disabled) setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragOver(false);
          if (!disabled) choose(event.dataTransfer.files?.[0] ?? null);
        }}
        style={{
          border: `2px dashed ${dragOver ? 'var(--sage-green)' : 'var(--sage-light)'}`,
          borderRadius: 12,
          background: dragOver ? '#E6EFE2' : '#F4F7F2',
          padding: '2rem 1.25rem',
          textAlign: 'center',
          cursor: disabled ? 'not-allowed' : 'pointer',
          transition: 'background 0.15s, border-color 0.15s',
          opacity: disabled ? 0.6 : 1,
        }}
      >
        <div style={{ color: 'var(--sage-dark)', display: 'inline-flex' }}><Icon name="upload" size={34} /></div>
        <p style={{ margin: '0.6rem 0 0.2rem', fontWeight: 600, color: 'var(--text-dark)' }}>Drag and drop your video here</p>
        <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-light)' }}>
          or <span style={{ color: 'var(--sage-dark)', textDecoration: 'underline' }}>browse files</span>
        </p>
        <p style={{ margin: '0.6rem 0 0', fontSize: '0.8rem', color: 'var(--text-light)' }}>Supported: {acceptLabel}</p>
        <input
          ref={inputRef}
          id={id}
          type="file"
          accept={accept}
          hidden
          disabled={disabled}
          onClick={(event) => event.stopPropagation()}
          onChange={(event) => choose(event.target.files?.[0] ?? null)}
        />
      </div>
      {message && <p role="alert" className="error" style={{ margin: '0.6rem 0 0' }}>{message}</p>}
    </div>
  );
}
