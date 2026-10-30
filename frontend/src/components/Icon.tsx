// Outlined icons drawn with stroke="currentColor" so they follow the surrounding text colour.
const PATHS: Record<string, string[]> = {
  download: ['M12 4v11', 'M7 10l5 5 5-5', 'M5 19h14'],
  upload: ['M12 16V5', 'M7 9l5-5 5 5', 'M5 19h14'],
  file: ['M7 3h7l5 5v13H7z', 'M14 3v5h5'],
  mic: ['M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z', 'M5 11a7 7 0 0 0 14 0', 'M12 18v3'],
  video: ['M3 7h12v10H3z', 'M15 10l6-3v10l-6-3'],
  alert: ['M12 3l10 18H2z', 'M12 10v5', 'M12 18h.01'],
};

interface IconProps {
  name: keyof typeof PATHS;
  size?: number;
  label?: string;
}

export function Icon({ name, size = 18, label }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden={label ? undefined : true}
      role={label ? 'img' : undefined}
      aria-label={label}
      style={{ verticalAlign: '-3px', marginRight: label === undefined ? '0.4rem' : 0 }}
    >
      {PATHS[name].map((d) => <path key={d} d={d} />)}
    </svg>
  );
}
