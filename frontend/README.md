# Parkinson's Detection Frontend

React + TypeScript + Vite frontend for the Parkinson's Detection assessment system.

## Setup

```powershell
cd frontend
npm install
```

## Environment

Copy `.env.example` to `.env` and configure:

```
VITE_API_URL=http://127.0.0.1:5000
```

## Development

```powershell
npm run dev
```

Frontend runs on `http://localhost:3000` with API proxy to backend.

## Build

```powershell
npm run build
```

## Features

- User authentication (register, login, logout)
- Assessment dashboard with status tracking
- Handwriting, speech, and tremor test pages
- Results display with fusion classification
- Assessment history view
- Responsive design with sage-green/cream palette

## Limitations

- Speech prediction: Raw audio feature extraction not implemented
- Handwriting prediction: Camera-to-tablet feature mapping unavailable
- Tremor: Self-reported input only, no model classification
- This is a research prototype, not a clinical diagnostic tool
