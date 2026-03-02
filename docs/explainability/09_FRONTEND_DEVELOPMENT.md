# 09 — Frontend & User Experience

## Introduction

The frontend is what clinicians interact with directly. It translates complex quantum-classical inference pipelines, heatmap generation, and segmentation outputs into an intuitive diagnostic interface. Built with **Next.js 14** (App Router), **React 18**, and **Tailwind CSS**, the frontend adopts a distinctive **terminal-aesthetic** design — monospace fonts, black/white/green colour scheme, and command-line-inspired UI elements — reinforcing the system's technical precision.

This document covers the frontend architecture, user flow, data visualisation, and design decisions.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| **Next.js 14** | React meta-framework with App Router, server components, and standalone output |
| **React 18** | Component library with hooks, state management, and effects |
| **Tailwind CSS** | Utility-first CSS framework for rapid, consistent styling |
| **Recharts** | React charting library for confidence visualisations |
| **EventSource API** | Server-Sent Events for real-time progress streaming |
| **TypeScript** | Type safety across the frontend codebase |

---

## Application Structure

```
frontend/
├── app/
│   ├── globals.css        ← Terminal aesthetic global styles
│   ├── layout.tsx         ← Root layout with header/footer
│   ├── page.tsx           ← Main diagnosis interface (~800 lines)
│   └── dashboard/
│       └── page.tsx       ← System status & model specs
├── lib/
│   └── utils.ts           ← Utility functions
├── public/                ← Static assets
├── next.config.js         ← Build configuration
├── tailwind.config.js     ← Tailwind customisation
├── postcss.config.js      ← PostCSS plugins
└── tsconfig.json          ← TypeScript configuration
```

---

## The Main Diagnosis Page

The primary interface (`page.tsx`) is an 800+ line React component that handles the full diagnostic workflow. It is divided into several functional zones:

### Zone 1: Image Upload

```tsx
// Drop zone with drag-and-drop support
<div
  onDragOver={handleDragOver}
  onDrop={handleDrop}
  onClick={() => fileInputRef.current?.click()}
  className="border-2 border-dashed border-green-500/50 rounded-lg p-12 
             text-center cursor-pointer hover:border-green-400 transition-colors"
>
  <p className="text-green-400 font-mono">
    retina-q&gt; DROP IMAGE HERE OR CLICK TO UPLOAD
  </p>
  <p className="text-gray-500 font-mono text-sm mt-2">
    Supported: JPEG, PNG | OCT or Fundus images
  </p>
</div>
```

The upload zone:
- Accepts drag-and-drop or click-to-browse
- Validates file type (JPEG/PNG only)
- Shows image preview after selection
- Allows switching between OCT and Fundus mode

### Zone 2: Scan Type Selection

```tsx
<div className="flex gap-4">
  <button
    onClick={() => setScanType("oct")}
    className={`px-6 py-3 font-mono ${
      scanType === "oct" ? "bg-green-600 text-black" : "bg-gray-800 text-green-400"
    }`}
  >
    [OCT SCAN]
  </button>
  <button
    onClick={() => setScanType("fundus")}
    className={`px-6 py-3 font-mono ${
      scanType === "fundus" ? "bg-green-600 text-black" : "bg-gray-800 text-green-400"
    }`}
  >
    [FUNDUS IMAGE]
  </button>
</div>
```

### Zone 3: Real-Time Progress Display

Once an image is submitted, the UI connects to the SSE stream and shows live progress:

```tsx
const startAnalysis = async () => {
  // Upload the image
  const formData = new FormData();
  formData.append("file", selectedFile);
  
  const response = await fetch(
    `${API_URL}/api/predict/${scanType}?async_mode=true`,
    { method: "POST", body: formData }
  );
  const { job_id } = await response.json();
  
  // Connect to SSE stream
  const eventSource = new EventSource(
    `${API_URL}/api/jobs/${job_id}/stream`
  );
  
  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    setProgress(data.progress);
    setCurrentStep(data.step);
    
    if (data.status === "complete") {
      setResult(data.result);
      eventSource.close();
    }
  };
};
```

The progress display mimics a terminal output:

```
retina-q> INITIATING QUANTUM ANALYSIS...
[████████████████░░░░░░░░░░░░░░] 52%
> Loading model weights.............. ✓
> Preprocessing image................ ✓
> Quantum circuit inference.......... ●
> Generating explainability map...... ○
> Compiling results.................. ○
```

Each step shows a checkmark (✓) when complete, a spinning indicator (●) when in progress, and an empty circle (○) when pending.

### Zone 4: Results Display (Tabbed Interface)

Results are shown in a tabbed layout:

**Tab 1: Classification**
```
╔══════════════════════════════════════╗
║  DIAGNOSIS: CSCR DETECTED           ║
║  CONFIDENCE: 87.3%                  ║
║  MODEL: Quantum Fundus (4-qubit)    ║
║  PROCESSING TIME: 12.4s             ║
╚══════════════════════════════════════╝
```

Along with a confidence bar and a pie chart (Recharts):

```tsx
<PieChart width={200} height={200}>
  <Pie
    data={[
      { name: "CSR", value: confidence },
      { name: "Normal", value: 1 - confidence }
    ]}
    dataKey="value"
    cx="50%"
    cy="50%"
    innerRadius={60}
    outerRadius={80}
  >
    <Cell fill="#22c55e" />   {/* Green for detected class */}
    <Cell fill="#374151" />   {/* Gray for other class */}
  </Pie>
</PieChart>
```

**Tab 2: Heatmap (Explainability)**

Displays the Grad-CAM or feature importance heatmap overlaid on the original image:

```tsx
<div className="relative">
  <img src={result.heatmap} alt="Explainability Heatmap" className="w-full rounded" />
  <p className="text-green-400 font-mono text-sm mt-2">
    Warm regions (red/yellow) indicate areas of highest model attention.
    Cool regions (blue) indicate low relevance to the prediction.
  </p>
</div>
```

**Tab 3: Segmentation** (Fundus only)

Shows the U-Net macular segmentation mask overlaid on the original:

```tsx
{result.segmentation && (
  <div>
    <img src={result.segmentation} alt="Macular Segmentation" className="w-full rounded" />
    <p className="text-green-400 font-mono text-sm mt-2">
      Green overlay: Segmented macular region (Dice: 0.9663)
    </p>
  </div>
)}
```

**Tab 4: Feature Importance** (OCT only)

Radar chart showing the contribution of each feature group:

```tsx
<RadarChart data={featureImportance}>
  <PolarGrid stroke="#22c55e33" />
  <PolarAngleAxis dataKey="name" tick={{ fill: "#22c55e" }} />
  <Radar dataKey="importance" stroke="#22c55e" fill="#22c55e" fillOpacity={0.3} />
</RadarChart>
```

### Zone 5: Doctor Feedback

After reviewing results, the clinician can submit feedback:

```tsx
<div className="flex gap-4">
  <button
    onClick={() => submitFeedback("concur")}
    className="bg-green-600 text-black px-6 py-3 font-mono"
  >
    [CONCUR WITH DIAGNOSIS]
  </button>
  <button
    onClick={() => setShowOverrideForm(true)}
    className="bg-red-600 text-white px-6 py-3 font-mono"
  >
    [OVERRIDE - INCORRECT]
  </button>
</div>
```

If overriding, a form appears to select the correct label and add notes.

---

## Dashboard Page

The dashboard (`dashboard/page.tsx`) provides system-level visibility:

### Model Specifications Table

| Model | Qubits | Layers | Parameters | Dataset | Target Accuracy |
|---|---|---|---|---|---|
| OCT Classifier | 8 | 8 | ~3K | Kermany2018 | ≥ 92% |
| Fundus Classifier | 4 | 6 | ~5.3M | ODIR-5K | ≥ 93% |
| U-Net Segmentation | N/A | 4 enc + 4 dec | ~31M | ODIR-5K | Dice ≥ 0.90 |

### System Status

- Backend API: `●` Connected / `○` Disconnected
- Redis: `●` Available / `○` Unavailable
- PostgreSQL: `●` Connected / `○` Disconnected
- Celery Workers: `2/2` active
- Models Loaded: `3/3`

### API Endpoint Documentation

An interactive list of all API endpoints with descriptions, methods, and example payloads.

---

## Terminal Aesthetic Design System

### Design Philosophy

The terminal aesthetic was chosen deliberately:

1. **Technical authority**: Monospace fonts and command-line prompts signal precision and expertise.
2. **Focus**: Black backgrounds with green text minimise visual distraction, directing attention to the medical content.
3. **Accessibility**: High contrast (green on black) is readable in dim clinical environments.
4. **Differentiation**: No other medical imaging tool looks like this — it immediately signals "advanced AI system."

### CSS Implementation

```css
/* globals.css */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

:root {
  --terminal-green: #22c55e;
  --terminal-bg: #0a0a0a;
  --terminal-dim: #374151;
}

body {
  font-family: 'IBM Plex Mono', monospace;
  background-color: var(--terminal-bg);
  color: #e5e7eb;
}

/* Animated loading bar */
@keyframes pulse-green {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}

.loading-bar {
  animation: pulse-green 1.5s ease-in-out infinite;
}

/* Terminal prompt styling */
.terminal-prompt::before {
  content: "retina-q> ";
  color: var(--terminal-green);
  font-weight: 600;
}

/* Scan line effect */
@keyframes scanline {
  0% { transform: translateY(-100%); }
  100% { transform: translateY(100%); }
}
```

### Colour Palette

| Colour | Hex | Usage |
|---|---|---|
| Terminal Green | `#22c55e` | Primary text, borders, active elements |
| Background | `#0a0a0a` | Page background |
| Card Background | `#111827` | Elevated surfaces |
| Dim Text | `#6b7280` | Secondary information |
| Alert Red | `#ef4444` | Errors, CSCR alerts, override buttons |
| White | `#f9fafb` | Headings, emphasis |

---

## Data Flow: Frontend ↔ Backend

```
┌─────────────────────────────┐
│         FRONTEND            │
│                             │
│  1. User selects image      │
│  2. User clicks "Analyse"   │
│         │                   │
│  POST /api/predict/fundus   │───────────▶  Backend receives upload
│         │                   │              Dispatches Celery task
│  ◄──────│───────────────────│──────────── Returns job_id (202)
│         │                   │
│  EventSource SSE stream     │───────────▶  Backend streams progress
│  onmessage: update UI       │◄──────────── {step, progress, status}
│         │                   │
│  Results received           │
│  3. Render classification   │
│  4. Render heatmap          │
│  5. Render segmentation     │
│         │                   │
│  6. Doctor clicks feedback  │
│  POST /api/feedback         │───────────▶  Backend logs feedback
│         │                   │
│  ◄──────│───────────────────│──────────── Returns confirmation
└─────────────────────────────┘
```

---

## Build & Deployment

### Development

```bash
cd frontend
npm install
npm run dev   # Starts on http://localhost:3000
```

### Production Build

```bash
npm run build   # Generates .next/ with standalone output
npm start       # Serves production build
```

### Docker

```dockerfile
# Multi-stage build
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
```

The multi-stage build produces a minimal production image:
1. **Builder stage**: Installs all dependencies and builds the Next.js application.
2. **Runner stage**: Copies only the standalone output (no `node_modules`), resulting in a ~100MB image.

### Next.js Configuration

```javascript
// next.config.js
module.exports = {
  output: "standalone",       // Self-contained output for Docker
  images: {
    unoptimized: true,        // Required for static export (Capacitor)
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  },
};
```

The `NEXT_PUBLIC_API_URL` environment variable controls which backend the frontend connects to — configurable per deployment environment.

---

## Responsive Design

The interface is designed for:

1. **Desktop monitors** (primary): Full-width layout with side-by-side panels for image and results.
2. **Tablets**: Single-column stacked layout.
3. **Mobile** (via Capacitor): Simplified layout with larger touch targets.

Tailwind's responsive prefixes handle this:

```tsx
<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
  <div className="col-span-1">{/* Image upload */}</div>
  <div className="col-span-1">{/* Results */}</div>
</div>
```

The next document (10) dives deep into the explainability pipeline — how Grad-CAM and feature importance maps are generated and displayed.
