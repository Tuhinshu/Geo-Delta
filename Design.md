# GeoDelta Design System & UI/UX Specifications
## Automated Geospatial Intelligence (GEOINT) Platform — SIH26227
**Target Environment:** Tactical Command Centers & Air-Gapped Defense Workstations  
**Design Aesthetic:** Next-Generation Tactical Military Dark Mode ("Command Center Glass")  
**Compliance Baseline:** [GeoDelta.md](file:///d:/Geo%20Delta%20-%20SIH/GeoDelta.md), [Knowledgebase.md](file:///d:/Geo%20Delta%20-%20SIH/Knowledgebase.md), [phases.md](file:///d:/Geo%20Delta%20-%20SIH/phases.md)

---

## 1. Design Philosophy & Tactical Aesthetic Vision

The GeoDelta user interface is engineered for high-stakes intelligence environments where rapid situational awareness, zero visual ambiguity, and continuous operator focus are mission-critical. The design language rejects generic corporate dashboards in favor of a specialized **Command Center Glass** aesthetic.

```|
+----------------------------------------------------------------------------------------------------+
|  GeoDelta  │ [LAT: 34.1234° N] [LON: 74.5678° E] [MGRS: 43R BK 12345 67890]                        |
+----------------------------------------------------------------------+-----------------------------+
|                                                                      | [OPERATIONAL DATA]          |
|   SPLIT-SWIPE GEOSPATIAL VIEWPORT                                    | ─────────────────────────── |
|   ================================                                   | ALTERED GROUND SURFACE      |
|   Left View: Pre-Event (t1)      │ Right View: Post-Event (t2)       | 18,450 m²  (1.845 ha)       |
|                                  │                                   | ─────────────────────────── |
|                                  │  [Detected Change Polygons]       | OBJECTS DETECTED: 7         |
|                                  │  (Glowing Crimson / Emerald)      | ─────────────────────────── |
|                                  │                                   | CONFIDENCE THRESHOLD (tau)  |
|                                  │                                   | [====O=========] 0.70       |
|                                  │                                   | ─────────────────────────── |
|                          [◀ ║ ▶]                                     | [ EXPORT(PDF) ]            |
|                      (Laser Divider)                                 |                             |
+----------------------------------------------------------------------+-----------------------------+
| [Query: "Identify newly paved airstrip extensions"] [Date Range] │ [RUN] [USER]                    |
+----------------------------------------------------------------------------------------------------+
```

### Core Design Principles
1. **High-Contrast Dark Ergonomics:** Designed for low-light defense briefing rooms and 24/7 watch centers. Uses deep obsidian/slate backgrounds with high-contrast tactical highlights, eliminating eye strain while prioritizing critical alerts.
2. **Zero-Ambiguity Information Hierarchy:** Visual weight is strictly calibrated to operational severity:
   - *Crimson / Amber:* Verified physical breaches, construction, and excavated footprints.
   - *Cyan / Reticle:* Active crosshair selections, spatial query boundaries, and sensor telemetry.
   - *Emerald:* Validated infrastructure baselines and cryptographic chain-of-custody proofs.
3. **Tactical Glassmorphism:** Subtle dark glass surfaces (`backdrop-filter: blur(16px)`) with fine, razor-sharp 1px borders (`#ffffff14`), giving floating control panels depth above high-resolution satellite imagery.
4. **Air-Gapped Self-Contained Assets:** Zero external CDN dependencies (no Google Fonts, FontAwesome, or public CDNs). All fonts, icons, and shader assets are bundled locally in the standalone build.

---

## 2. Color System & Design Tokens

The color architecture is built around an intentional, military-grade palette defined in HSL and Hex for complete design token consistency across CSS, Tailwind, MapLibre GL, and deck.gl shaders.

### 2.1 Base & Surface Tokens (Slate Gunmetal)
| Token Name | Hex Code | HSL Value | Purpose & Application |
| :--- | :--- | :--- | :--- |
| `--bg-void` | `#06080e` | `hsl(225, 40%, 4%)` | Root application background; deepest black underlayer |
| `--bg-surface` | `#0b0f19` | `hsl(223, 39%, 7%)` | Primary canvas & viewport framing background |
| `--bg-panel` | `#111827` | `hsl(220, 39%, 11%)` | Floating HUD panels, toolbars, and modal card surfaces |
| `--bg-panel-elevated` | `#162032` | `hsl(218, 38%, 14%)` | Hover states, active dropdowns, and popup cards |
| `--border-subtle` | `#1f2937` | `hsl(215, 28%, 17%)` | Primary 1px panel separation and grid boundaries |
| `--border-active` | `#374151` | `hsl(217, 19%, 27%)` | Focused input fields, active card outlines, and dividers |
| `--border-accent` | `#06b6d4` | `hsl(189, 94%, 43%)` | Selected polygon reticle, active tool focus borders |

### 2.2 Tactical Accents & Signal Indicators
| Token Name | Hex Code | HSL Value | Tactical Semantics |
| :--- | :--- | :--- | :--- |
| `--alert-crimson` | `#ef4444` | `hsl(0, 84%, 60%)` | Critical tactical change: graded roads, fortifications, trenching |
| `--alert-amber` | `#f59e0b` | `hsl(38, 92%, 50%)` | Moderate change / warning: seasonal boundary shifts, equipment movement |
| `--verified-emerald` | `#10b981` | `hsl(160, 84%, 39%)` | Verified baseline, cryptographic SHA-256 match, system healthy |
| `--reticle-cyan` | `#06b6d4` | `hsl(189, 94%, 43%)` | Operator crosshair, active AOI bounding box, laser split divider |
| `--command-indigo` | `#6366f1` | `hsl(239, 84%, 67%)` | Primary action triggers ("Run Analysis", "Compile Dossier") |

### 2.3 Text & Telemetry Contrast Hierarchy
| Token Name | Hex Code | HSL Value | Contrast Ratio vs `--bg-panel` | Application |
| :--- | :--- | :--- | :--- | :--- |
| `--text-primary` | `#f8fafc` | `hsl(210, 40%, 98%)` | **16.8 : 1** (AAA) | High-priority metrics, classification headers, active titles |
| `--text-secondary` | `#94a3b8` | `hsl(215, 20%, 65%)` | **7.4 : 1** (AAA) | Label subtitles, parameter descriptions, dates |
| `--text-muted` | `#64748b` | `hsl(215, 16%, 47%)` | **4.6 : 1** (AA) | Grid markings, disabled controls, placeholder text |
| `--text-mono-cyan` | `#67e8f9` | `hsl(186, 94%, 69%)` | **12.1 : 1** (AAA) | Coordinates (WGS84 / MGRS), areas ($m^2$), confidence scores |

### 2.4 Heatmap & Vector Probability Spectrum
For continuous probability heatmaps $P(Y=1) \in [0.0, 1.0]$, the color ramp transitions smoothly from transparent non-active regions to high-visibility tactical crimson:

```
0.00 ──────────── 0.30 ──────────── 0.50 ──────────── 0.75 ──────────── 1.00
Transparent      Sub-Threshold     Amber Notice      Tactical Alert   High-Confidence Breach
#00000000       #f59e0b33         #f59e0b99         #ef4444bf        #dc2626f2
```

---

## 3. Typography System & Type Scale

The typographic system utilizes a dual-font architecture: a high-readability geometric sans-serif for UI navigation and executive titles, paired with an engineered monospace font for telemetry, coordinates, and metrics.

### 3.1 Font Family Pairing
- **Primary Interface Font:** `Inter` (Local WOFF2)
  - *Weights:* Medium (`500`), Semi-Bold (`600`), Bold (`700`)
  - *Characteristics:* High x-height, clear letter differentiation under high pixel densities, clean rendering on dark backgrounds.
- **Technical & Telemetry Monospace Font:** `JetBrains Mono` (Local WOFF2)
  - *Weights:* Regular (`400`), Medium (`500`), Bold (`700`)
  - *Characteristics:* Strict tabular numerals (equal width for every digit), preventing visual jitter when coordinates or metrics update dynamically. Features distinctive zero slashing (`0`) to prevent confusion with `O`.

### 3.2 Typographic Hierarchy & Scale
| Level | Font Family | Size | Weight | Line Height | Letter Spacing | Example Text |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Banner Marker** | `JetBrains Mono` | `11px` | `700` | `16px` | `+0.15em` | `RESTRICTED // GEOINT ASSESSMENT` |
| **Display Metric** | `JetBrains Mono` | `28px` | `700` | `32px` | `-0.02em` | `18,450 m²` |
| **Panel Heading (H1)** | `Inter` | `18px` | `700` | `24px` | `-0.01em` | `OPERATIONS HUD` |
| **Section Title (H2)** | `Inter` | `14px` | `600` | `20px` | `0.00em` | `Altered Surface Footprint` |
| **Body Primary** | `Inter` | `13px` | `500` | `18px` | `0.00em` | `Identify newly graded airstrip extensions` |
| **Telemetry Value** | `JetBrains Mono` | `12px` | `500` | `16px` | `+0.05em` | `34.1234° N, 74.5678° E` |
| **Grid / MGRS Badge** | `JetBrains Mono` | `11px` | `700` | `14px` | `+0.10em` | `43R BK 12345 67890` |
| **Micro Caption** | `Inter` | `10px` | `500` | `14px` | `+0.05em` | `SENSOR: SENTINEL-2 L2A (10M GSD)` |

---

## 4. Elevation, Glassmorphism & Atmospheric Lighting

### 4.1 Layered Elevation Structure
The interface utilizes 5 discrete elevation planes to ensure floating inspection tools never visually blend with the underlying satellite raster:

```
[Level 4: Modal Inspection Dialogs & PDF Previews] (z-index: 1000)
    ▲
[Level 3: Polygon Click Inspector Popovers]        (z-index: 500)
    ▲
[Level 2: Floating HUD Panels, Sliders, & Toolbar] (z-index: 100)
    ▲
[Level 1: MapLibre Viewport Split Divider]         (z-index: 50)
    ▲
[Level 0: Geospatial Raster Canvas & deck.gl]      (z-index: 0)
```

### 4.2 Tactical Glassmorphism Specifications
Floating panels (Operations HUD, query bar, inspectors) implement a specialized dark glass treatment:

```css
.tactical-glass-panel {
  background: #111827d1; /* #111827 at 82% opacity */
  backdrop-filter: blur(16px) saturate(180%);
  -webkit-backdrop-filter: blur(16px) saturate(180%);
  border: 1px solid #ffffff14;
  box-shadow: 
    0 4px 6px -1px #00000080,
    0 10px 15px -3px #000000b3,
    inset 0 1px 0 0 #ffffff0f;
}

.tactical-glass-panel-elevated {
  background: #162032e6;
  backdrop-filter: blur(20px);
  border: 1px solid #06b6d440; /* Subtle Cyan glow border */
  box-shadow: 
    0 12px 24px -4px #000000cc,
    0 0 15px 0 #06b6d41f; /* Ambient cyan illumination */
}
```

---

## 5. Component Blueprint & Visual Specifications

### 5.1 Top Classification Ribbon & Command Header Bar
- **Classification Banner:**
  - Full-width top ribbon styled in tactical amber/gold for restricted access:
    - Background: `#f59e0b1f` with a 1px bottom border in `#f59e0b59`.
    - Text: `RESTRICTED // GEOINT ASSESSMENT // FOR OFFICIAL USE ONLY` (`11px`, `JetBrains Mono Bold`, color `#fbbf24`).
- **Command Search Bar:**
  - Unstructured natural language input field with rounded-lg (`8px`) corners.
  - Prefix icon: Tactical reticle / magnifying glass in `#06b6d4`.
  - Placeholder: `"Enter semantic tactical query (e.g. 'Identify newly paved runway or perimeter bunkers')..."`
  - Integrated **Negative Prompt Badge Pill:** Analysts can toggle an attached negative condition tag (e.g., `[ - NEG: Seasonal Soil Drying × ]`), styled in dark crimson glass (`#ef444426` with `#fca5a5` text).
- **Temporal Date Selector:**
  - Dual date pill displaying $t_1$ (Pre-Event) and $t_2$ (Post-Event) with sensor badge:
    `[ 15 JAN 2025 ] ➔ [ 10 JUN 2025 ] | S2-L2A (10m)`
- **"Execute Analysis" Action Trigger:**
  - High-impact button with gradient fill (`linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)`).
  - Active scanning state triggers a horizontal radar sweep animation across the button surface.

### 5.2 Synchronized Split-Swipe Viewport (MapLibre GL + deck.gl)
- **Visual Separation:**
  - Left Side: Pre-event optical raster ($t_1$) with subtle `"T1: JAN 2025"` watermark.
  - Right Side: Post-event optical raster ($t_2$) with matching `"T2: JUN 2025"` watermark.
- **Vertical Laser Split Divider:**
  - 2px wide vertical divider in glowing cyan (`#06b6d4`), illuminated with a lateral glow (`box-shadow: 0 0 8px #06b6d4`).
  - **Center Drag Handle:** A circular glass medallion (`36px` diameter) with left/right directional arrows (`◀ ║ ▶`).
  - Hover state: Expands to `42px` with increased bloom effect.
- **deck.gl Change Polygons:**
  - Staged rendering: Vector boundaries rendered continuously across the split slider.
  - **Stroke:** 2px high-visibility outline in `#ef4444`.
  - **Fill:** Semi-transparent red modulated by confidence $\tau$: `#ef4444` with dynamic alpha channel `4d` to `d9` (`#ef44444d` to `#ef4444d9`).
  - **Hover Bloom:** Hovering over any polygon intensifies fill to `#06b6d499` with cyan pulse.

### 5.3 Operations HUD & Metrics Rail (Right Sidebar)
- **Altered Surface Area Card:**
  - Huge monospace metric: `18,450 m²` with dynamic sub-label `(1.845 ha)`.
  - Micro-trend sparkline or comparison indicator against theater average.
- **Detection Summary Card:**
  - Object count badge: `7 Tactical Anomalies Verified`.
  - Breakdown chip pills: `4 Roads / Tracks`, `2 Fortifications`, `1 Clearing`.
- **Interactive Confidence Threshold Slider ($\tau$):**
  - Continuous slider track (`0.30` to `0.95`):
    - Track fill: Dynamic gradient (`#f59e0b` at 0.30 ➔ `#ef4444` at 0.70 ➔ `#10b981` at 0.90+).
    - Draggable thumb: Glowing white circle with cyan ring (`#06b6d4`).
    - Numeric readout: Real-time update in JetBrains Mono (`"tau = 0.70"`).
- **"Export Intelligence Dossier" Button:**
  - Full-width tactical button with crimson/steel border, download icon, and subtext `"1-Click Cryptographic PDF"`.

### 5.4 Polygon Click Inspector Card (Tactical Popover)
When an operator clicks a detected polygon, a high-contrast HUD popover anchors directly above the polygon centroid:

```
+--------------------------------------------------------+
| [TAC-04] UNPAVED ROADWAY EXTENSION         [CONF: 94%] |
| ────────────────────────────────────────────────────── |
|  CENTROID (WGS84): 34.1234° N, 74.5678° E      [COPY]  |
|  GRID (MGRS):      43R BK 12345 67890          [COPY]  |
|  GROUND FOOTPRINT: 14,250 m² (1.425 ha)                |
|  PERIMETER:        780 meters                          |
|  DELTA WINDOW:     15 JAN 2025 ➔ 10 JUN 2025           |
| ────────────────────────────────────────────────────── |
|  [ PREVIEW HIGH-RES CHIP ]     [ ADD TO DOSSIER QUEUE ]|
+--------------------------------------------------------+
```

### 5.5 Bottom Telemetry & Status Bar
- Fixed bottom strip (`28px` height) in deep void (`#06080e`):
  - **Left Section:** Real-time cursor coordinates in WGS 84 (`LAT: 34.1234° N | LON: 74.5678° E`) and MGRS (`43R BK 12345 67890`), plus elevation (`ELEV: 1,420m`).
  - **Center Section:** Active Celery job indicator with animated radar pulse: `PIPELINE: ECC ALIGNMENT (68%)`.
  - **Right Section:** Hardware telemetry badge (`GPU: 4.2 / 8.0 GB VRAM (52%)`) and WebSocket status beacon (solid green dot: `WSS: CONNECTED`).

---

## 6. Micro-Interactions, State Transitions & Keyframe Animations

Micro-animations are utilized strictly for functional telemetry and spatial orientation, avoiding gratuitous decorative movement.

### 6.1 Scanning Radar Sweep (`radar-sweep`)
Applied to active processing states and map loading overlays:
```css
@keyframes radar-sweep {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

.radar-sweep-indicator {
  position: relative;
  border-radius: 50%;
  border: 1px solid #06b6d44d;
}

.radar-sweep-indicator::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 50%;
  background: conic-gradient(from 0deg, #06b6d466 0deg, transparent 60deg);
  animation: radar-sweep 2.5s linear infinite;
}
```

### 6.2 Tactical Breach Pulse (`breach-pulse`)
Applied to newly identified high-confidence change polygons on initial render:
```css
@keyframes breach-pulse {
  0% {
    box-shadow: 0 0 0 0 #ef4444b3;
  }
  70% {
    box-shadow: 0 0 0 10px #ef444400;
  }
  100% {
    box-shadow: 0 0 0 0 #ef444400;
  }
}
```

### 6.3 Split-Swipe Scrubber Tracking
- The vertical laser divider tracks the mouse/touch horizontal coordinate with a buttery $60\text{ FPS}$ hardware-accelerated CSS `transform: translate3d(x, 0, 0)`.
- When dragged, the scissor rect clipping mask on the MapLibre canvas updates instantaneously with zero layout thrashing.

---

## 7. Cryptographic Intelligence Dossier PDF Visual Specifications

The exported PDF intelligence dossier must mirror official military documentation standards:

```
+----------------------------------------------------------------------------------------------------+
|                      RESTRICTED // GEOINT ASSESSMENT // FOR OFFICIAL USE ONLY                      |
+----------------------------------------------------------------------------------------------------+
|  OPERATION IDENTIFIER: OP-DELTA-2025-06-B                                ISSUE DATE: 2026-09-29    |
|  TARGET REGION: SECTOR 4-ALPHA BOUNDARY                            ANALYST ID: OPERATOR-ALPHA-7    |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  SECTION 1: SATELLITE BITEMPORAL VISUAL EVIDENCE CHIPS                                             |
|  ======================================================                                            |
|  +--------------------------+  +--------------------------+  +--------------------------+          |
|  | [PRE-EVENT (t1)]         |  | [POST-EVENT (t2)]        |  | [VECTOR CHANGE OVERLAY]  |          |
|  | Date: 15 JAN 2025        |  | Date: 10 JUN 2025        |  | High-Risk Breaches       |          |
|  | Sensor: Sentinel-2 L2A   |  | Sensor: Sentinel-2 L2A   |  | Red Outlines on t2       |          |
|  +--------------------------+  +--------------------------+  +--------------------------+          |
|                                                                                                    |
|  SECTION 2: QUANTITATIVE TACTICAL METRICS TABLE                                                    |
|  ===============================================                                                   |
|  | Feature ID | Tactical Class           | Area (m²) | Area (ha) | Centroid (WGS84) | Conf  |      |
|  |------------|--------------------------|-----------|-----------|------------------|-------|      |
|  | TAC-POL-01 | Unpaved Road Expansion   | 14,250 m² | 1.425 ha  | 34.1234, 74.5678 | 94.2% |      |
|  | TAC-POL-02 | Perimeter Revetment      |  3,200 m² | 0.320 ha  | 34.1250, 74.5690 | 88.5% |      |
|  | TAC-POL-03 | Graded Earth Berm        |  1,000 m² | 0.100 ha  | 34.1210, 74.5640 | 81.0% |      |
|                                                                                                    |
|  SECTION 3: CRYPTOGRAPHIC CHAIN OF CUSTODY (SHA-256)                                               |
|  ====================================================                                              |
|  Source Raster t1 SHA-256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855      |
|  Source Raster t2 SHA-256: 7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069      |
|  GeoJSON Feature SHA-256:  a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0      |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
|                      RESTRICTED // GEOINT ASSESSMENT // FOR OFFICIAL USE ONLY                      |
+----------------------------------------------------------------------------------------------------+
```

### PDF Layout Parameters (ReportLab / PyMuPDF)
- **Page Dimensions:** Standard A4 (`210mm \times 297mm`) in Portrait.
- **Margins:** $15\text{ mm}$ outer margins for high print density.
- **Header & Footer:** Solid black banner bar ($8\text{ mm}$ height) with white/gold centered bold text (`RESTRICTED // GEOINT ASSESSMENT`).
- **Color Space:** CMYK-safe equivalents for defense printing:
  - Tactical Crimson: `CMYK(0, 85, 75, 5)`
  - Tactical Cyan: `CMYK(80, 5, 10, 0)`
  - Deep Gunmetal: `CMYK(75, 65, 50, 60)`

---

## 8. Complete CSS Design System & Variables Block

The following CSS definitions form the foundation of `frontend/src/app/globals.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  /* Surface & Base Colors */
  --color-void: #06080e;
  --color-surface: #0b0f19;
  --color-panel: #111827;
  --color-panel-elevated: #162032;

  /* Borders */
  --border-subtle: #1f2937;
  --border-active: #374151;
  --border-accent: #06b6d4;

  /* Tactical Signals */
  --signal-crimson: #ef4444;
  --signal-amber: #f59e0b;
  --signal-emerald: #10b981;
  --signal-cyan: #06b6d4;
  --signal-indigo: #6366f1;

  /* Text Contrast Hierarchy */
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --text-mono-cyan: #67e8f9;

  /* Typography */
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

body {
  background-color: var(--color-void);
  color: var(--text-primary);
  font-family: var(--font-sans);
  overflow-x: hidden;
  user-select: none;
}

/* Monospace Telemetry Utilities */
.telemetry-text {
  font-family: var(--font-mono);
  font-feature-settings: 'tnum' on, 'zero' on;
}

/* Glassmorphism Classes */
.glass-panel {
  background: #111827d9;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border-subtle);
}

.glass-panel-elevated {
  background: #162032eb;
  backdrop-filter: blur(20px);
  border: 1px solid #06b6d459;
  box-shadow: 0 10px 25px -5px #000000cc, 0 0 15px #06b6d426;
}

/* Custom Tactical Scrollbars */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: var(--color-void);
}

::-webkit-scrollbar-thumb {
  background: var(--border-active);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--signal-cyan);
}
```

---

## 9. Tailwind CSS Configuration Blueprint

In `frontend/tailwind.config.ts`, these design tokens are mapped directly to Tailwind utility classes:

```typescript
import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        void: '#06080e',
        surface: '#0b0f19',
        panel: {
          DEFAULT: '#111827',
          elevated: '#162032',
        },
        tactical: {
          crimson: '#ef4444',
          amber: '#f59e0b',
          emerald: '#10b981',
          cyan: '#06b6d4',
          indigo: '#6366f1',
        },
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 15px #06b6d440',
        'glow-crimson': '0 0 15px #ef444459',
      },
    },
  },
  plugins: [],
};

export default config;
```

---

## 10. Visual Accessibility & Verification Checklist

| Aspect | Target Standard | Verification Method |
| :--- | :--- | :--- |
| **Contrast Ratio** | WCAG 2.1 AAA for text ($\ge 7.0 : 1$), AA for telemetry ($\ge 4.5 : 1$). | Automated color contrast checker on all token pairings. |
| **Colorblind Usability** | Protanopia / Deuteranopia safe: Alerts do not rely purely on Red/Green; accompanied by distinctive shape icons (⚠️ warning triangle, 🛡️ verified shield, 🎯 reticle crosshair). | Chrome DevTools Vision Deficiency simulation tests. |
| **Air-Gapped Assets** | 100% self-contained WOFF2 font files bundled in `frontend/public/fonts/`. Zero network requests to Google Fonts or Adobe Typekit. | Network tab offline inspection (`Offline` toggle in DevTools). |
| **Frame Rate** | $\ge 45\text{ FPS}$ during split-swipe scrubbing with 2,500 active vector polygons rendered via deck.gl. | Chrome DevTools FPS meter performance profiling. |
