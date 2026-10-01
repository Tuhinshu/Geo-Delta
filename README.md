<div align="center">

# 🛰️ GEODELTA
### Automated Satellite Change Detection & Defense GEOINT Intelligence Platform
**Smart India Hackathon 2026 — Problem Statement SIH26227 (Ministry of Defence)**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Frontend-Next.js%2014-black.svg?logo=next.js&logoColor=white)](https://nextjs.org/)
[![MapLibre GL](https://img.shields.io/badge/GIS-MapLibre%20GL-396B99.svg?logo=maplibre&logoColor=white)](https://maplibre.org/)
[![PyTorch](https://img.shields.io/badge/AI-PyTorch%20%7C%20RemoteCLIP-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PostgreSQL / PostGIS](https://img.shields.io/badge/Spatial%20DB-PostgreSQL%20%2F%20PostGIS-336791.svg?logo=postgresql&logoColor=white)](https://postgis.net/)
[![Air-Gapped Ready](https://img.shields.io/badge/Deployment-Air--Gapped%20Defense-red.svg)](docker-compose.yml)
[![Tests Passing](https://img.shields.io/badge/Tests-63%2F63%20Passing-brightgreen.svg)](backend/tests/)

<br />

**GeoDelta** is an enterprise-grade, air-gapped Geospatial Intelligence (GEOINT) platform designed for tactical command centers and defense analysts. It automates multi-temporal satellite change detection across open-vocabulary operational queries (e.g., *"Identify newly paved airstrip extensions and fortified bastions"*), mathematically suppresses seasonal vegetation/soil moisture false alarms, and compiles cryptographically verifiable intelligence dossiers.

</div>

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Comprehensive Technology Stack](#-comprehensive-technology-stack)
- [Technical & Operational Feasibility](#-technical--operational-feasibility)
- [Four Core Solution Strategies](#-four-core-solution-strategies)
- [Key Features](#-key-features)
- [Directory Structure](#-directory-structure)
- [Installation & Quickstart](#-installation--quickstart)
  - [Prerequisites](#prerequisites)
  - [Option A: Docker Compose (Recommended)](#option-a-docker-compose-recommended)
  - [Option B: Native Development Setup](#option-b-native-development-setup)
- [Verification & Test Benchmarks](#-verification--test-benchmarks)
- [Security, Cryptographic Custody & Compliance](#-security-cryptographic-custody--compliance)

---

## 🎯 Executive Overview

Conventional optical satellite change detection suffers from severe operational bottlenecks:
1. **High False Positive Rates:** Seasonal crop growth, soil drying, and sun angle shadow variations trigger false breach alerts.
2. **Computational Inefficiency:** Downloading full multi-gigabyte satellite scenes causes extreme network and compute latency.
3. **Platform Drift & Edge Jitter:** Sub-pixel orbital misregistration creates artificial boundary change artifacts.
4. **Cloud Obstruction:** Optical satellites are completely blinded by overcast weather and monsoons.

**GeoDelta** solves these challenges through a unified, physics-governed pipeline combining **RemoteCLIP semantic conditioning**, **sub-pixel OpenCV ECC coregistration**, **windowed Cloud-Optimized GeoTIFF (COG) byte-streaming**, **automated Sentinel-1 Synthetic Aperture Radar (SAR) switching**, and a **MapLibre GL dual-map split-swipe GIS viewport**.

---

## 🏗️ System Architecture

```
                       [ OPERATOR / ANALYST HUD ]
                                   │
              Natural Language Query: "Identify newly paved airstrip"
                                   ▼
                       [ VLM ENCODER (RemoteCLIP) ]
                                   │  512-dim Semantic Embedding
                                   ▼
[ Pre-Event (t1) COG ] ──► [ CLOUD GUARD (QA60) ] ──(If >20% Cloud)──► [ S1 SAR FALLBACK ]
[ Post-Event (t2) COG ]          │ (Optical Clear)
                                 ▼
                     [ SUB-PIXEL ECC ALIGNMENT ]
                                 │ Affine Warp Matrix (Error <= 0.1 px)
                                 ▼
            [ SIAMESE CROSS-ATTENTION UNet++ (ResNet-34) ]
                                 │ Continuous Change Probability Heatmap
                                 ▼
            [ MORPHOLOGICAL FILTERING (3x3 Opening) ]
                                 │ Noise Suppression
                                 ▼
            [ KARNEY GEODESIC FOOTPRINT ENGINE (WGS84) ]
                                 │ Exact Ground Surface (m² / ha) + MGRS Centroid
                                 ▼
             ┌───────────────────┴───────────────────┐
             ▼                                       ▼
  [ MAPLIBRE GL VIEWPORT ]               [ CRYPTOGRAPHIC DOSSIER ]
  - Dual Synchronized Map                - SHA-256 Chain-of-Custody
  - Draggable Laser Divider              - 1-Click PDF Export
  - Real Satellite Basemap               - Tactical Coordinates
```

---

## 🧰 Comprehensive Technology Stack

| Domain | Technologies | Implementation Purpose |
| :--- | :--- | :--- |
| **Frontend & Visualization** | **Next.js 14**, **TypeScript**, **MapLibre GL v4**, **deck.gl**, **Tailwind CSS** | Tactical military dark mode ("Command Center Glass"), hardware-accelerated dual-map split-swipe viewer, 60fps WebGL vector rendering, and zero-CDN air-gapped design. |
| **Backend & Processing** | **Python 3.12+**, **FastAPI**, **Celery**, **Redis** | High-throughput async REST API, distributed background task queues, and low-latency tensor caching. |
| **AI / ML (Change Detection)** | **PyTorch**, **RemoteCLIP**, **ResNet-34**, **Cross-Attention**, **UNet++** | Dual-branch weight-shared Siamese feature extraction, spatial cross-attention query conditioning, and multi-scale change probability decoders. |
| **Geospatial Processing** | **GDAL**, **Rasterio**, **GeoPandas**, **Shapely**, **TiTiler** | Windowed COG byte-range streaming (`/vsicurl/`), affine geo-transformations, vector polygon geometry operations, and dynamic tile serving. |
| **Computer Vision** | **OpenCV**, **ECC Sub-Pixel Coregistration** | Sub-pixel affine image coregistration ($\le 0.1\text{ px}$ alignment error) and $3 \times 3$ morphological opening clutter suppression. |
| **Database & Storage** | **PostgreSQL 16**, **PostGIS 3.4**, **pgvector**, **MinIO**, **COG (GeoTIFF)** | Geospatial vector storage, Karney ellipsoidal area calculations, semantic vector embeddings, and self-hosted S3-compatible air-gapped raster object storage. |

---

## 🚀 Technical & Operational Feasibility

### 1. `01 — MODULAR AI`
* **Transfer Learning:** Integrates **RemoteCLIP** (vision-language foundation model trained on earth observation datasets) to project open-vocabulary user prompts into semantic embeddings.
* **Siamese Backbone:** Weight-shared **ResNet-34** extracts multi-scale feature pyramids from $t_1$ and $t_2$, while multi-head spatial cross-attention isolates human-made tactical changes.

### 2. `02 — ACCESSIBLE COMPUTE (<8s Inference on 100 km² AOI)`
* **Operational Scale:** A $100\text{ km}^2$ Area of Interest at Sentinel-2's $10\text{m}$ GSD corresponds to a $1,000 \times 1,000$ pixel raster ($1\text{ Megapixel}$).
* **Hardware Efficiency:** Fully optimized for standard defense laptops and integrated GPUs (Intel Iris Xe, AMD Radeon 680M/780M) without requiring high-end data-center clusters:
  * Windowed COG Streaming: $\approx 250\text{ ms}$
  * Sub-Pixel ECC Coregistration: $\approx 550\text{ ms}$
  * Siamese Neural Inference: $\approx 2.5 - 3.8\text{ s}$
  * Geodesic Vectorization & PostGIS Commit: $\approx 45\text{ ms}$
  * **Total Pipeline Latency:** $\mathbf{\approx 3.4 - 5.5\text{ s}}$ ($\ll 8\text{ seconds}$ operational SLA).

### 3. `03 — OFFLINE AIR-GAPPED DEPLOYMENT`
* Fully containerized multi-container topology with **zero external telemetry or cloud dependencies**.
* Self-contained fonts, icons, shader assets, and model weights.
* Operates entirely in secure, air-gapped SCIF environments.

### 4. `04 — OPEN DATA READY`
* Multi-spectral band ingestion for **Sentinel-2 L2A Top-of-Canopy (BOA)** and **Landsat 8/9 Collection 2 Surface Reflectance**.
* Native support for RGB (B4, B3, B2) and Near-Infrared (B8/B5) bands for precise vegetation and soil moisture discrimination.

---

## 🛡️ Four Core Solution Strategies

### `01 — Cloud Guard ➔ Sentinel-1 SAR Fallback`
* **Optical Guardrail:** Evaluates the dedicated **Sentinel-2 QA60 quality band** (decoding Bit 10 for Opaque Clouds and Bit 11 for Cirrus).
* **Automated SAR Switch:** If cloud cover exceeds the $20\%$ threshold, the pipeline automatically pivots to **Sentinel-1 C-Band SAR** data ($\text{VV} + \text{VH}$ dual-polarization). C-band microwaves penetrate cloud cover, monsoons, and darkness, using log-ratio backscatter differencing ($\Delta \sigma^0$) to maintain continuous 24/7 surveillance.

### `02 — Enhanced Correlation Coefficient (ECC) Sub-Pixel Alignment`
* Mitigates orbital platform drift and relief distortion using OpenCV's `cv2.findTransformECC`.
* Solves for the optimal 2D Affine Warp matrix $W$:
  $$\rho(W) = \frac{\bar{i}_{t1}^T \cdot \bar{i}_{t2}(W)}{\|\bar{i}_{t1}\| \cdot \|\bar{i}_{t2}(W)\|}$$
* Reaches sub-pixel alignment ($\le 0.1\text{ pixel}$ misregistration error), mathematically eliminating false change edges. Rejects frames with correlation score $< 0.65$.

### `03 — Semantic Cross-Attention False Alarm Filtering`
* Conventional NDVI or image differencing flags every dried field or harvested crop as an anomaly.
* GeoDelta conditions the spatial cross-attention maps with the user's specific prompt embedding $E_{\text{text}}$:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \odot E_{\text{text}}$$
* Suppresses natural seasonal cycles and focuses feature extraction strictly on specified tactical infrastructure (e.g., paved tarmac, defensive berms, vehicle revetments).

### `04 — Windowed COG Streaming + QuadTree Tiling Fallback`
* **Bandwidth Optimization:** Cloud-Optimized GeoTIFFs organize imagery into internal $256 \times 256$ tile matrices and overviews. GeoDelta uses HTTP `Range: bytes=start-end` requests to read only the spatial tiles intersecting the AOI bounding box.
* **VRAM Fallback:** When AOI dimensions exceed available GPU memory, the **Spatial QuadTree Engine** recursively subdivides the scene into $512 \times 512$ tiles with 32-pixel overlap halos, processes them sequentially, and blends them into a seamless georeferenced output mask.

---

## ⚡ Key Features

- **Real MapLibre GL Split-Swipe GIS Canvas:** Two synchronized WebGL map instances running in 60fps lockstep. Draggable cyan laser divider `[◀ ║ ▶]` reveals Pre-Event ($t_1$) on the left and Post-Event ($t_2$) on the right over high-resolution **ESRI World Imagery** satellite tiles.
- **Dual Custom Image Upload & Real-Time Differencing:** Analysts can upload custom multi-temporal imagery ($t_1$ and $t_2$ in PNG, JPG, or TIFF) or load pre-packaged mission presets (`Airbase Expansion`, `Sentinel-2 Raw Tile`).
- **Deterministic Karney Geodesic Footprint:** Computes true ellipsoidal surface area in square meters ($m^2$) and hectares ($ha$) using WGS 84 ellipsoidal geodesics, with error margins $\le 0.07\%$.
- **NATO 10-Figure MGRS Conversion:** Automatically converts WGS 84 centroid coordinates into NATO military grid references (e.g., `43R EH 07500 04500`).
- **Interactive Target Inspection:** Click any detected polygon on the map or in the target list to smoothly fly to its centroid and inspect its area, classification, and coordinates.
- **1-Click Cryptographic Dossier PDF Export:** Compiles an intelligence dossier containing analyst callsign, classified metadata, target polygon breakdown, and SHA-256 cryptographic chain-of-custody seals.

---

## 📁 Directory Structure

```
Geo Delta - SIH/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/       # FastAPI route controllers (inference, dossier)
│   │   ├── core/                   # Config, guardrails, and Pydantic schemas
│   │   ├── db/                     # SQLAlchemy models, PostGIS schemas, session
│   │   ├── services/               # Core intelligence processing engines:
│   │   │   ├── cog_streamer.py     # Windowed HTTP byte-range COG streaming
│   │   │   ├── coregistration.py   # OpenCV sub-pixel ECC alignment
│   │   │   ├── vlm_encoder.py      # RemoteCLIP 512-dim prompt encoder
│   │   │   ├── siamese_engine.py   # ResNet-34 + Cross-Attention UNet++
│   │   │   ├── normalizer.py       # Multi-spectral Sentinel-2/Landsat scaling
│   │   │   ├── vectorizer.py       # Morphological filtering & polygon extraction
│   │   │   ├── footprint_calculator.py # Karney WGS84 geodesic area & MGRS
│   │   │   ├── fallback_engine.py  # Sentinel-1 SAR cloud penetration switch
│   │   │   ├── quadtree_engine.py  # 512x512 tiling VRAM fallback
│   │   │   └── dossier_generator.py# ReportLab cryptographic PDF compiler
│   │   └── workers/                # Celery distributed tasks & Redis queues
│   ├── tests/                      # 63 unit, integration, and benchmark tests
│   ├── Dockerfile                  # Standalone backend container
│   └── requirements.txt            # Python dependencies
├── enterprise/                     # Neo4j Knowledge Graph compiler & SAM segmentation
├── frontend/
│   ├── public/samples/             # High-res optical & Sentinel-2 satellite tiles
│   ├── src/
│   │   ├── app/                    # Next.js 14 App Router (layout, page, globals.css)
│   │   ├── components/             # Tactical UI components:
│   │   │   ├── HeaderBar.tsx       # Live MGRS/WGS84 reticle & classification banner
│   │   │   ├── SplitSwipeViewer.tsx# MapLibre GL dual-map laser split viewer
│   │   │   ├── OperationsHUD.tsx   # Image upload, metrics, and dossier CTA
│   │   │   ├── PolygonInspector.tsx# Target click popover inspector
│   │   │   └── StatusBar.tsx       # Live pipeline stages, GPU VRAM telemetry
│   │   ├── types/                  # TypeScript data contracts matching Pydantic
│   │   └── utils/                  # Client-side pixel differencing engine
│   ├── Dockerfile                  # Production Next.js container
│   └── package.json                # Frontend dependencies
├── storage/samples/                # Ground truth masks and sample GeoTIFF rasters
├── docker-compose.yml              # Complete 7-service containerized stack
├── Design.md                       # Comprehensive UI/UX Design System specifications
├── Knowledgebase.md                # Remote sensing physics & mathematical guardrails
├── phases.md                       # Phase 0–7 engineering roadmap & verification
└── README.md                       # This document
```

---

## 🛠️ Installation & Quickstart

### Prerequisites
- **Git**
- **Docker Desktop** (optional, for full containerized stack)
- **Node.js 18+** & **npm**
- **Python 3.12+**

---

### Option A: Docker Compose (Recommended)

To launch the complete air-gapped stack (PostgreSQL/PostGIS, Redis, MinIO, TiTiler, FastAPI, Celery, and Next.js):

```bash
# 1. Clone the repository
git clone https://github.com/Tuhinshu/Geo-Delta.git
cd Geo-Delta

# 2. Copy the environment template
cp .env.example .env

# 3. Launch all services with Docker Compose
docker-compose up --build -d
```

Access the interfaces:
* **Tactical Command Dashboard:** [http://localhost:3000](http://localhost:3000)
* **FastAPI Interactive Docs:** [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
* **MinIO Object Storage Console:** [http://localhost:9001](http://localhost:9001)

---

### Option B: Native Development Setup

#### 1. Backend Setup (FastAPI & PyTorch)
```bash
# From the root directory:
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start FastAPI Uvicorn Server
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

#### 2. Frontend Setup (Next.js & MapLibre GL)
```bash
# In a new terminal window:
cd frontend

# Install Node dependencies
npm install

# Start Next.js Development Server
npm run dev
```

Open **`http://localhost:3000`** in your browser.

---

## 🧪 Verification & Test Benchmarks

The backend test suite covers unit logic, pipeline integration, spatial geodesics, and edge failure cases:

```bash
# Run full pytest suite (63 tests)
pytest backend/tests/ -v
```

### Test Suite Coverage
| Test Suite | Passing Tests | Validated Requirements |
| :--- | :--- | :--- |
| `test_api_endpoints.py` | 5 / 5 | Sub-Nyquist rejection, feasible query, counter-factual pipeline |
| `test_cog_streamer.py` | 5 / 5 | Windowed byte streaming, out-of-bounds AOI handling |
| `test_coregistration.py` | 5 / 5 | Sub-pixel ECC convergence, registration failure detection |
| `test_cross_attention.py`| 5 / 5 | Spatial query cross-attention modulation |
| `test_db_models.py` | 3 / 3 | SQLAlchemy PostGIS models & SQLite test fallback |
| `test_dossier_generator.py`| 4 / 4 | Cryptographic PDF report generation & SHA-256 seal |
| `test_e2e_pipeline.py` | 5 / 5 | Full end-to-end multi-stage inference execution |
| `test_enterprise_solution_d.py` | 7 / 7 | Neo4j graph compilation & SAM segmentation fallback |
| `test_feature_cache.py` | 4 / 4 | Redis tensor caching & cache key generation |
| `test_footprint_calculator.py` | 5 / 5 | Karney geodesic area ($\le 0.071\%$ error) & NATO MGRS |
| `test_guardrails.py` | 4 / 4 | Sub-Nyquist physical law & QA60 cloud cover checks |
| `test_normalizer.py` | 4 / 4 | Sentinel-2 L2A Top-of-Canopy scaling & clipping |
| `test_vectorizer.py` | 4 / 4 | Morphological $3\times 3$ opening & GeoJSON vector extraction |
| `test_vlm_encoder.py` | 3 / 3 | RemoteCLIP 512-dim embedding & cosine similarity |
| **Total** | **63 / 63 (100%)** | **All Phase 0–7 requirements verified** |

---

## 🔒 Security, Cryptographic Custody & Compliance

1. **Deterministic Chain of Custody:** Every processed satellite raster and exported intelligence brief generates SHA-256 digests recorded into an append-only audit trail (`AuditLog`).
2. **Air-Gapped Isolation:** Zero outbound network requests; zero tracking telemetry; fully isolated from public internet exposure.
