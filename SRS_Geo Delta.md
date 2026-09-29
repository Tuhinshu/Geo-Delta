# Software Requirements Specification (SRS)
## Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
**Document Version:** 1.0.0  
**Target System:** Automated Geospatial Intelligence (GEOINT) Platform  
**Problem Statement ID:** SIH26227  
**Sponsoring Agency:** Ministry of Defence (MoD), Government of India  
**System Architecture Paradigm:** Hybrid Dual-Stream Cross-Attention Siamese Network (Tactical SIH Core) & Topological Scene Graph-Delta Space (Enterprise Scaling Roadmap)

---

## 1. Introduction & Executive Overview

### 1.1 Purpose
This Software Requirements Specification (SRS) defines the functional, technical, performance, and environmental requirements for the engineering, deployment, and validation of the automated GEOINT pipeline under Problem Statement SIH26227. This document establishes an unambiguous baseline for autonomous development agents, software architects, and machine learning engineers to construct the system and define subsequent Scope of Work (SOW) deliverables.

### 1.2 Problem Statement Identification
*   **Problem Statement ID:** SIH26227
*   **Title:** Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
*   **Organization:** Ministry of Defence (MoD)
*   **Category:** Software
*   **Primary Objective:** Ingest heterogeneous multi-temporal satellite rasters (optical multispectral/SAR) and allow non-GIS defense analysts to execute unstructured natural language queries (e.g., *"Identify unpaved road expansions and newly established perimeter revetments"*). The system must semantically condition the temporal difference space, generate sub-pixel coregistered change masks, suppress environmental/seasonal false alarms, compute ground surface deltas ($m^2$), and export vector GeoJSON alerts.

### 1.3 Scope of the Software
The software product operates as an on-premise, air-gapped web platform and inference pipeline comprising:
1.  A tokenized natural language processing pipeline translating defense intent into semantic feature vectors.
2.  A spatiotemporal indexing and retrieval interface operating over Cloud-Optimized GeoTIFFs (COGs).
3.  A bitemporal computer vision engine executing sub-pixel registration and Cross-Attention Dual-Stream Siamese segmentation.
4.  A post-processing vectorization engine that converts raster probability heatmaps into georeferenced spatial vectors (EPSG:4326).
5.  A dual-viewport synchronized analyst dashboard with dynamic split-swipe visualization and metric logging.
6.  An architectural graph extraction bridge laying the foundation for enterprise-scale Topological Scene Graph Delta analysis.

### 1.4 Definitions, Acronyms, and Abbreviations
*   **AOI:** Area of Interest.
*   **COG:** Cloud-Optimized GeoTIFF (tiled, multi-resolution raster supporting HTTP GET range requests).
*   **ECC:** Enhanced Correlation Coefficient Maximization (sub-pixel image alignment).
*   **GEOINT:** Geospatial Intelligence.
*   **GSD:** Ground Sample Distance (spatial resolution in meters per pixel).
*   **MGRS:** Military Grid Reference System.
*   **NIR / SWIR:** Near-Infrared / Short-Wave Infrared spectral bands.
*   **RBAC:** Role-Based Access Control.
*   **RemoteCLIP:** Contrastive Language-Image Pretraining fine-tuned on Remote Sensing datasets.
*   **STAC:** SpatioTemporal Asset Catalog.
*   **VLM:** Vision-Language Model.
*   **WGS 84:** World Geodetic System 1984 (EPSG:4326).

---

## 2. Overall Description & Operational Context

### 2.1 Problem Description & Operational Deficiencies
Current defense satellite workflows face severe operational bottlenecks:
*   **Semantic Disconnect:** Current search is strictly metadata-driven (sensor, timestamp, cloud-cover percentage). Operators cannot search for visual patterns or tactical entities using natural language.
*   **False-Positive Saturation:** Naive change detection algorithms (NDVI/pixel-differencing) register non-tactical seasonal shifts (vegetation growth, dry soil, sun-angle shadows, snowmelt) as anomalies, overwhelming defense analysts.
*   **Registration Misalignments:** Sensor trajectory variations create pixel misalignments across temporal acquisitions, yielding high-frequency edge artifacts along mountain ridges and coastlines.
*   **Computational Bottlenecks:** Re-running deep learning inference over hundreds of thousands of square kilometers of border terrain on every query wastes GPU compute.

### 2.2 Proposed Solution Overview
The system implements a phased operational paradigm:
*   **Tactical Core (Solution B - SIH Prototype):** A lightweight, high-precision deep learning engine. Natural language prompts are embedded via RemoteCLIP. Dual-date imagery ($I_{t_1}, I_{t_2}$) is processed through a weight-shared convolutional backbone (ResNet-34/ConvNeXt). At the bottleneck, a Cross-Attention module modulates the visual difference tensor against the language embedding, activating only pixels that match the semantic description and ignoring background noise. A dense UNet++ decoder outputs the target binary mask.
*   **Enterprise Scaling Roadmap (Solution D - Production):** An offline background pipeline converts static terrain rasters into Topological Scene Graphs ($G_{t_1}, G_{t_2}$) using foundation models (SAM-Geo). Changes are modeled as topological graph deltas ($\Delta G = G_{t_2} \ominus G_{t_1}$). Real-time searches execute over graph databases in milliseconds via Cypher queries before invoking localized GPU models.

### 2.3 User Classes and Personas
1.  **Field Analyst (Operator):** Issues queries in plain language, inspects localized AOIs, validates bounding polygons, adjusts confidence thresholds, and exports intelligence reports.
2.  **Strategic Command Officer (Viewer/Decision Maker):** Reviews theater-wide summaries, inspects quantitative area expansion metrics, and assesses strategic trends across multi-year temporal datasets.
3.  **System Administrator / Data Engineer (Admin):** Manages local COG ingestion, registers sensor profiles, manages air-gapped container weights, and monitors GPU worker queues.

### 2.4 Operational User Journeys

#### As-Is Workflow (Current Manual Model)
1.  Analyst receives report of suspected activity near an international boundary sector.
2.  Analyst manually searches image repositories for two cloud-free dates ($t_1, t_2$).
3.  Analyst downloads multi-gigabyte raw GeoTIFF files to a local workstation.
4.  Analyst performs manual split-screen panning or runs a threshold script generating thousands of false-positive vegetation changes.
5.  Analyst manually traces polygons using desktop GIS software.
6.  **Total Latency:** 4 to 8 hours per target sector.

#### To-Be Workflow (Target System)
1.  Analyst selects AOI on the interactive map and enters query: *"Detect newly cleared tracks or bunker revetments between Jan 2025 and June 2025."*
2.  FastAPI backend retrieves sub-window COG tiles via HTTP Range requests.
3.  Celery GPU worker performs ECC sub-pixel alignment, computes text-guided cross-attention Siamese inference, and polygonizes the mask.
4.  Frontend renders change polygons over the map in under 10 seconds, displaying surface footprint ($m^2$) and confidence metrics.
5.  Analyst reviews the result using the interactive split-swipe slider and clicks "Export Military Intelligence Dossier" (GeoJSON/PDF).
6.  **Total Latency:** < 15 seconds.

### 2.5 Operating Environment & Hardware Assumptions
*   **Air-Gapped Operation:** The platform must execute without outbound internet access. Basemaps, ML model weights, and tile servers must be hosted locally.
*   **Minimum Target Prototype Hardware:**
    *   Compute: 8-Core Intel/AMD x86_64 CPU, 32 GB RAM.
    *   GPU Acceleration: 1x NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (RTX 3060/4060 or T4/A10 cloud instance).
    *   Storage: 500 GB NVMe SSD (high read-throughput for COG window operations).
*   **Production Deployment Target:** Multi-node cluster with NVIDIA A100/H100 GPUs, MinIO distributed object storage, and PostgreSQL/PostGIS read-replicas.

---

## 3. System Architecture & Technical Specifications

### 3.1 Architectural Decomposition

[ FRONTEND LAYER: Next.js 14 + MapLibre GL + deck.gl ]
│
▼ (HTTPS / WebSockets)
[ API GATEWAY: FastAPI + Pydantic v2 + OAuth2/JWT RBAC ]
│
┌────────────────┴────────────────┐
▼                                 ▼
[ ASYNC TASK QUEUE ]            [ SPATIAL / VECTOR DB ]
Redis + Celery Workers          PostgreSQL 16 + PostGIS 3.4 + pgvector
│                                 │
▼                                 │
[ CORE INFERENCE ENGINE (Solution B) ]     │

Radiometric Prep & ECC Registration  │

RemoteCLIP Text Encoder (e_t)        │

Shared Siamese Backbone (F_t1, F_t2) │

Cross-Attention Semantic Bottleneck  │

Dense UNet++ Decoder                 │

Vectorizer (Shapes -> GeoJSON) ──────┘
│
▼ (Future Enterprise Extraction Bridge)
[ GRAPH SCALING PIPELINE (Solution D Roadmap) ]
Neo4j / PyG Topological Knowledge Graphs
│
▼
[ RASTER OBJECT STORAGE: MinIO / Local S3 ]
Cloud-Optimized GeoTIFFs (COGs) + TiTiler Dynamic XYZ Proxy


### 3.2 Solution B Pipeline (Cross-Attention Dual-Stream Siamese Engine)
1.  **Input Conditioning:** Two co-located temporal rasters $I_{t_1}, I_{t_2} \in \mathbb{R}^{C \times H \times W}$ ($C=4$: Red, Green, Blue, NIR) and natural language prompt $T$.
2.  **Radiometric Normalization & ECC Alignment:**
    *   Clip bands to 2nd and 98th percentiles: $I_{\text{norm}} = \frac{\text{clip}(I, P_2, P_{98}) - P_2}{P_{98} - P_2}$.
    *   Sub-pixel alignment: Maximize Enhanced Correlation Coefficient over panchromatic/luminance channel to compute warp matrix $W$:
        $$W^* = \arg\max_W \text{ECC}(I_{t_1}, \mathcal{W}(I_{t_2}; W))$$
3.  **Vision-Language Encoding:**
    *   Prompt $T$ is passed through RemoteCLIP Text Transformer to produce normalized vector $\mathbf{e}_t \in \mathbb{R}^{1 \times 512}$.
    *   Image pair is passed through weight-shared ResNet-34 branches, extracting feature hierarchies at stages $l \in \{1, 2, 3, 4\}$:
        $$\mathbf{F}_{t_1}^l, \mathbf{F}_{t_2}^l \in \mathbb{R}^{C_l \times H_l \times W_l}$$
4.  **Bitemporal Feature Difference & Fusion:**
    *   $$\mathbf{F}_{\Delta}^l = \text{Conv}_{1\times 1}\left(\left[\mathbf{F}_{t_1}^l \mathbin{\Vert} \mathbf{F}_{t_2}^l \mathbin{\Vert} \vert{}\mathbf{F}_{t_1}^l - \mathbf{F}_{t_2}^l\vert{}\right]\right)$$
5.  **Cross-Attention Semantic Bottleneck Modulation:**
    *   At stage $l=4$ ($C_4=512$), project visual difference into spatial queries: $Q = \mathbf{F}_{\Delta}^4 W_Q$ where $W_Q \in \mathbb{R}^{512 \times d_k}$.
    *   Project text embedding $\mathbf{e}_t$ into keys and values: $K = \mathbf{e}_t W_K, V = \mathbf{e}_t W_V$ where $W_K, W_V \in \mathbb{R}^{512 \times d_k}$.
    *   Compute modulated spatial representation:
        $$\mathbf{A} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
    *   Fused feature: $\mathbf{F}_{\text{fused}} = \mathbf{F}_{\Delta}^4 + \text{Proj}(\mathbf{A})$.
6.  **Decoder & Vector Mask Generation:**
    *   UNet++ decoder applies dense skip connections, restoring spatial resolution from $H/32 \times W/32$ up to $H \times W$.
    *   Sigmoid layer outputs continuous probability map $P(Y=1\vert{}I_{t_1}, I_{t_2}, T) \in [0, 1]^{H \times W}$.
    *   Map is binarized against threshold $\tau$ (default 0.70), processed via morphological opening ($3 \times 3$ structural element), and converted to vector polygons using `rasterio.features.shapes`.

### 3.3 Solution D Enterprise Scaling Roadmap (Topological Scene Graph Delta)
For border-wide surveillance ($> 50,000\text{ km}^2$), the system includes an offline background worker that converts raw imagery into topological scene graphs:
*   **Node Formulation ($V$):** Objects (buildings, airstrips, towers) detected via Segment Anything Geospatial (SAM-Geo) with attributes: `{id, type, centroid, area_m2, perimeter}`.
*   **Edge Formulation ($E$):** Spatial proximities and infrastructure networks computed via Delaunay Triangulation: `{source_id, target_id, relation_type: ['connected_by_road', 'adjacent_to'], distance_m}`.
*   **Topological Delta Formulation ($\Delta G$):**
    $$\Delta G = G_{t_2} \ominus G_{t_1} = (V_{t_2} \setminus V_{t_1}) \cup (E_{t_2} \setminus E_{t_1}) \cup \{\Delta \text{Attributes}\}$$
*   **Sub-millisecond Search:** Natural language is parsed into Cypher queries over Neo4j, isolating bounding boxes in $< 15\text{ ms}$ before selectively invoking the heavy Solution B neural network only on altered tiles.

### 3.4 Data Flow and Sequence
1.  Client issues `POST /api/v1/inference/analyze` with JSON payload (AOI bounding box, dates, query text).
2.  FastAPI logs job, issues Celery task ID, returns HTTP 202 Accepted.
3.  Client connects to WebSocket `/ws/v1/tasks/{task_id}` for streaming status.
4.  Worker requests windowed COG byte ranges from MinIO object storage.
5.  Worker runs ECC alignment $\rightarrow$ RemoteCLIP encoding $\rightarrow$ Siamese Cross-Attention forward pass.
6.  Worker converts probability array to vector GeoJSON, computes geodesic areas via PostGIS, and inserts results into PostgreSQL.
7.  Worker pushes completion payload over WebSocket; Frontend renders GeoJSON polygons on the MapLibre split-swipe viewport.

---

## 4. Pre-requisite Assets, Tech Stack & Download Matrix

### 4.1 Software & Framework Dependencies

| Layer | Component | Version | Installation Directive |
| :--- | :--- | :--- | :--- |
| **Runtime Environment** | Python | 3.11.x | Base interpreter |
| **Runtime Environment** | Node.js | 20.x LTS | Frontend build environment |
| **Deep Learning Framework**| PyTorch | $\ge 2.2.0$ (CUDA 12.1) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| **Vision-Language Engine** | Transformers / HuggingFace| $\ge 4.38.0$ | `pip install transformers accelerate` |
| **Geospatial Processing** | GDAL / Rasterio | GDAL 3.8+, Rasterio 1.3+ | `conda install -c conda-forge gdal rasterio` |
| **Spatial Vector Engine** | GeoPandas / Shapely | $\ge 0.14.0$ / $\ge 2.0.0$ | `pip install geopandas shapely` |
| **Computer Vision Core** | OpenCV (Headless) | $\ge 4.9.0$ | `pip install opencv-python-headless` |
| **Backend API Gateway** | FastAPI / Uvicorn | $\ge 0.110.0$ | `pip install fastapi uvicorn[standard] pydantic` |
| **Task Queue & Broker** | Celery / Redis | $\ge 5.3.6$ / $\ge 5.0.0$ | `pip install celery redis` |
| **Database Engines** | PostgreSQL / PostGIS | Postgre 16 / PostGIS 3.4| System APT / Docker image: `postgis/postgis:16-3.4` |
| **Tile Service** | TiTiler | $\ge 0.18.0$ | `pip install titiler.core uvicorn` |
| **Frontend Framework** | Next.js / TypeScript | Next 14.x / TS 5.x | `npx create-next-app@14 --typescript --tailwind` |
| **Geospatial UI Viewport** | MapLibre GL / deck.gl | $\ge 4.0.0$ / $\ge 9.0.0$ | `npm install maplibre-gl deck.gl @deck.gl/geo-layers` |

### 4.2 Pre-trained Weights & Models

1.  **RemoteCLIP (Domain VLM Text Encoder):**
    *   *Model Architecture:* ViT-B/32 or ResNet-50 trained on 800k remote sensing image-caption pairs.
    *   *Repository:* `https://github.com/Chen-Guanzhou/RemoteCLIP`
    *   *Weight Asset:* `RemoteCLIP-ViT-B-32.pt` (Hugging Face repository: `chendelong/RemoteCLIP`).
    *   *Local Storage Directory:* `/models/weights/remoteclip/`
2.  **Siamese Change Detection Backbone Baseline:**
    *   *Pre-trained Weights:* ResNet-34 ImageNet baseline (`torchvision.models.resnet34(weights='DEFAULT')`) combined with SNUNet-CD / LEVIR-CD pre-trained checkpoints.
    *   *Repository Reference:* `https://github.com/fitzp/SNUNet-CD`
    *   *Local Storage Directory:* `/models/weights/siamese_backbone/`
3.  **Segment Anything for Geospatial (SAM-Geo - Enterprise Scaling Roadmap):**
    *   *Model Architecture:* ViT-Base (`sam_vit_b_01ec64.pth`).
    *   *Repository:* `https://github.com/opengeos/segment-anything-geospatial`
    *   *Local Storage Directory:* `/models/weights/sam_geo/`

### 4.3 Datasets (Download Locations & Specifications)

1.  **LEVIR-CD (Large-scale Building Change Detection Dataset):**
    *   *Specifications:* 637 ultra-high-resolution ($0.5\text{ m/pixel}$) bitemporal patch pairs ($1024 \times 1024$), capturing structural growth, unpaved clearings, and construction.
    *   *Primary Host:* `https://justchenhao.github.io/LEVIR/`
    *   *Format:* Dual-date PNGs with binary ground-truth change masks.
2.  **WHU Building Change Detection Dataset:**
    *   *Specifications:* Aerial image pair ($0.2\text{ m}$ GSD) covering earthquake and post-disaster/construction expansion over $450\text{ km}^2$.
    *   *Primary Host:* `http://gpcv.whu.edu.cn/data/building_dataset.html`
    *   *Format:* Paired GeoTIFFs + raster change masks.
3.  **SpaceNet 7 Multi-Temporal Urban Development Challenge (Satellite Time-Series):**
    *   *Specifications:* Deep temporal stacks (24 monthly satellite observations) across 100 global locations ($4\text{ m}$ GSD imagery) tracking physical construction.
    *   *Access Point:* Radiant Earth Foundation / AWS Open Data (`s3://spacenet-dataset/spacenet/SN7_buildings/`)
    *   *Format:* GeoTIFF time series + matching geojson building footprints.
4.  **Sentinel-2 Level-2A (Cloud-Optimized Demonstration Tiles):**
    *   *Specifications:* 10m/20m multispectral operational imagery (B02, B03, B04, B08).
    *   *Access Point:* AWS Open Data Sentinel-2 COG Registry (`s3://sentinel-cogs/`) or Microsoft Planetary Computer STAC API (`https://planetarycomputer.microsoft.com/api/stac/v1`).

### 4.4 Local Infrastructure Requirements (Docker Compose Structure)
The platform must bootstrap via a single command: `docker compose up --build`. Containers include:
*   `db`: PostgreSQL 16 with PostGIS 3.4 extensions initialized.
*   `redis`: In-memory broker for Celery job queues.
*   `minio`: Local S3 object store holding sample COG files and STAC metadata.
*   `titiler`: Dynamic raster tile streaming engine reading directly from MinIO.
*   `api`: FastAPI gateway serving REST endpoints and managing task delegation.
*   `worker`: GPU-enabled PyTorch container executing the deep learning pipeline.
*   `web`: Next.js 14 server running the user interface.

---

## 5. Functional Requirements (FR)

### Module 1: Authentication & Role-Based Access Control (FR-AUTH)
*   `FR-AUTH-001`: The system shall enforce JSON Web Token (JWT) authentication over HTTPS/WSS for all operational endpoints.
*   `FR-AUTH-002`: The system shall support two roles: `ROLE_ANALYST` (view, query, export AOI) and `ROLE_COMMANDER` (configure data ingest, inspect system-wide audit trails, adjust system-wide confidence limits).
*   `FR-AUTH-003`: The system shall maintain an immutable, append-only audit trail in PostgreSQL recording every submitted text query, target AOI bounding coordinates, analyst identifier, and execution timestamp.

### Module 2: Natural Language Query & Embedding Ingestion (FR-NLQ)
*   `FR-NLQ-001`: The system shall provide an unstructured text search field accepting plain-text natural language queries up to 256 characters.
*   `FR-NLQ-002`: The system shall ingest the query and generate a unit-normalized 512-dimensional vector embedding $\mathbf{e}_t$ using the RemoteCLIP text transformer.
*   `FR-NLQ-003`: The system shall reject prompts consisting entirely of non-alphanumeric noise or SQL/Script injection attempts via Pydantic input sanitization.

### Module 3: Geospatial Raster Preprocessing & Alignment (FR-GEO)
*   `FR-GEO-001`: The system shall read Cloud-Optimized GeoTIFF (COG) rasters directly from local MinIO storage using HTTP Range GET requests, downloading only spatial windows bounding the user-defined AOI.
*   `FR-GEO-002`: The system shall automatically mask optical cloud cover using the Scene Classification Layer (SCL) or QA60 band, rejecting tiles with cloud contamination exceeding $35\%$ within the selected AOI.
*   `FR-GEO-003`: The system shall execute sub-pixel image registration on image pair ($I_{t_1}, I_{t_2}$) using Enhanced Correlation Coefficient (ECC) maximization over the panchromatic/luminance band before passing tensors to neural modules.

### Module 4: Semantic Cross-Attention Inference (FR-INF)
*   `FR-INF-001`: The system shall pass both temporal rasters through a weight-shared Siamese backbone (ResNet-34) to extract multi-scale feature hierarchies at 1/4, 1/8, 1/16, and 1/32 resolutions.
*   `FR-INF-002`: The system shall modulate the bottleneck visual difference tensor $\mathbf{F}_{\Delta}^4$ with the text embedding $\mathbf{e}_t$ through a multi-head Cross-Attention mechanism, where spatial features act as Queries ($Q$) and the language embedding acts as Keys ($K$) and Values ($V$).
*   `FR-INF-003`: The system shall process the cross-attention conditioned tensor through a UNet++ dense skip-connection decoder to produce a single-channel continuous probability heatmap ($P \in [0.0, 1.0]^{H \times W}$).
*   `FR-INF-004`: The system shall allow the operator to adjust the continuous classification threshold ($\tau \in [0.30, 0.95]$) with real-time recalculation of binary boundaries.

### Module 5: Morphological Filtering & Vectorization (FR-VEC)
*   `FR-VEC-001`: The system shall apply a $3 \times 3$ morphological opening filter on the binarized mask to eliminate single-pixel sensor noise and isolated false-positive edges.
*   `FR-VEC-002`: The system shall vectorize contiguous positive raster pixel regions into discrete polygon geometries using `rasterio.features.shapes`.
*   `FR-VEC-003`: The system shall calculate the ground surface footprint area in square meters ($m^2$) for each polygon using PostGIS ellipsoidal computations (`ST_Area(geom::geography)`).
*   `FR-VEC-004`: The system shall discard detected polygons with a ground surface footprint smaller than $50\text{ m}^2$ to suppress sub-tactical clutter.

### Module 6: Map Visualization & Dual-Viewport UI (FR-UI)
*   `FR-UI-001`: The system shall render an interactive geospatial viewport supporting pan, zoom, and rotate using MapLibre GL JS and deck.gl.
*   `FR-UI-002`: The system shall provide an interactive split-swipe slider control allowing operators to inspect pre-event imagery ($t_1$) on the left, post-event imagery ($t_2$) on the right, and the vector change overlay across both.
*   `FR-UI-003`: The system shall render detected change polygons as color-coded vector layers, where color intensity reflects model confidence ($[0.0, 1.0]$).
*   `FR-UI-004`: Clicking on any detected change polygon shall open an inspector panel displaying: Tactical Classification, Confidence Score, Centroid Coordinates (WGS84 & MGRS), and Total Altered Area ($m^2$).
*   `FR-UI-005`: The UI shall provide a one-click export generating an intelligence dossier containing a GeoJSON feature collection, cropped high-resolution before/after raster chips, and an analyst sign-off sheet (PDF format).

### Module 7: Topological Graph Delta Compilation (FR-GRAPH - Enterprise Roadmap)
*   `FR-GRAPH-001`: The system shall support an offline batch pipeline executing SAM-Geo instance segmentation across static imagery baselines to extract physical primitives (nodes) and transport connections (edges).
*   `FR-GRAPH-002`: The system shall support graph-level temporal difference operators ($\Delta G = G_{t_2} \ominus G_{t_1}$) persisted within a graph database (Neo4j).
*   `FR-GRAPH-003`: The system shall translate natural language structural relationship queries (e.g., *"Show structures built within 200 meters of existing airstrips"*) into structured Cypher queries executed across $\Delta G$.

---

## 6. External Interface Requirements

### 6.1 User Interfaces (UI Specifications)
*   **Theme:** Dark-mode tactical military theme (slate-gray base, high-contrast amber/emerald/crimson indicators for vector overlays).
*   **Layout:**
    *   *Top Bar:* Classification banner (`SECRET // NOFORN` simulation), Query Search Input, Temporal Date Range Selectors ($t_1, t_2$), Run Analysis Button.
    *   *Central Viewport:* MapLibre GL canvas with synchronized dual-raster split-swipe controller and vector layer rendering.
    *   *Right Sidebar:* Real-time analytics HUD (Total Area Changed, Total Objects Detected, Confidence Slider, Export Actions).
    *   *Bottom Status Bar:* System status, Celery queue load, GPU VRAM usage, Active Coordinate Readout (Lat/Lon/MGRS).

### 6.2 Hardware Interfaces
*   The software interacts directly with NVIDIA GPUs via CUDA 12.x drivers and PyTorch C++ bindings for hardware-accelerated tensor operations.
*   Standard physical interfaces: Mouse/Trackpad for geospatial map navigation; Keyboard for natural language query entry.

### 6.3 Software Interfaces
*   **PostgreSQL / PostGIS:** Interfaced via SQL over port `5432` using SQLAlchemy and asyncpg.
*   **MinIO Object Storage:** Interfaced via S3-compatible REST API over port `9000` using boto3 / minio-py.
*   **TiTiler Dynamic Tile Server:** Interfaced via HTTP REST calls on port `8001` providing XYZ web-mercator tiles to MapLibre GL.
*   **Redis Message Broker:** Interfaced via standard Redis protocol over port `6379`.

### 6.4 Communications Interfaces
*   **REST API:** JSON payloads over HTTP/1.1 and HTTP/2.
*   **WebSockets:** Full-duplex WebSocket connection (`/ws/v1/tasks/{task_id}`) for streaming asynchronous worker processing status and inference percentages.
*   **Map Tiles:** Standard Web Mercator XYZ vector and raster tile endpoints (`/tiles/{z}/{x}/{y}.png`).

---

## 7. Non-Functional Requirements (NFR)

### 7.1 Performance & Latency Benchmarks
*   `NFR-PERF-001`: The complete end-to-end inference pipeline for a standard $1024 \times 1024$ pixel tile at 10m GSD ($100\text{ km}^2$ ground footprint) shall complete within $\le 8.0\text{ seconds}$ on an NVIDIA RTX 3060/4060 GPU.
*   `NFR-PERF-002`: The RemoteCLIP text embedding generation for a 256-character query shall complete within $\le 150\text{ milliseconds}$.
*   `NFR-PERF-003`: Map tile rendering latency via TiTiler shall not exceed $250\text{ milliseconds}$ per raster tile under single-user conditions.
*   `NFR-PERF-004`: Vector polygon rendering on the frontend via MapLibre GL shall maintain $\ge 45\text{ frames per second}$ during active pan/zoom for scenes containing up to 2,500 polygons.

### 7.2 Safety, Fault Tolerance & Circuit Breakers
*   `NFR-SAFE-001 (Cloud Circuit Breaker):` If target AOI optical cloud contamination exceeds $35\%$, the inference engine shall reject the optical job, log a warning, and prompt the analyst to switch to Sentinel-1 SAR imagery.
*   `NFR-SAFE-002 (Alignment Abort):` If ECC sub-pixel alignment correlation score fails to exceed $0.65$ (indicating massive cloud shift or extreme topography variation), the system shall abort inference and alert the user rather than outputting misaligned edge false positives.
*   `NFR-SAFE-003 (GPU OOM Fallback):` If a `torch.cuda.OutOfMemoryError` is caught during cross-attention computation, the Celery worker shall automatically clear the CUDA cache, partition the AOI into 4 equal quadrants ($512 \times 512$ with $10\%$ overlap), process them sequentially, and stitch the resulting masks.

### 7.3 Security & Air-Gapped Compliance
*   `NFR-SEC-001`: Zero external API dependencies. The platform must not attempt outbound connections to external CDNs, cloud LLMs, or mapping services.
*   `NFR-SEC-002`: Data at Rest Encryption: All GeoTIFF archives stored in MinIO and vector records in PostgreSQL shall be encrypted using AES-256.
*   `NFR-SEC-003`: Supply Chain Integrity: Container startup scripts must verify the SHA-256 checksum of all PyTorch weight files (`.pt`, `.pth`, `.onnx`) before instantiating models into memory.

### 7.4 Software Quality Attributes
*   **Maintainability:** All Python code must strictly adhere to PEP 8 standards, pass `flake8` and `mypy` static typing checks, and achieve $\ge 80\%$ unit test coverage across spatial utility functions.
*   **Portability:** The entire application must be deployable across Linux distributions (Ubuntu 22.04 LTS certified) via standard Docker and Docker Compose definitions.

---

## 8. Verification & Validation Criteria (Acceptance Criteria)

### 8.1 Model Metrics

The Siamese Cross-Attention model must meet or exceed the following quantitative benchmarks when validated against the LEVIR-CD and WHU-CD holdout test sets:

$$\text{Precision} = \frac{TP}{TP + FP} \ge 0.88$$

$$\text{Recall} = \frac{TP}{TP + FN} \ge 0.84$$

$$\text{Intersection over Union (IoU)} = \frac{TP}{TP + FP + FN} \ge 0.78$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \ge 0.86$$

### 8.2 End-to-End System Integration Tests
1.  **Test Case TC-E2E-01 (Semantic Selectivity):**
    *   *Input:* Test pair with mixed changes (seasonal farmland drying + newly constructed military hangar).
    *   *Query:* *"Show newly constructed military hangar"*
    *   *Pass Criteria:* Output mask highlights hangar polygon; agricultural dry patches must show $0\%$ activation.
2.  **Test Case TC-E2E-02 (Air-Gapped Bootstrap):**
    *   *Condition:* Host network interface completely disconnected from the internet (`nmcli networking off`).
    *   *Execution:* Run `docker compose up`.
    *   *Pass Criteria:* Web interface loads, basemaps render from local vector tiles, and inference executes successfully using local model weights.
3.  **Test Case TC-E2E-03 (Split-Swipe Dynamic Scrubbing):**
    *   *Execution:* Analyst drags split slider from coordinate $X=0$ to $X=1920$.
    *   *Pass Criteria:* Visual sync between pre-event raster on the left and post-event raster on the right with zero pixel lag or projection mismatch.

### 8.3 Defense Operator Acceptance Checklist
*   [ ] Operator can define an AOI polygon using map drawing tools.
*   [ ] System executes natural language query in under 10 seconds.
*   [ ] Detected changes are rendered as vector polygons with ground area ($m^2$).
*   [ ] Cloud and seasonal changes are effectively suppressed.
*   [ ] Intelligence dossier can be downloaded as an industry-standard GeoJSON and cryptographic PDF.

---

## 9. Scope of Work (SOW) Alignment & Hackathon Boundaries

+---------------------------------------------------------------------------------------+
|                               HACKATHON MUST-BUILD BOUNDARY                           |
|                                                                                       |
|   IN-SCOPE (SIH Working Prototype)           OUT-OF-SCOPE (Do NOT Attempt for SIH)    |
|   --------------------------------           -------------------------------------    |
|   1. Solution B Siamese Cross-Attention      1. Real-time military UAV video stream   |
|   2. RemoteCLIP Text Encoder integration     2. Sub-meter tactical object recognition |
|   3. FastAPI + Celery + Redis worker         3. Full national border deployment       |
|   4. PostgreSQL + PostGIS vector store       4. Physical satellite ground station     |
|   5. Next.js 14 MapLibre split-swipe UI         telemetry interfaces                  |
|   6. Cloud-Optimized GeoTIFF streaming                                                |
|   7. LEVIR-CD / Sentinel-2 test scenes                                                |
|                                                                                       |
+---------------------------------------------------------------------------------------+
│
▼
+---------------------------------------------------------------------------------------+
|                       ENTERPRISE PRODUCTION ROADMAP (Judges Q&A)                      |
|                                                                                       |
|   1. Topological Scene Graph Delta Compilation (Solution D)                           |
|   2. SAM-Geo terrain primitive pre-vectorization                                      |
|   3. Neo4j graph database hosting national border graph topology                     |
|   4. Sub-millisecond Cypher query execution bypassing raster GPU inference           |
+---------------------------------------------------------------------------------------+


### 9.1 In-Scope: The SIH Hackathon Working Prototype
The engineering team must construct and present a working end-to-end prototype strictly bounded by:
1.  **AI Engine:** Solution B architecture (Weight-shared ResNet-34 Siamese backbone + RemoteCLIP ViT-B/32 text projection + Cross-Attention bottleneck + UNet++ decoder).
2.  **Imagery Data:** Open-source remote sensing data (Sentinel-2 L2A optical 10m bands and LEVIR-CD building change benchmark pairs).
3.  **Core Services:** Docker Compose orchestration hosting FastAPI, Celery, Redis, PostgreSQL/PostGIS, and Next.js 14.
4.  **UI/UX:** Synchronized split-screen MapLibre GL JS viewer with interactive swipe slider, natural language query bar, and GeoJSON polygon visualizer.
5.  **Metrics:** Dynamic calculation of model confidence and ground footprint area ($m^2$).

### 9.2 Enterprise Scalability Scope (Production Roadmap for Judging Defense)
These elements are strictly reserved as an architectural roadmap to defend national-scale performance during judging Q&A:
1.  Full-scale offline raster compilation into **Topological Scene Graphs (Solution D)** using SAM-Geo.
2.  Deployment of distributed Neo4j cluster running Cypher topological path traversals over $\Delta G$.
3.  Multi-region Kubernetes deployment scaling GPU workers over thousands of square kilometers.

### 9.3 Strict Out-of-Scope Boundaries (Failure Prevention)
To ensure hackathon completion, the team shall **NOT**:
1.  Attempt to train a foundational Vision-Language Model from scratch (utilize frozen RemoteCLIP weights with fine-tuned cross-attention adapters).
2.  Attempt live real-time full-motion video (FMV) streaming from drones/UAVs.
3.  Attempt classified military hardware cryptographic hardening.
4.  Attempt integration with live ISRO/DRDO satellite ground station telemetry uplinks.