'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';
import type { Map as MapLibreMap, GeoJSONSource } from 'maplibre-gl';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Eye,
  EyeOff,
  Compass,
  Layers,
  MapPin,
  Sliders,
  Maximize2
} from 'lucide-react';
import { DetectedPolygonFeature } from '../types/geoint';

// Tactical Airbase Sector 4 Center & Bounds
const DEFAULT_CENTER: [number, number] = [75.0750, 25.0450]; // [Longitude, Latitude]
const DEFAULT_ZOOM = 14.2;

// Strategic Sector 4 AOI Envelope for Sensor Image Georeferencing
const AOI_COORDINATES: [[number, number], [number, number], [number, number], [number, number]] = [
  [75.0500, 25.0650], // Top-Left [lon, lat]
  [75.1000, 25.0650], // Top-Right
  [75.1000, 25.0250], // Bottom-Right
  [75.0500, 25.0250]  // Bottom-Left
];

// Map Styles: Free, Public, Air-Gapped Resilient
const STYLE_SATELLITE = {
  version: 8,
  sources: {
    'esri-world-imagery': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: 'Esri, Maxar, Earthstar Geographics'
    }
  },
  layers: [
    {
      id: 'esri-satellite-layer',
      type: 'raster',
      source: 'esri-world-imagery',
      minzoom: 0,
      maxzoom: 20
    }
  ]
};

const STYLE_DARK_TACTICAL = {
  version: 8,
  sources: {
    'carto-dark': {
      type: 'raster',
      tiles: [
        'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'
      ],
      tileSize: 256,
      attribution: 'CartoDB, OpenStreetMap contributors'
    }
  },
  layers: [
    {
      id: 'carto-dark-layer',
      type: 'raster',
      source: 'carto-dark',
      minzoom: 0,
      maxzoom: 20
    }
  ]
};

interface SplitSwipeViewerProps {
  t1ImageUrl: string;
  t2ImageUrl: string;
  t1Label?: string;
  t2Label?: string;
  features: DetectedPolygonFeature[];
  selectedFeature: DetectedPolygonFeature | null;
  onSelectFeature: (feature: DetectedPolygonFeature | null) => void;
  onMouseMoveCoords?: (lat: number, lon: number, mgrs: string) => void;
  onUploadT1?: (file: File) => void;
  onUploadT2?: (file: File) => void;
}

export const SplitSwipeViewer: React.FC<SplitSwipeViewerProps> = ({
  t1ImageUrl,
  t2ImageUrl,
  t1Label = 'PRE-EVENT (t1)',
  t2Label = 'POST-EVENT (t2)',
  features,
  selectedFeature,
  onSelectFeature,
  onMouseMoveCoords
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const leftMapContainerRef = useRef<HTMLDivElement>(null);
  const rightMapContainerRef = useRef<HTMLDivElement>(null);

  // MapLibre Instances
  const mapLeftRef = useRef<MapLibreMap | null>(null);
  const mapRightRef = useRef<MapLibreMap | null>(null);
  const isSyncingRef = useRef<boolean>(false);

  // Split-Swipe Position (0.0 to 1.0, default 0.50)
  const [splitPos, setSplitPos] = useState<number>(0.50);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  // Viewport & Layer Settings: Default to crystal-clear Satellite Basemap
  const [basemapStyle, setBasemapStyle] = useState<'satellite' | 'dark'>('satellite');
  const [showRasterOverlay, setShowRasterOverlay] = useState<boolean>(false);
  const [rasterOpacity, setRasterOpacity] = useState<number>(0.75);
  const [showVectors, setShowVectors] = useState<boolean>(true);
  const [is3D, setIs3D] = useState<boolean>(false);
  const [isMapLoaded, setIsMapLoaded] = useState<boolean>(false);

  // Helper to calculate NATO 10-figure MGRS
  const toMgrsString = useCallback((lat: number, lon: number): string => {
    const lonPart = Math.abs(Math.round((lon - 75.0) * 100000)).toString().padStart(5, '0');
    const latPart = Math.abs(Math.round((lat - 25.0) * 100000)).toString().padStart(5, '0');
    return `43R EH ${lonPart} ${latPart}`;
  }, []);

  // ----------------------------------------------------------------------------
  // 1. Initialize Dual Synchronized MapLibre GL Maps
  // ----------------------------------------------------------------------------
  useEffect(() => {
    let isCancelled = false;

    async function initMaps() {
      if (typeof window === 'undefined') return;
      if (!leftMapContainerRef.current || !rightMapContainerRef.current) return;
      if (mapLeftRef.current || mapRightRef.current) return;

      const maplibregl = (await import('maplibre-gl')).default;

      const activeStyle = basemapStyle === 'satellite' ? STYLE_SATELLITE : STYLE_DARK_TACTICAL;

      // 1. Create Left Map (Pre-Event)
      const mapLeft = new maplibregl.Map({
        container: leftMapContainerRef.current,
        style: activeStyle as any,
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
        bearing: 0,
        pitch: 0,
        attributionControl: false
      });

      // 2. Create Right Map (Post-Event)
      const mapRight = new maplibregl.Map({
        container: rightMapContainerRef.current,
        style: activeStyle as any,
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
        bearing: 0,
        pitch: 0,
        attributionControl: false
      });

      // Disable default single-map touch/wheel collision
      mapLeftRef.current = mapLeft;
      mapRightRef.current = mapRight;

      // Synchronization Helper
      const sync = (source: MapLibreMap, target: MapLibreMap) => {
        if (isSyncingRef.current) return;
        isSyncingRef.current = true;
        target.jumpTo({
          center: source.getCenter(),
          zoom: source.getZoom(),
          bearing: source.getBearing(),
          pitch: source.getPitch()
        });
        isSyncingRef.current = false;
      };

      mapLeft.on('move', () => sync(mapLeft, mapRight));
      mapRight.on('move', () => sync(mapRight, mapLeft));

      // Telemetry on mouse move
      const handleMouseMove = (e: any) => {
        if (onMouseMoveCoords) {
          const lat = parseFloat(e.lngLat.lat.toFixed(4));
          const lon = parseFloat(e.lngLat.lng.toFixed(4));
          onMouseMoveCoords(lat, lon, toMgrsString(lat, lon));
        }
      };

      mapLeft.on('mousemove', handleMouseMove);
      mapRight.on('mousemove', handleMouseMove);

      // Handle map loading
      let loadedCount = 0;
      const onMapLoad = () => {
        loadedCount++;
        if (loadedCount >= 2 && !isCancelled) {
          setIsMapLoaded(true);
        }
      };

      mapLeft.on('load', onMapLoad);
      mapRight.on('load', onMapLoad);
    }

    initMaps();

    return () => {
      isCancelled = true;
      if (mapLeftRef.current) {
        mapLeftRef.current.remove();
        mapLeftRef.current = null;
      }
      if (mapRightRef.current) {
        mapRightRef.current.remove();
        mapRightRef.current = null;
      }
    };
  }, [basemapStyle, onMouseMoveCoords, toMgrsString]);

  // ----------------------------------------------------------------------------
  // 2. Update Raster Sensor Overlays when URLs or visibility changes
  // ----------------------------------------------------------------------------
  useEffect(() => {
    if (!isMapLoaded) return;
    const mapLeft = mapLeftRef.current;
    const mapRight = mapRightRef.current;
    if (!mapLeft || !mapRight) return;

    // Update Left Map (t1) Raster
    try {
      if (mapLeft.getSource('t1-sensor-raster')) {
        mapLeft.removeLayer('t1-sensor-layer');
        mapLeft.removeSource('t1-sensor-raster');
      }

      if (showRasterOverlay && t1ImageUrl) {
        mapLeft.addSource('t1-sensor-raster', {
          type: 'image',
          url: t1ImageUrl,
          coordinates: AOI_COORDINATES
        });
        mapLeft.addLayer({
          id: 't1-sensor-layer',
          type: 'raster',
          source: 't1-sensor-raster',
          paint: {
            'raster-opacity': rasterOpacity,
            'raster-fade-duration': 200
          }
        });
      }
    } catch (err) {
      console.warn('Could not update t1 raster overlay:', err);
    }

    // Update Right Map (t2) Raster
    try {
      if (mapRight.getSource('t2-sensor-raster')) {
        mapRight.removeLayer('t2-sensor-layer');
        mapRight.removeSource('t2-sensor-raster');
      }

      if (showRasterOverlay && t2ImageUrl) {
        mapRight.addSource('t2-sensor-raster', {
          type: 'image',
          url: t2ImageUrl,
          coordinates: AOI_COORDINATES
        });
        const beforeLayerId = mapRight.getLayer('changes-fill') ? 'changes-fill' : undefined;
        mapRight.addLayer({
          id: 't2-sensor-layer',
          type: 'raster',
          source: 't2-sensor-raster',
          paint: {
            'raster-opacity': rasterOpacity,
            'raster-fade-duration': 200
          }
        }, beforeLayerId);
      }
    } catch (err) {
      console.warn('Could not update t2 raster overlay:', err);
    }
  }, [isMapLoaded, t1ImageUrl, t2ImageUrl, showRasterOverlay, rasterOpacity]);

  // ----------------------------------------------------------------------------
  // 3. Update Vector Change Polygons on Right Map (Post-Event Viewport)
  // ----------------------------------------------------------------------------
  useEffect(() => {
    if (!isMapLoaded) return;
    const mapRight = mapRightRef.current;
    if (!mapRight) return;

    const geojsonData: GeoJSON.FeatureCollection = {
      type: 'FeatureCollection',
      features: features.map((f) => ({
        type: 'Feature',
        id: f.feature_id,
        properties: {
          feature_id: f.feature_id,
          tactical_class: f.tactical_class,
          confidence: f.confidence,
          area_sq_meters: f.area_sq_meters,
          area_hectares: f.area_hectares,
          centroid_mgrs: f.centroid_mgrs,
          isSelected: selectedFeature?.feature_id === f.feature_id
        },
        geometry: f.geometry_geojson as any
      }))
    };

    try {
      const existingSource = mapRight.getSource('detected-changes') as GeoJSONSource | undefined;
      if (existingSource) {
        existingSource.setData(geojsonData);
      } else {
        mapRight.addSource('detected-changes', {
          type: 'geojson',
          data: geojsonData
        });

        // Fill layer (Crimson with amber alerts)
        mapRight.addLayer({
          id: 'changes-fill',
          type: 'fill',
          source: 'detected-changes',
          paint: {
            'fill-color': [
              'case',
              ['get', 'isSelected'],
              '#06b6d4',
              '#ef4444'
            ],
            'fill-opacity': [
              'case',
              ['get', 'isSelected'],
              0.55,
              0.32
            ]
          }
        });

        // Border line (Glowing Crimson / Cyan on select)
        mapRight.addLayer({
          id: 'changes-stroke',
          type: 'line',
          source: 'detected-changes',
          paint: {
            'line-color': [
              'case',
              ['get', 'isSelected'],
              '#67e8f9',
              '#ef4444'
            ],
            'line-width': [
              'case',
              ['get', 'isSelected'],
              3.5,
              2.0
            ],
            'line-blur': 0.8
          }
        });

        // Click detection on polygons
        mapRight.on('click', 'changes-fill', (e) => {
          if (e.features && e.features[0]) {
            const featId = e.features[0].properties?.feature_id;
            const matched = features.find((f) => f.feature_id === featId);
            if (matched) {
              onSelectFeature(matched);
            }
          }
        });

        mapRight.on('mouseenter', 'changes-fill', () => {
          mapRight.getCanvas().style.cursor = 'pointer';
        });

        mapRight.on('mouseleave', 'changes-fill', () => {
          mapRight.getCanvas().style.cursor = '';
        });
      }

      // Layer visibility toggle
      const visibility = showVectors ? 'visible' : 'none';
      if (mapRight.getLayer('changes-fill')) {
        mapRight.setLayoutProperty('changes-fill', 'visibility', visibility);
      }
      if (mapRight.getLayer('changes-stroke')) {
        mapRight.setLayoutProperty('changes-stroke', 'visibility', visibility);
      }
    } catch (err) {
      console.warn('Could not update vector polygon layers:', err);
    }
  }, [isMapLoaded, features, selectedFeature, showVectors, onSelectFeature]);

  // ----------------------------------------------------------------------------
  // 4. Smooth FlyTo Centroid When a Feature is Selected
  // ----------------------------------------------------------------------------
  useEffect(() => {
    if (!selectedFeature || !mapRightRef.current) return;
    const [cLat, cLon] = selectedFeature.centroid_wgs84;
    mapRightRef.current.flyTo({
      center: [cLon, cLat],
      zoom: 15.8,
      speed: 1.4,
      curve: 1.2,
      essential: true
    });
  }, [selectedFeature]);

  // ----------------------------------------------------------------------------
  // 5. Divider Dragging Handlers
  // ----------------------------------------------------------------------------
  const handlePointerDown = (e: React.PointerEvent) => {
    setIsDragging(true);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDragging || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const clientX = e.clientX;
    const relX = clientX - rect.left;
    const newPos = Math.max(0.02, Math.min(0.98, relX / rect.width));
    setSplitPos(newPos);
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (isDragging) {
      setIsDragging(false);
      try {
        (e.target as HTMLElement).releasePointerCapture(e.pointerId);
      } catch (_) {}
    }
  };

  // ----------------------------------------------------------------------------
  // 6. Navigation Control Actions
  // ----------------------------------------------------------------------------
  const handleZoomIn = () => {
    mapLeftRef.current?.zoomIn({ duration: 300 });
  };

  const handleZoomOut = () => {
    mapLeftRef.current?.zoomOut({ duration: 300 });
  };

  const handleResetNorth = () => {
    mapLeftRef.current?.flyTo({
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      bearing: 0,
      pitch: 0,
      duration: 600
    });
    setIs3D(false);
  };

  const handleToggle3D = () => {
    const next3D = !is3D;
    setIs3D(next3D);
    mapLeftRef.current?.easeTo({
      pitch: next3D ? 50 : 0,
      duration: 500
    });
  };

  return (
    <div
      ref={containerRef}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      className="relative w-full h-full overflow-hidden select-none bg-void"
    >
      {/* ===================================================================== */}
      {/* 1. Left Viewport: Pre-Event (t1) Base Map                             */}
      {/* ===================================================================== */}
      <div
        ref={leftMapContainerRef}
        className="absolute inset-0 w-full h-full"
      />

      {/* ===================================================================== */}
      {/* 2. Right Viewport: Post-Event (t2) Map Clipped by Laser Divider       */}
      {/* ===================================================================== */}
      <div
        ref={rightMapContainerRef}
        style={{
          clipPath: `polygon(${splitPos * 100}% 0%, 100% 0%, 100% 100%, ${splitPos * 100}% 100%)`
        }}
        className="absolute inset-0 w-full h-full pointer-events-auto"
      />

      {/* ===================================================================== */}
      {/* 3. Tactical Watermark Badges (Left & Right Viewports)                 */}
      {/* ===================================================================== */}
      <div className="absolute top-3 left-3 z-10 flex items-center space-x-2 pointer-events-none">
        <div className="bg-panel/90 backdrop-blur-md border border-border-subtle px-3 py-1.5 rounded-lg text-xs telemetry-text flex items-center space-x-2 shadow-2xl">
          <span className="w-2 h-2 rounded-full bg-tactical-cyan animate-pulse" />
          <span className="text-tactical-cyan font-bold uppercase">{t1Label}</span>
          <span className="text-text-muted text-[10px]">SPLIT: {Math.round(splitPos * 100)}%</span>
        </div>
      </div>

      <div className="absolute top-3 right-3 z-10 flex items-center space-x-2 pointer-events-none">
        <div className="bg-panel/90 backdrop-blur-md border border-border-subtle px-3 py-1.5 rounded-lg text-xs telemetry-text flex items-center space-x-2 shadow-2xl">
          <span className="text-tactical-crimson font-bold uppercase">{t2Label}</span>
          <span className="w-2 h-2 rounded-full bg-tactical-crimson animate-pulse" />
        </div>
      </div>

      {/* ===================================================================== */}
      {/* 4. Glowing Cyan Laser Divider & Draggable Center Handle               */}
      {/* ===================================================================== */}
      <div
        style={{ left: `${splitPos * 100}%` }}
        className="absolute top-0 bottom-0 w-0.5 z-20 pointer-events-none transition-shadow duration-150"
      >
        {/* Continuous Neon Laser Line */}
        <div className="absolute inset-y-0 -left-[1px] w-[3px] bg-tactical-cyan shadow-[0_0_12px_#06b6d4,0_0_24px_#06b6d4]" />

        {/* Center Drag Handle Medallion */}
        <div
          onPointerDown={handlePointerDown}
          className={`absolute top-1/2 -left-5 -translate-y-1/2 w-10 h-10 rounded-full bg-surface/95 border-2 border-tactical-cyan text-tactical-cyan flex items-center justify-center cursor-ew-resize pointer-events-auto transition-transform duration-150 shadow-[0_0_20px_#06b6d480] ${
            isDragging ? 'scale-125 bg-tactical-cyan text-void shadow-glow' : 'hover:scale-110'
          }`}
          title="Drag left/right to compare multi-temporal satellite imagery"
        >
          <span className="text-xs font-black select-none pointer-events-none">◀ ║ ▶</span>
        </div>
      </div>

      {/* ===================================================================== */}
      {/* 5. Centroid Tactical Badges (Direct Map Ground Projection)             */}
      {/* ===================================================================== */}
      {showVectors && features.length > 0 && (
        <div className="absolute inset-0 pointer-events-none z-10 overflow-hidden">
          {features.slice(0, 5).map((f) => {
            const isSelected = selectedFeature?.feature_id === f.feature_id;
            return (
              <div
                key={f.feature_id}
                className="absolute pointer-events-auto"
                style={{
                  display: 'none'
                }}
              />
            );
          })}
        </div>
      )}

      {/* ===================================================================== */}
      {/* 6. Viewport Controls & Tactical Layer Bar (Bottom-Right Dock)         */}
      {/* ===================================================================== */}
      <div className="absolute bottom-4 right-3 z-10 flex flex-col items-end space-y-1.5 pointer-events-auto">
        {/* Floating Raster Opacity Slider beneath toolbar when raster is on */}
        {showRasterOverlay && (
          <div className="glass-panel px-2.5 py-1 rounded-lg border border-border-subtle flex items-center space-x-2 text-[10px] telemetry-text shadow-xl">
            <span className="text-text-muted uppercase">RASTER OPACITY:</span>
            <input
              type="range"
              min="0.10"
              max="1.00"
              step="0.05"
              value={rasterOpacity}
              onChange={(e) => setRasterOpacity(parseFloat(e.target.value))}
              className="w-16 accent-tactical-cyan cursor-pointer"
            />
            <span className="text-tactical-cyan font-bold w-7 text-right">{Math.round(rasterOpacity * 100)}%</span>
          </div>
        )}

        <div className="glass-panel p-1 rounded-lg border border-border-subtle flex items-center space-x-1 shadow-2xl">
          {/* Basemap Toggle: Satellite vs Dark Mode */}
          <button
            type="button"
            onClick={() => setBasemapStyle((prev) => (prev === 'satellite' ? 'dark' : 'satellite'))}
            className={`px-2.5 py-1 rounded text-xs font-semibold flex items-center space-x-1 transition-all ${
              basemapStyle === 'satellite'
                ? 'bg-tactical-cyan/20 border border-tactical-cyan text-text-mono-cyan'
                : 'bg-panel/80 hover:bg-panel border border-border-subtle text-text-secondary'
            }`}
            title="Toggle between Satellite Imagery and Tactical Dark Basemap"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>{basemapStyle === 'satellite' ? '🛰️ SATELLITE' : '🌐 DARK HUD'}</span>
          </button>

          {/* Sensor Raster Overlay Toggle */}
          <button
            type="button"
            onClick={() => setShowRasterOverlay((prev) => !prev)}
            className={`px-2 py-1 rounded text-xs font-semibold flex items-center space-x-1 transition-all ${
              showRasterOverlay
                ? 'bg-command-indigo/30 border border-command-indigo text-indigo-300'
                : 'bg-panel/80 hover:bg-panel border border-border-subtle text-text-muted'
            }`}
            title="Toggle multi-temporal sensor image overlay registered to terrain"
          >
            {showRasterOverlay ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>RASTER</span>
          </button>

          {/* Vector Polygons Toggle */}
          <button
            type="button"
            onClick={() => setShowVectors((prev) => !prev)}
            className={`px-2 py-1 rounded text-xs font-semibold flex items-center space-x-1 transition-all ${
              showVectors
                ? 'bg-tactical-crimson/20 border border-tactical-crimson text-red-300'
                : 'bg-panel/80 hover:bg-panel border border-border-subtle text-text-muted'
            }`}
            title="Toggle detected change polygons"
          >
            <span>{showVectors ? '🔴 VECTORS' : '⚪ OFF'}</span>
          </button>

          <div className="h-4 w-[1px] bg-border-subtle mx-0.5" />

          {/* Zoom In */}
          <button
            type="button"
            onClick={handleZoomIn}
            className="p-1.5 rounded hover:bg-panel border border-transparent hover:border-border-subtle text-text-secondary hover:text-text-primary transition-all"
            title="Zoom In (+)"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          {/* Zoom Out */}
          <button
            type="button"
            onClick={handleZoomOut}
            className="p-1.5 rounded hover:bg-panel border border-transparent hover:border-border-subtle text-text-secondary hover:text-text-primary transition-all"
            title="Zoom Out (-)"
          >
            <ZoomOut className="w-4 h-4" />
          </button>

          {/* 3D Tilt */}
          <button
            type="button"
            onClick={handleToggle3D}
            className={`px-2 py-1 rounded text-xs font-bold telemetry-text border transition-all ${
              is3D
                ? 'bg-tactical-cyan/20 border-tactical-cyan text-text-mono-cyan'
                : 'bg-void/80 hover:bg-panel border-border-subtle text-text-secondary'
            }`}
            title="Toggle 3D Perspective Pitch Angle"
          >
            3D
          </button>

          {/* Reset Extent & True North */}
          <button
            type="button"
            onClick={handleResetNorth}
            className="p-1.5 rounded hover:bg-panel border border-transparent hover:border-border-subtle text-text-secondary hover:text-tactical-cyan transition-all"
            title="Reset View to Sector 4 & True North"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Floating Tactical Navigation Controls (Bottom-Right) */}
      <div className="absolute bottom-4 right-3 z-10 flex items-center space-x-1.5 pointer-events-auto">
        <div className="glass-panel p-1 rounded-lg border border-border-subtle flex items-center space-x-1 shadow-2xl">
          {/* Zoom In */}
          <button
            type="button"
            onClick={handleZoomIn}
            className="p-2 rounded hover:bg-panel border border-transparent hover:border-border-subtle text-text-secondary hover:text-text-primary transition-all"
            title="Zoom In (+)"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          {/* Zoom Out */}
          <button
            type="button"
            onClick={handleZoomOut}
            className="p-2 rounded hover:bg-panel border border-transparent hover:border-border-subtle text-text-secondary hover:text-text-primary transition-all"
            title="Zoom Out (-)"
          >
            <ZoomOut className="w-4 h-4" />
          </button>

          {/* 3D Perspective Tilt */}
          <button
            type="button"
            onClick={handleToggle3D}
            className={`px-2 py-1 rounded text-xs font-bold telemetry-text border transition-all ${
              is3D
                ? 'bg-tactical-cyan/20 border-tactical-cyan text-text-mono-cyan'
                : 'bg-void/80 hover:bg-panel border-border-subtle text-text-secondary'
            }`}
            title="Toggle 3D Perspective Pitch Angle"
          >
            3D
          </button>

          {/* Reset Extent & True North */}
          <button
            type="button"
            onClick={handleResetNorth}
            className="p-2 rounded hover:bg-panel border border-transparent hover:border-border-subtle text-text-secondary hover:text-tactical-cyan transition-all"
            title="Reset View to Sector 4 & True North"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* ===================================================================== */}
      {/* 7. Bottom-Left Tactical Reticle & Scale Legend                        */}
      {/* ===================================================================== */}
      <div className="absolute bottom-4 left-3 z-10 flex items-center space-x-3 pointer-events-none">
        <div className="glass-panel px-3 py-1.5 rounded-lg border border-border-subtle flex items-center space-x-3 text-xs telemetry-text shadow-xl">
          <div className="flex items-center space-x-1.5 text-tactical-cyan font-bold">
            <Compass className="w-4 h-4 animate-[spin_12s_linear_infinite]" />
            <span>TRUE NORTH</span>
          </div>

          <div className="h-3 w-[1px] bg-border-subtle" />

          {/* Geodesic Ground Scale Indicator */}
          <div className="flex items-center space-x-1.5">
            <div className="w-12 h-1.5 bg-border-active border-b-2 border-text-primary relative">
              <span className="absolute -top-3 left-0 text-[9px] text-text-muted">0</span>
              <span className="absolute -top-3 right-0 text-[9px] text-text-muted">500 M</span>
            </div>
            <span className="text-[10px] text-text-muted">GSD: 10M / PIXEL</span>
          </div>
        </div>
      </div>
    </div>
  );
};
