'use client';

import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import type { Map as MapLibreMap, GeoJSONSource } from 'maplibre-gl';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Eye,
  EyeOff,
  Crosshair,
  Upload,
  RefreshCw
} from 'lucide-react';
import { DetectedPolygonFeature } from '../types/geoint';

// Tactical Airbase Sector 4 Center & Bounds
const DEFAULT_CENTER: [number, number] = [75.0750, 25.0450]; // [Longitude, Latitude]
const DEFAULT_ZOOM = 14.2;

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
  heatmapOverlayUrl?: string | null;
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
  heatmapOverlayUrl,
  t1Label = 'PRE-EVENT (t1)',
  t2Label = 'POST-EVENT (t2)',
  features,
  selectedFeature,
  onSelectFeature,
  onMouseMoveCoords,
  onUploadT1,
  onUploadT2
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const leftMapContainerRef = useRef<HTMLDivElement>(null);
  const rightMapContainerRef = useRef<HTMLDivElement>(null);

  // Hidden File Input References for Direct Click-to-Browse
  const fileInputT1Ref = useRef<HTMLInputElement>(null);
  const fileInputT2Ref = useRef<HTMLInputElement>(null);
  const [isDraggingT1, setIsDraggingT1] = useState<boolean>(false);
  const [isDraggingT2, setIsDraggingT2] = useState<boolean>(false);

  // MapLibre Instances
  const mapLeftRef = useRef<MapLibreMap | null>(null);
  const mapRightRef = useRef<MapLibreMap | null>(null);
  const isSyncingRef = useRef<boolean>(false);

  // Viewport & Layer Settings
  const [displayMode, setDisplayMode] = useState<'canvas' | 'map'>('canvas');
  const [basemapStyle, setBasemapStyle] = useState<'satellite' | 'dark'>('satellite');
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [rasterOpacity, setRasterOpacity] = useState<number>(0.80);
  const [showVectors, setShowVectors] = useState<boolean>(true);
  const [isMapLoaded, setIsMapLoaded] = useState<boolean>(false);

  // Synchronized Canvas Pan & Zoom State
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const startPanRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Hovered feature state for tooltip
  const [hoveredFeature, setHoveredFeature] = useState<DetectedPolygonFeature | null>(null);

  // Helper to calculate NATO 10-figure MGRS
  const toMgrsString = useCallback((lat: number, lon: number): string => {
    const lonPart = Math.abs(Math.round((lon - 75.0) * 100000)).toString().padStart(5, '0');
    const latPart = Math.abs(Math.round((lat - 25.0) * 100000)).toString().padStart(5, '0');
    return `43R EH ${lonPart} ${latPart}`;
  }, []);

  // ----------------------------------------------------------------------------
  // 1. Initialize Dual Synchronized MapLibre GL Maps (Side-by-Side)
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
        attributionControl: false,
        pitchWithRotate: true,
        dragRotate: true
      });

      // 2. Create Right Map (Post-Event)
      const mapRight = new maplibregl.Map({
        container: rightMapContainerRef.current,
        style: activeStyle as any,
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
        attributionControl: false,
        pitchWithRotate: true,
        dragRotate: true
      });

      mapLeftRef.current = mapLeft;
      mapRightRef.current = mapRight;

      // Synchronize cameras between Left and Right maps
      const syncCameras = (sourceMap: MapLibreMap, targetMap: MapLibreMap) => {
        if (isSyncingRef.current) return;
        isSyncingRef.current = true;
        targetMap.jumpTo({
          center: sourceMap.getCenter(),
          zoom: sourceMap.getZoom(),
          bearing: sourceMap.getBearing(),
          pitch: sourceMap.getPitch()
        });
        isSyncingRef.current = false;
      };

      mapLeft.on('move', () => syncCameras(mapLeft, mapRight));
      mapRight.on('move', () => syncCameras(mapRight, mapLeft));

      // Telemetry hover updates
      const handleMouseMove = (e: any) => {
        if (!onMouseMoveCoords) return;
        const { lng, lat } = e.lngLat;
        const mgrs = toMgrsString(lat, lng);
        onMouseMoveCoords(lat, lng, mgrs);
      };

      mapLeft.on('mousemove', handleMouseMove);
      mapRight.on('mousemove', handleMouseMove);

      let mapsLoadedCount = 0;
      const onMapLoad = () => {
        mapsLoadedCount += 1;
        if (mapsLoadedCount >= 2 && !isCancelled) {
          setIsMapLoaded(true);
        }
      };

      mapLeft.on('load', onMapLoad);
      mapRight.on('load', onMapLoad);
    }

    if (displayMode === 'map') {
      initMaps();
    }

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
      setIsMapLoaded(false);
    };
  }, [displayMode, basemapStyle, onMouseMoveCoords, toMgrsString]);

  // ----------------------------------------------------------------------------
  // 2. MapLibre GeoJSON Vector Polygons
  // ----------------------------------------------------------------------------
  useEffect(() => {
    if (!isMapLoaded || !mapRightRef.current) return;

    const map = mapRightRef.current;
    try {
      const geojsonFeatureCollection: GeoJSON.FeatureCollection = {
        type: 'FeatureCollection',
        features: (features || []).map((f) => ({
          type: 'Feature',
          id: f.feature_id,
          properties: {
            feature_id: f.feature_id,
            tactical_class: f.tactical_class,
            confidence: f.confidence,
            area_sq_meters: f.area_sq_meters,
            area_hectares: f.area_hectares,
            isSelected: selectedFeature?.feature_id === f.feature_id
          },
          geometry: f.geometry_geojson as any
        }))
      };

      const source = map.getSource('tactical-polygons-source') as GeoJSONSource;
      if (source) {
        source.setData(geojsonFeatureCollection);
      } else {
        map.addSource('tactical-polygons-source', {
          type: 'geojson',
          data: geojsonFeatureCollection
        });

        // Polygon Fill Layer
        map.addLayer({
          id: 'tactical-polygons-fill',
          type: 'fill',
          source: 'tactical-polygons-source',
          paint: {
            'fill-color': [
              'case',
              ['get', 'isSelected'],
              '#06b6d4',
              '#ef4444'
            ],
            'fill-opacity': showVectors ? 0.35 : 0.0
          }
        });

        // Polygon Boundary Neon Stroke
        map.addLayer({
          id: 'tactical-polygons-line',
          type: 'line',
          source: 'tactical-polygons-source',
          paint: {
            'line-color': [
              'case',
              ['get', 'isSelected'],
              '#06b6d4',
              '#ef4444'
            ],
            'line-width': [
              'case',
              ['get', 'isSelected'],
              3.5,
              2.0
            ],
            'line-opacity': showVectors ? 1.0 : 0.0
          }
        });

        map.on('click', 'tactical-polygons-fill', (e) => {
          if (e.features && e.features[0]) {
            const featId = e.features[0].properties?.feature_id;
            const target = features.find((f) => f.feature_id === featId);
            if (target) onSelectFeature(target);
          }
        });
      }
    } catch (err) {
      console.warn('Could not update vector polygon layers:', err);
    }
  }, [isMapLoaded, features, selectedFeature, showVectors, onSelectFeature]);

  // ----------------------------------------------------------------------------
  // 3. Canvas Synchronized Pan & Zoom Handlers
  // ----------------------------------------------------------------------------
  const handlePointerDown = (e: React.PointerEvent) => {
    if (displayMode !== 'canvas' || !t1ImageUrl || !t2ImageUrl) return;
    setIsPanning(true);
    startPanRef.current = {
      x: e.clientX - pan.x,
      y: e.clientY - pan.y
    };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isPanning || displayMode !== 'canvas') return;
    setPan({
      x: e.clientX - startPanRef.current.x,
      y: e.clientY - startPanRef.current.y
    });
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (isPanning) {
      setIsPanning(false);
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      } catch (_) {}
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (displayMode !== 'canvas' || !t1ImageUrl || !t2ImageUrl) return;
    e.preventDefault();
    const zoomDelta = e.deltaY < 0 ? 0.15 : -0.15;
    setZoom((prev) => Math.max(0.6, Math.min(5.0, prev + zoomDelta)));
  };

  const handleZoomIn = () => {
    if (displayMode === 'canvas') {
      setZoom((prev) => Math.min(5.0, prev + 0.25));
    } else {
      mapLeftRef.current?.zoomIn({ duration: 300 });
    }
  };

  const handleZoomOut = () => {
    if (displayMode === 'canvas') {
      setZoom((prev) => Math.max(0.6, prev - 0.25));
    } else {
      mapLeftRef.current?.zoomOut({ duration: 300 });
    }
  };

  const handleResetNorth = () => {
    if (displayMode === 'canvas') {
      setZoom(1.0);
      setPan({ x: 0, y: 0 });
    } else {
      mapLeftRef.current?.flyTo({
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
        bearing: 0,
        pitch: 0,
        duration: 600
      });
    }
  };

  // ----------------------------------------------------------------------------
  // 4. Transform GeoJSON Polygon Coordinates to SVG (512x512 normalized space)
  // ----------------------------------------------------------------------------
  const svgPolygons = useMemo(() => {
    if (!features || features.length === 0) return [];

    return features.map((f, idx) => {
      const geom = f.geometry_geojson;
      const rings: string[] = [];

      if (geom && geom.coordinates) {
        if (geom.type === 'Polygon') {
          const ring = geom.coordinates[0] || [];
          const pts = ring.map(([lon, lat]: [number, number]) => {
            const x = Math.max(0, Math.min(512, (lon - 75.050) / 0.0001));
            const y = Math.max(0, Math.min(512, (25.045 - lat) / 0.0001));
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          }).join(' ');
          if (pts) rings.push(pts);
        } else if (geom.type === 'MultiPolygon') {
          for (const poly of geom.coordinates) {
            const ring = (poly && poly[0]) || [];
            const pts = ring.map(([lon, lat]: [number, number]) => {
              const x = Math.max(0, Math.min(512, (lon - 75.050) / 0.0001));
              const y = Math.max(0, Math.min(512, (25.045 - lat) / 0.0001));
              return `${x.toFixed(1)},${y.toFixed(1)}`;
            }).join(' ');
            if (pts) rings.push(pts);
          }
        }
      }

      // Compute centroid pixel for badge pin
      let cx = 256;
      let cy = 256;
      if (f.centroid_wgs84 && f.centroid_wgs84.length === 2) {
        const [cLat, cLon] = f.centroid_wgs84;
        cx = Math.max(20, Math.min(492, (cLon - 75.050) / 0.0001));
        cy = Math.max(20, Math.min(492, (25.045 - cLat) / 0.0001));
      }

      const isSelected = selectedFeature?.feature_id === f.feature_id;

      return {
        feature: f,
        index: idx + 1,
        rings,
        cx,
        cy,
        isSelected
      };
    });
  }, [features, selectedFeature]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full flex flex-row overflow-hidden select-none bg-void"
    >
      {/* Hidden File Input Elements for Native File Picker Selection */}
      <input
        ref={fileInputT1Ref}
        type="file"
        accept="image/*,.tif,.tiff"
        onChange={(e) => {
          if (e.target.files && e.target.files[0] && onUploadT1) {
            onUploadT1(e.target.files[0]);
          }
        }}
        className="hidden"
      />
      <input
        ref={fileInputT2Ref}
        type="file"
        accept="image/*,.tif,.tiff"
        onChange={(e) => {
          if (e.target.files && e.target.files[0] && onUploadT2) {
            onUploadT2(e.target.files[0]);
          }
        }}
        className="hidden"
      />

      {/* SVG Neon Glow Filters */}
      <svg className="absolute w-0 h-0" aria-hidden="true">
        <defs>
          <filter id="neon-glow-red" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="neon-glow-cyan" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      </svg>

      {/* ===================================================================== */}
      {/* 1. LEFT PANE: PRE-EVENT (t1) BASELINE REFERENCE & DROPZONE            */}
      {/* ===================================================================== */}
      <div
        className={`flex-1 relative h-full border-r border-border-subtle/80 overflow-hidden bg-void flex items-center justify-center transition-colors ${
          !t1ImageUrl ? 'cursor-pointer hover:bg-surface/30' : 'cursor-grab active:cursor-grabbing'
        }`}
        onPointerDown={t1ImageUrl ? handlePointerDown : undefined}
        onPointerMove={t1ImageUrl ? handlePointerMove : undefined}
        onPointerUp={t1ImageUrl ? handlePointerUp : undefined}
        onWheel={t1ImageUrl ? handleWheel : undefined}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDraggingT1(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDraggingT1(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDraggingT1(false);
          if (e.dataTransfer.files && e.dataTransfer.files[0] && onUploadT1) {
            onUploadT1(e.dataTransfer.files[0]);
          }
        }}
      >
        {/* Top Header Badge */}
        <div className="absolute top-3 left-3 z-20 flex items-center space-x-2 pointer-events-auto">
          <div className="bg-panel/90 backdrop-blur-md border border-tactical-cyan/40 px-3 py-1.5 rounded-lg text-xs telemetry-text flex items-center space-x-2 shadow-2xl">
            <span className="w-2 h-2 rounded-full bg-tactical-cyan animate-pulse" />
            <span className="text-tactical-cyan font-bold uppercase">{t1Label}</span>
            <span className="text-text-muted text-[10px]">BASELINE REFERENCE</span>
            {t1ImageUrl && (
              <button
                type="button"
                onClick={() => fileInputT1Ref.current?.click()}
                className="ml-2 text-[10px] text-tactical-cyan hover:underline flex items-center space-x-1 border-l border-border-subtle pl-2"
                title="Replace T1 Image"
              >
                <RefreshCw className="w-2.5 h-2.5" />
                <span>REPLACE</span>
              </button>
            )}
          </div>
        </div>

        {/* Viewport Content */}
        <div
          ref={leftMapContainerRef}
          className={`absolute inset-0 w-full h-full ${displayMode === 'canvas' ? 'hidden' : 'block'}`}
        />

        {displayMode === 'canvas' && (
          <div className="relative w-full h-full flex items-center justify-center p-4">
            {t1ImageUrl ? (
              <div
                style={{
                  transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`,
                  transformOrigin: 'center center',
                  transition: isPanning ? 'none' : 'transform 0.1s ease-out'
                }}
                className="relative max-w-full max-h-full flex items-center justify-center shadow-2xl rounded overflow-hidden"
              >
                <img
                  src={t1ImageUrl}
                  alt="t1 pre-event baseline"
                  className="w-full h-full max-h-[calc(100vh-180px)] object-contain select-none"
                />

                {/* Ghost Reference Outlines (Dotted) showing alteration area on baseline */}
                {showVectors && svgPolygons.length > 0 && (
                  <svg
                    viewBox="0 0 512 512"
                    preserveAspectRatio="none"
                    className="absolute inset-0 w-full h-full pointer-events-none"
                  >
                    {svgPolygons.map(({ feature, rings, isSelected }) => (
                      <g key={`t1-ghost-${feature.feature_id}`}>
                        {rings.map((pts, rIdx) => (
                          <polygon
                            key={rIdx}
                            points={pts}
                            fill="rgba(6, 182, 212, 0.08)"
                            stroke={isSelected ? '#06b6d4' : 'rgba(6, 182, 212, 0.5)'}
                            strokeWidth={isSelected ? '2.5' : '1.5'}
                            strokeDasharray="4 3"
                          />
                        ))}
                      </g>
                    ))}
                  </svg>
                )}
              </div>
            ) : (
              /* High-Tech Tactical Dropzone Card for T1 */
              <div
                onClick={() => fileInputT1Ref.current?.click()}
                className={`w-full max-w-md p-8 rounded-2xl border-2 border-dashed flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 pointer-events-auto ${
                  isDraggingT1
                    ? 'bg-tactical-cyan/20 border-tactical-cyan shadow-[0_0_30px_#06b6d480] scale-105'
                    : 'bg-surface/50 border-tactical-cyan/40 hover:border-tactical-cyan hover:bg-surface/80 hover:shadow-glow'
                }`}
              >
                <div className="w-16 h-16 rounded-2xl bg-tactical-cyan/10 border border-tactical-cyan/30 flex items-center justify-center mb-4 text-tactical-cyan">
                  <Upload className="w-8 h-8 animate-pulse" />
                </div>
                <div className="text-sm font-extrabold uppercase tracking-wider text-text-primary mb-1">
                  UPLOAD PRE-EVENT (T1) BASELINE
                </div>
                <p className="text-xs text-text-muted mb-4 max-w-xs leading-relaxed">
                  Drag & drop your baseline satellite or drone image here, or click to browse
                </p>
                <div className="flex items-center space-x-2 text-[10px] telemetry-text text-tactical-cyan">
                  <span className="px-2 py-0.5 rounded bg-void border border-border-subtle">PNG</span>
                  <span className="px-2 py-0.5 rounded bg-void border border-border-subtle">JPEG</span>
                  <span className="px-2 py-0.5 rounded bg-void border border-border-subtle">GeoTIFF</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ===================================================================== */}
      {/* CENTER DIVIDER: High-Tech Tactical Telemetry Separator                 */}
      {/* ===================================================================== */}
      <div className="relative w-0.5 z-20 bg-border-subtle flex items-center justify-center pointer-events-none">
        <div className="absolute inset-y-0 -left-[1px] w-[3px] bg-tactical-cyan/40 shadow-[0_0_8px_#06b6d440]" />
        <div className="absolute top-1/2 -translate-y-1/2 -left-6 z-30 px-1.5 py-1 rounded bg-panel/95 border border-tactical-cyan/50 text-[9px] telemetry-text text-tactical-cyan font-extrabold uppercase shadow-2xl flex items-center space-x-1">
          <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
          <span>SYNC</span>
        </div>
      </div>

      {/* ===================================================================== */}
      {/* 2. RIGHT PANE: POST-EVENT (t2) TARGET WITH DROPZONE & HIGHLIGHTS      */}
      {/* ===================================================================== */}
      <div
        className={`flex-1 relative h-full overflow-hidden bg-void flex items-center justify-center transition-colors ${
          !t2ImageUrl ? 'cursor-pointer hover:bg-surface/30' : 'cursor-grab active:cursor-grabbing'
        }`}
        onPointerDown={t2ImageUrl ? handlePointerDown : undefined}
        onPointerMove={t2ImageUrl ? handlePointerMove : undefined}
        onPointerUp={t2ImageUrl ? handlePointerUp : undefined}
        onWheel={t2ImageUrl ? handleWheel : undefined}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDraggingT2(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDraggingT2(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDraggingT2(false);
          if (e.dataTransfer.files && e.dataTransfer.files[0] && onUploadT2) {
            onUploadT2(e.dataTransfer.files[0]);
          }
        }}
      >
        {/* Top Header Badge with Delta Counter */}
        <div className="absolute top-3 right-3 z-20 flex items-center space-x-2 pointer-events-auto">
          <div className="bg-panel/90 backdrop-blur-md border border-tactical-crimson/50 px-3 py-1.5 rounded-lg text-xs telemetry-text flex items-center space-x-2 shadow-2xl">
            <span className="text-tactical-crimson font-bold uppercase">{t2Label}</span>
            <span className="w-2 h-2 rounded-full bg-tactical-crimson animate-pulse" />
            {features.length > 0 && (
              <span className="bg-tactical-crimson/20 border border-tactical-crimson/40 text-red-300 px-2 py-0.5 rounded text-[10px] font-bold">
                {features.length} TARGET{features.length > 1 ? 'S' : ''} HIGHLIGHTED
              </span>
            )}
            {t2ImageUrl && (
              <button
                type="button"
                onClick={() => fileInputT2Ref.current?.click()}
                className="ml-2 text-[10px] text-tactical-crimson hover:underline flex items-center space-x-1 border-l border-border-subtle pl-2"
                title="Replace T2 Image"
              >
                <RefreshCw className="w-2.5 h-2.5" />
                <span>REPLACE</span>
              </button>
            )}
          </div>
        </div>

        {/* Viewport Content */}
        <div
          ref={rightMapContainerRef}
          className={`absolute inset-0 w-full h-full ${displayMode === 'canvas' ? 'hidden' : 'block'}`}
        />

        {displayMode === 'canvas' && (
          <div className="relative w-full h-full flex items-center justify-center p-4">
            {t2ImageUrl ? (
              <div
                style={{
                  transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`,
                  transformOrigin: 'center center',
                  transition: isPanning ? 'none' : 'transform 0.1s ease-out'
                }}
                className="relative max-w-full max-h-full flex items-center justify-center shadow-2xl rounded overflow-hidden"
              >
                {/* 1. Base Post-Event Target Satellite Image */}
                <img
                  src={t2ImageUrl}
                  alt="t2 post-event target"
                  className="w-full h-full max-h-[calc(100vh-180px)] object-contain select-none"
                />

                {/* 2. Deep Learning Probability Heatmap Overlay */}
                {heatmapOverlayUrl && showHeatmap && (
                  <img
                    src={heatmapOverlayUrl}
                    alt="deep learning change heatmap"
                    style={{ opacity: rasterOpacity }}
                    className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen transition-opacity duration-150"
                  />
                )}

                {/* 3. Interactive Vector Polygons & Change Highlights */}
                {showVectors && svgPolygons.length > 0 && (
                  <svg
                    viewBox="0 0 512 512"
                    preserveAspectRatio="none"
                    className="absolute inset-0 w-full h-full pointer-events-auto"
                  >
                    {svgPolygons.map(({ feature, index, rings, cx, cy, isSelected }) => (
                      <g
                        key={`poly-${feature.feature_id}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectFeature(feature);
                        }}
                        onMouseEnter={() => setHoveredFeature(feature)}
                        onMouseLeave={() => setHoveredFeature(null)}
                        className="cursor-pointer group"
                      >
                        {/* Glowing Polygon Fill & Stroke */}
                        {rings.map((pts, rIdx) => (
                          <polygon
                            key={rIdx}
                            points={pts}
                            fill={isSelected ? 'rgba(6, 182, 212, 0.40)' : 'rgba(239, 68, 68, 0.28)'}
                            stroke={isSelected ? '#06b6d4' : '#ef4444'}
                            strokeWidth={isSelected ? '3.5' : '2.5'}
                            filter={isSelected ? 'url(#neon-glow-cyan)' : 'url(#neon-glow-red)'}
                            className="transition-all duration-150 group-hover:fill-red-500/40"
                          />
                        ))}

                        {/* Centroid Tactical Target Crosshair Pin */}
                        <g transform={`translate(${cx}, ${cy})`} className="pointer-events-none">
                          <circle
                            r="6"
                            fill={isSelected ? '#06b6d4' : '#ef4444'}
                            className="animate-pulse"
                          />
                          <circle
                            r="12"
                            fill="none"
                            stroke={isSelected ? '#06b6d4' : '#ef4444'}
                            strokeWidth="1.5"
                            strokeDasharray="2 2"
                          />
                          {/* Target Tag Label */}
                          <foreignObject x="15" y="-12" width="180" height="26">
                            <div className="bg-void/90 backdrop-blur-sm border border-tactical-crimson/80 text-text-primary px-1.5 py-0.5 rounded text-[9px] telemetry-text font-bold truncate shadow-lg flex items-center space-x-1">
                              <span className="text-tactical-crimson">🎯 T-{index}</span>
                              <span className="text-text-muted">|</span>
                              <span className="text-tactical-emerald">{(feature.confidence * 100).toFixed(0)}%</span>
                            </div>
                          </foreignObject>
                        </g>
                      </g>
                    ))}
                  </svg>
                )}
              </div>
            ) : (
              /* High-Tech Tactical Dropzone Card for T2 */
              <div
                onClick={() => fileInputT2Ref.current?.click()}
                className={`w-full max-w-md p-8 rounded-2xl border-2 border-dashed flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 pointer-events-auto ${
                  isDraggingT2
                    ? 'bg-tactical-crimson/20 border-tactical-crimson shadow-[0_0_30px_#ef444480] scale-105'
                    : 'bg-surface/50 border-tactical-crimson/40 hover:border-tactical-crimson hover:bg-surface/80 hover:shadow-glow'
                }`}
              >
                <div className="w-16 h-16 rounded-2xl bg-tactical-crimson/10 border border-tactical-crimson/30 flex items-center justify-center mb-4 text-tactical-crimson">
                  <Upload className="w-8 h-8 animate-pulse" />
                </div>
                <div className="text-sm font-extrabold uppercase tracking-wider text-text-primary mb-1">
                  UPLOAD POST-EVENT (T2) TARGET
                </div>
                <p className="text-xs text-text-muted mb-4 max-w-xs leading-relaxed">
                  Drag & drop your post-event comparison image here, or click to browse
                </p>
                <div className="flex items-center space-x-2 text-[10px] telemetry-text text-tactical-crimson">
                  <span className="px-2 py-0.5 rounded bg-void border border-border-subtle">PNG</span>
                  <span className="px-2 py-0.5 rounded bg-void border border-border-subtle">JPEG</span>
                  <span className="px-2 py-0.5 rounded bg-void border border-border-subtle">GeoTIFF</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ===================================================================== */}
      {/* 3. Floating Hover Tooltip for Detected Targets                        */}
      {/* ===================================================================== */}
      {hoveredFeature && (
        <div className="absolute bottom-20 left-1/2 -translate-x-1/2 z-30 pointer-events-none bg-panel/95 border border-tactical-cyan/80 p-2 rounded-lg text-xs shadow-2xl backdrop-blur-md">
          <div className="text-tactical-cyan font-bold uppercase text-[11px] mb-0.5">
            {hoveredFeature.tactical_class}
          </div>
          <div className="flex items-center space-x-3 text-[10px] telemetry-text text-text-secondary">
            <span>CONFIDENCE: <strong className="text-tactical-emerald">{(hoveredFeature.confidence * 100).toFixed(1)}%</strong></span>
            <span>AREA: <strong className="text-text-primary">{hoveredFeature.area_sq_meters.toLocaleString()} m²</strong></span>
            <span>MGRS: <strong className="text-tactical-amber">{hoveredFeature.centroid_mgrs}</strong></span>
          </div>
        </div>
      )}

      {/* ===================================================================== */}
      {/* 4. Tactical Control Dock (Bottom Center)                              */}
      {/* ===================================================================== */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex items-center space-x-2 pointer-events-auto">
        <div className="glass-panel p-1.5 rounded-xl border border-border-subtle flex items-center space-x-1.5 shadow-2xl backdrop-blur-lg">
          {/* Display Mode: Canvas vs GIS Map */}
          <button
            type="button"
            onClick={() => setDisplayMode((prev) => (prev === 'canvas' ? 'map' : 'canvas'))}
            className="px-2.5 py-1 rounded-lg text-xs font-semibold flex items-center space-x-1 bg-void/80 hover:bg-panel border border-border-subtle text-text-secondary hover:text-text-primary transition-all"
            title="Toggle between Direct Pixel Split Screen and GIS Satellite Map"
          >
            <span>{displayMode === 'canvas' ? '🖼️ PIXEL SPLIT' : '🛰️ GIS MAP'}</span>
          </button>

          {/* Heatmap Toggle */}
          <button
            type="button"
            onClick={() => setShowHeatmap((prev) => !prev)}
            disabled={!heatmapOverlayUrl}
            className={`px-2.5 py-1 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
              showHeatmap && heatmapOverlayUrl
                ? 'bg-tactical-crimson/20 border border-tactical-crimson text-red-300'
                : 'bg-void/80 hover:bg-panel border border-border-subtle text-text-muted'
            }`}
            title="Toggle Deep Learning Change Heatmap"
          >
            {showHeatmap ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>HEATMAP</span>
          </button>

          {/* Heatmap Opacity Slider */}
          {showHeatmap && heatmapOverlayUrl && (
            <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-void/90 border border-border-subtle text-[10px] telemetry-text">
              <span className="text-text-muted">OPACITY:</span>
              <input
                type="range"
                min="0.10"
                max="1.00"
                step="0.05"
                value={rasterOpacity}
                onChange={(e) => setRasterOpacity(parseFloat(e.target.value))}
                className="w-16 accent-tactical-crimson cursor-pointer"
              />
              <span className="text-tactical-crimson font-bold w-6 text-right">
                {Math.round(rasterOpacity * 100)}%
              </span>
            </div>
          )}

          {/* Polygon Vectors Toggle */}
          <button
            type="button"
            onClick={() => setShowVectors((prev) => !prev)}
            disabled={features.length === 0}
            className={`px-2.5 py-1 rounded-lg text-xs font-semibold flex items-center space-x-1 transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
              showVectors && features.length > 0
                ? 'bg-tactical-cyan/20 border border-tactical-cyan text-text-mono-cyan'
                : 'bg-void/80 hover:bg-panel border border-border-subtle text-text-muted'
            }`}
            title="Toggle Detected Change Polygon Highlights"
          >
            <Crosshair className="w-3.5 h-3.5" />
            <span>{showVectors && features.length > 0 ? 'POLYGONS ON' : 'POLYGONS OFF'}</span>
          </button>

          <div className="h-4 w-[1px] bg-border-subtle mx-0.5" />

          {/* Zoom In */}
          <button
            type="button"
            onClick={handleZoomIn}
            className="p-1.5 rounded-lg hover:bg-panel border border-transparent hover:border-border-subtle text-text-secondary hover:text-text-primary transition-all"
            title="Synchronized Zoom In (+)"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          {/* Zoom Out */}
          <button
            type="button"
            onClick={handleZoomOut}
            className="p-1.5 rounded-lg hover:bg-panel border border-transparent hover:border-border-subtle text-text-secondary hover:text-text-primary transition-all"
            title="Synchronized Zoom Out (-)"
          >
            <ZoomOut className="w-4 h-4" />
          </button>

          {/* Reset Zoom & Pan */}
          <button
            type="button"
            onClick={handleResetNorth}
            className="p-1.5 rounded-lg hover:bg-panel border border-transparent hover:border-border-subtle text-text-secondary hover:text-tactical-cyan transition-all"
            title="Reset Synchronized 1:1 View"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          {displayMode === 'canvas' && (
            <span className="telemetry-text text-[10px] text-text-muted px-1.5">
              {Math.round(zoom * 100)}%
            </span>
          )}
        </div>
      </div>

      {/* Hidden File Inputs for Native Drag-and-Drop and Click-to-Browse */}
      <input
        id="t1-split-file-input"
        type="file"
        ref={fileInputT1Ref}
        onChange={(e) => {
          if (e.target.files && e.target.files[0] && onUploadT1) {
            onUploadT1(e.target.files[0]);
            e.target.value = '';
          }
        }}
        accept="image/*,.tif,.tiff"
        className="hidden"
      />
      <input
        id="t2-split-file-input"
        type="file"
        ref={fileInputT2Ref}
        onChange={(e) => {
          if (e.target.files && e.target.files[0] && onUploadT2) {
            onUploadT2(e.target.files[0]);
            e.target.value = '';
          }
        }}
        accept="image/*,.tif,.tiff"
        className="hidden"
      />
    </div>
  );
};
