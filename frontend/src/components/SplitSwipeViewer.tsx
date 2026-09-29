'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Eye,
  EyeOff,
  Compass
} from 'lucide-react';
import { DetectedPolygonFeature } from '../types/geoint';

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
  onMouseMoveCoords,
  onUploadT1,
  onUploadT2
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Split-swipe divider position (0.0 to 1.0, default 0.50)
  const [splitPos, setSplitPos] = useState<number>(0.50);
  const [isDraggingDivider, setIsDraggingDivider] = useState<boolean>(false);
  const [hoveredFeature, setHoveredFeature] = useState<DetectedPolygonFeature | null>(null);

  // Pan and Zoom viewport state
  const [zoom, setZoom] = useState<number>(1.0);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [panStart, setPanStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [showOverlays, setShowOverlays] = useState<boolean>(true);

  // Loaded HTMLImageElements cache
  const [imgT1, setImgT1] = useState<HTMLImageElement | null>(null);
  const [imgT2, setImgT2] = useState<HTMLImageElement | null>(null);

  // Viewport spatial extent (Sector 4 AOI)
  const aoi = {
    min_lat: 25.000,
    max_lat: 25.100,
    min_lon: 75.000,
    max_lon: 75.100
  };

  // Load Images whenever URLs change
  useEffect(() => {
    let isCancelled = false;

    const loadImg = (url: string): Promise<HTMLImageElement> => {
      return new Promise((resolve, reject) => {
        const img = new Image();
        if (!url.startsWith('data:') && !url.startsWith('blob:')) {
          img.crossOrigin = 'anonymous';
        }
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
        img.src = url;
      });
    };

    loadImg(t1ImageUrl)
      .then((img) => {
        if (!isCancelled) setImgT1(img);
      })
      .catch((e) => console.warn('Could not load t1 image:', e));

    loadImg(t2ImageUrl)
      .then((img) => {
        if (!isCancelled) setImgT2(img);
      })
      .catch((e) => console.warn('Could not load t2 image:', e));

    return () => {
      isCancelled = true;
    };
  }, [t1ImageUrl, t2ImageUrl]);

  // Coordinate projection: WGS84 -> Canvas (x, y) with Pan & Zoom
  const projectCoords = useCallback(
    (lon: number, lat: number, width: number, height: number): [number, number] => {
      // Normalized 0 to 1 inside AOI
      const normX = (lon - aoi.min_lon) / (aoi.max_lon - aoi.min_lon);
      const normY = (aoi.max_lat - lat) / (aoi.max_lat - aoi.min_lat);

      // Center with pan and zoom
      const centerX = width / 2;
      const centerY = height / 2;

      const baseX = normX * width;
      const baseY = normY * height;

      const screenX = centerX + (baseX - centerX) * zoom + panOffset.x;
      const screenY = centerY + (baseY - centerY) * zoom + panOffset.y;

      return [screenX, screenY];
    },
    [zoom, panOffset]
  );

  // Inverse projection: Canvas (x, y) -> WGS84
  const unprojectCoords = useCallback(
    (screenX: number, screenY: number, width: number, height: number): [number, number] => {
      const centerX = width / 2;
      const centerY = height / 2;

      const baseX = (screenX - panOffset.x - centerX) / zoom + centerX;
      const baseY = (screenY - panOffset.y - centerY) / zoom + centerY;

      const lon = aoi.min_lon + (baseX / width) * (aoi.max_lon - aoi.min_lon);
      const lat = aoi.max_lat - (baseY / height) * (aoi.max_lat - aoi.min_lat);

      return [lat, lon];
    },
    [zoom, panOffset]
  );

  // Handle Divider Dragging & Pan
  const handlePointerDown = (e: React.PointerEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const clientX = e.clientX;
    const clientY = e.clientY;
    const relX = clientX - rect.left;
    const splitPx = rect.width * splitPos;

    // If clicking near the laser divider (within 24px), drag divider
    if (Math.abs(relX - splitPx) <= 24) {
      setIsDraggingDivider(true);
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    } else {
      // Otherwise initiate viewport pan
      setIsPanning(true);
      setPanStart({ x: clientX - panOffset.x, y: clientY - panOffset.y });
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const clientX = e.clientX;
    const clientY = e.clientY;
    const relX = clientX - rect.left;
    const relY = clientY - rect.top;

    if (isDraggingDivider) {
      const newPos = Math.max(0.02, Math.min(0.98, relX / rect.width));
      setSplitPos(newPos);
    } else if (isPanning) {
      setPanOffset({
        x: clientX - panStart.x,
        y: clientY - panStart.y
      });
    }

    // Telemetry updates
    if (onMouseMoveCoords) {
      const [lat, lon] = unprojectCoords(relX, relY, rect.width, rect.height);
      const fakeMgrs = `43R EH ${Math.round((lon - 75.0) * 100000).toString().padStart(5, '0')} ${Math.round((lat - 25.0) * 100000).toString().padStart(5, '0')}`;
      onMouseMoveCoords(lat, lon, fakeMgrs);
    }

    // Polygon Hover Detection
    if (!isDraggingDivider && !isPanning && showOverlays) {
      let matched: DetectedPolygonFeature | null = null;
      for (const feat of features) {
        const [cLat, cLon] = feat.centroid_wgs84;
        const [cx, cy] = projectCoords(cLon, cLat, rect.width, rect.height);
        const dist = Math.hypot(relX - cx, relY - cy);
        if (dist < 36 * zoom) {
          matched = feat;
          break;
        }
      }
      setHoveredFeature(matched);
    }
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (isDraggingDivider) {
      setIsDraggingDivider(false);
      try {
        (e.target as HTMLElement).releasePointerCapture(e.pointerId);
      } catch (_) {}
    }
    if (isPanning) {
      setIsPanning(false);
    }
  };

  const handleClick = (e: React.MouseEvent) => {
    if (!containerRef.current || isDraggingDivider) return;
    const rect = containerRef.current.getBoundingClientRect();
    const relX = e.clientX - rect.left;
    const relY = e.clientY - rect.top;

    for (const feat of features) {
      const [cLat, cLon] = feat.centroid_wgs84;
      const [cx, cy] = projectCoords(cLon, cLat, rect.width, rect.height);
      const dist = Math.hypot(relX - cx, relY - cy);
      if (dist < 40 * zoom) {
        onSelectFeature(feat);
        return;
      }
    }
    onSelectFeature(null);
  };

  // Zoom with mouse wheel
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    setZoom((prev) => Math.max(0.5, Math.min(prev * zoomFactor, 8.0)));
  };

  const resetView = () => {
    setZoom(1.0);
    setPanOffset({ x: 0, y: 0 });
    setSplitPos(0.50);
  };

  // ----------------------------------------------------------------------------
  // Master Canvas Render Loop: Real Satellite Imagery + Split-Swipe + Overlays
  // ----------------------------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear frame
    ctx.clearRect(0, 0, width, height);
    const splitPx = width * splitPos;

    // Image destination dimensions with zoom and pan
    const centerX = width / 2;
    const centerY = height / 2;
    const destW = width * zoom;
    const destH = height * zoom;
    const destX = centerX - destW / 2 + panOffset.x;
    const destY = centerY - destH / 2 + panOffset.y;

    // --------------------------------------------------------------------------
    // 1. Render Left Viewport: Pre-Event (t1) Satellite Image
    // --------------------------------------------------------------------------
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, splitPx, height);
    ctx.clip();

    if (imgT1) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(imgT1, destX, destY, destW, destH);
    } else {
      // Fallback if loading
      ctx.fillStyle = '#0b0f19';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#64748b';
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.fillText('LOADING PRE-EVENT (t1) SATELLITE TILE...', 30, height / 2);
    }

    // Left Viewport Ambient Tint & Scanlines
    ctx.fillStyle = 'rgba(6, 182, 212, 0.02)';
    ctx.fillRect(0, 0, splitPx, height);

    ctx.restore();

    // --------------------------------------------------------------------------
    // 2. Render Right Viewport: Post-Event (t2) Satellite Image
    // --------------------------------------------------------------------------
    ctx.save();
    ctx.beginPath();
    ctx.rect(splitPx, 0, width - splitPx, height);
    ctx.clip();

    if (imgT2) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(imgT2, destX, destY, destW, destH);
    } else {
      ctx.fillStyle = '#0e1626';
      ctx.fillRect(splitPx, 0, width - splitPx, height);
      ctx.fillStyle = '#64748b';
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.fillText('LOADING POST-EVENT (t2) SATELLITE TILE...', splitPx + 30, height / 2);
    }

    // Right Viewport Ambient Tint
    ctx.fillStyle = 'rgba(239, 68, 68, 0.03)';
    ctx.fillRect(splitPx, 0, width - splitPx, height);

    ctx.restore();

    // --------------------------------------------------------------------------
    // 3. Render Detected Change Vector Overlays (Continuous Across Both Viewports)
    // --------------------------------------------------------------------------
    if (showOverlays && features.length > 0) {
      features.forEach((feat) => {
        const isSelected = selectedFeature?.feature_id === feat.feature_id;
        const isHovered = hoveredFeature?.feature_id === feat.feature_id;

        const allRings: number[][][] = [];
        const geomType = feat.geometry_geojson.type;
        const rawCoords = feat.geometry_geojson.coordinates;

        if (geomType === 'MultiPolygon' && Array.isArray(rawCoords)) {
          for (const poly of rawCoords) {
            if (Array.isArray(poly)) {
              for (const ring of poly) {
                if (Array.isArray(ring) && ring.length >= 3) {
                  allRings.push(ring);
                }
              }
            }
          }
        } else if (geomType === 'Polygon' && Array.isArray(rawCoords)) {
          for (const ring of rawCoords) {
            if (Array.isArray(ring) && ring.length >= 3) {
              allRings.push(ring);
            }
          }
        }

        allRings.forEach((ring) => {
          ctx.beginPath();
          ring.forEach(([lon, lat], ptIdx) => {
            const [x, y] = projectCoords(lon, lat, width, height);
            if (ptIdx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.closePath();

          // Fill styling
          if (isSelected) {
            ctx.fillStyle = 'rgba(6, 182, 212, 0.45)';
          } else if (isHovered) {
            ctx.fillStyle = 'rgba(239, 68, 68, 0.55)';
          } else {
            ctx.fillStyle = 'rgba(239, 68, 68, 0.35)';
          }
          ctx.fill();

          // Stroke styling with military glow
          ctx.save();
          if (isSelected) {
            ctx.strokeStyle = '#06b6d4';
            ctx.lineWidth = 3.0;
            ctx.shadowColor = '#06b6d4';
            ctx.shadowBlur = 12;
          } else if (isHovered) {
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 3.0;
            ctx.shadowColor = '#ef4444';
            ctx.shadowBlur = 15;
          } else {
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 2.0;
            ctx.shadowColor = '#ef4444';
            ctx.shadowBlur = 6;
          }
          ctx.stroke();
          ctx.restore();
        });

        // Tactical Reticle Label Box at Polygon Centroid
        const [cLat, cLon] = feat.centroid_wgs84;
        const [cx, cy] = projectCoords(cLon, cLat, width, height);

        ctx.save();
        ctx.font = '10px "JetBrains Mono", monospace';
        const labelText = `${feat.tactical_class.toUpperCase()} [${Math.round(feat.area_sq_meters).toLocaleString()} m²]`;
        const textMetrics = ctx.measureText(labelText);
        const tagW = textMetrics.width + 12;
        const tagH = 18;
        const tagX = cx - tagW / 2;
        const tagY = cy - 22;

        // Tag background badge
        ctx.fillStyle = isSelected ? 'rgba(6, 182, 212, 0.90)' : isHovered ? 'rgba(239, 68, 68, 0.90)' : 'rgba(15, 23, 42, 0.85)';
        ctx.strokeStyle = isSelected ? '#06b6d4' : isHovered ? '#ffffff' : '#ef4444';
        ctx.lineWidth = 1;
        ctx.fillRect(tagX, tagY, tagW, tagH);
        ctx.strokeRect(tagX, tagY, tagW, tagH);

        // Center reticle dot
        ctx.fillStyle = isSelected ? '#06b6d4' : '#ef4444';
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fill();

        // Tag text
        ctx.fillStyle = isSelected || isHovered ? '#050811' : '#f8fafc';
        ctx.font = 'bold 9px "JetBrains Mono", monospace';
        ctx.fillText(labelText, tagX + 6, tagY + 13);
        ctx.restore();
      });
    }

    // --------------------------------------------------------------------------
    // 4. Render Tactical Laser Split Divider [◀ ║ ▶]
    // --------------------------------------------------------------------------
    ctx.save();
    // Glowing cyan vertical laser line
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = '#06b6d4';
    ctx.shadowBlur = 10;

    ctx.beginPath();
    ctx.moveTo(splitPx, 0);
    ctx.lineTo(splitPx, height);
    ctx.stroke();

    // Laser handle pill at center vertical position
    const handleY = height / 2;
    const pillW = 44;
    const pillH = 34;

    ctx.shadowBlur = 15;
    ctx.fillStyle = '#0b0f19';
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.roundRect(splitPx - pillW / 2, handleY - pillH / 2, pillW, pillH, 8);
    ctx.fill();
    ctx.stroke();

    // Arrows and Grip Marks: [◀ ║ ▶]
    ctx.fillStyle = '#06b6d4';
    ctx.font = 'bold 11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('◀ ║ ▶', splitPx, handleY);

    ctx.restore();

    // --------------------------------------------------------------------------
    // 5. Tactical Viewport Framing & Corner Reticles
    // --------------------------------------------------------------------------
    ctx.strokeStyle = '#33415580';
    ctx.lineWidth = 1;
    // Top-left reticle
    ctx.beginPath();
    ctx.moveTo(10, 25);
    ctx.lineTo(10, 10);
    ctx.lineTo(25, 10);
    ctx.stroke();

    // Top-right reticle
    ctx.beginPath();
    ctx.moveTo(width - 25, 10);
    ctx.lineTo(width - 10, 10);
    ctx.lineTo(width - 10, 25);
    ctx.stroke();

    // Bottom-left reticle
    ctx.beginPath();
    ctx.moveTo(10, height - 25);
    ctx.lineTo(10, height - 10);
    ctx.lineTo(25, height - 10);
    ctx.stroke();

    // Bottom-right reticle
    ctx.beginPath();
    ctx.moveTo(width - 25, height - 10);
    ctx.lineTo(width - 10, height - 10);
    ctx.lineTo(width - 10, height - 25);
    ctx.stroke();
  }, [
    splitPos,
    zoom,
    panOffset,
    showOverlays,
    features,
    selectedFeature,
    hoveredFeature,
    imgT1,
    imgT2,
    projectCoords
  ]);

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onClick={handleClick}
      onWheel={handleWheel}
      className="relative w-full h-full bg-void overflow-hidden select-none cursor-crosshair"
    >
      {/* Primary HTML5 Geospatial Split-Swipe Canvas */}
      <canvas ref={canvasRef} className="block w-full h-full" />

      {/* Top Banner Tag: Left View (t1) vs Right View (t2) Indicator */}
      <div className="absolute top-3 left-3 z-10 flex items-center space-x-2 pointer-events-none">
        <div className="bg-surface/90 border border-tactical-cyan/40 px-2.5 py-1 rounded shadow-lg backdrop-blur-md flex items-center space-x-2">
          <span className="w-2 h-2 rounded-full bg-tactical-cyan animate-pulse" />
          <span className="text-[11px] font-extrabold text-tactical-cyan tracking-wider telemetry-text uppercase">
            {t1Label}
          </span>
          <span className="text-border-active text-xs">│</span>
          <span className="text-[10px] text-text-muted telemetry-text">
            SPLIT: {(splitPos * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      <div className="absolute top-3 right-3 z-10 flex items-center space-x-2 pointer-events-none">
        <div className="bg-surface/90 border border-tactical-crimson/40 px-2.5 py-1 rounded shadow-lg backdrop-blur-md flex items-center space-x-2">
          <span className="text-[11px] font-extrabold text-tactical-crimson tracking-wider telemetry-text uppercase">
            {t2Label}
          </span>
          <span className="w-2 h-2 rounded-full bg-tactical-crimson animate-pulse" />
        </div>
      </div>

      {/* Floating Viewport Navigation Toolbar */}
      <div className="absolute bottom-4 right-4 z-10 flex items-center space-x-1.5 bg-surface/90 border border-border-subtle p-1.5 rounded-lg shadow-2xl backdrop-blur-md pointer-events-auto">
        <button
          type="button"
          onClick={() => setShowOverlays(!showOverlays)}
          title="Toggle Change Vector Overlay"
          className={`p-1.5 rounded transition-all text-xs flex items-center space-x-1 ${
            showOverlays ? 'bg-tactical-crimson/20 text-tactical-crimson border border-tactical-crimson/40' : 'hover:bg-panel text-text-muted'
          }`}
        >
          {showOverlays ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          <span className="text-[10px] font-bold telemetry-text">{showOverlays ? 'VECTORS ON' : 'VECTORS OFF'}</span>
        </button>

        <span className="text-border-active text-xs">│</span>

        <button
          type="button"
          onClick={() => setZoom((z) => Math.min(z * 1.25, 8.0))}
          title="Zoom In"
          className="p-1.5 hover:bg-panel rounded text-text-secondary hover:text-text-primary transition-all"
        >
          <ZoomIn className="w-4 h-4" />
        </button>

        <button
          type="button"
          onClick={() => setZoom((z) => Math.max(z * 0.8, 0.5))}
          title="Zoom Out"
          className="p-1.5 hover:bg-panel rounded text-text-secondary hover:text-text-primary transition-all"
        >
          <ZoomOut className="w-4 h-4" />
        </button>

        <button
          type="button"
          onClick={resetView}
          title="Reset Viewport Pan & Zoom"
          className="p-1.5 hover:bg-panel rounded text-text-secondary hover:text-text-primary transition-all flex items-center space-x-1"
        >
          <RotateCcw className="w-4 h-4" />
          <span className="text-[10px] telemetry-text text-text-muted font-bold">1:1</span>
        </button>
      </div>

      {/* Bottom Left Scale & Orientation HUD */}
      <div className="absolute bottom-4 left-4 z-10 flex items-center space-x-3 bg-surface/90 border border-border-subtle px-3 py-1.5 rounded shadow-xl backdrop-blur-md pointer-events-none">
        <div className="flex items-center space-x-1.5 text-[11px] telemetry-text font-bold text-tactical-cyan">
          <Compass className="w-4 h-4 text-tactical-cyan animate-spin-slow" />
          <span>TRUE NORTH</span>
        </div>
        <span className="text-border-active text-xs">│</span>
        <div className="flex items-center space-x-2">
          <div className="w-16 h-1.5 border-b-2 border-l-2 border-r-2 border-text-primary" />
          <span className="text-[10px] telemetry-text text-text-primary font-bold">500 M</span>
        </div>
        <span className="text-border-active text-xs">│</span>
        <span className="text-[10px] telemetry-text text-text-muted">GSD: 10M / PIXEL</span>
      </div>
    </div>
  );
};
