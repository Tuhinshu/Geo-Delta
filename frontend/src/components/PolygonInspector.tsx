'use client';

import React from 'react';
import {
  Crosshair,
  MapPin,
  Maximize2,
  Shield,
  X,
  Compass,
  AlertTriangle,
  CheckCircle2
} from 'lucide-react';
import { DetectedPolygonFeature } from '../types/geoint';

interface PolygonInspectorProps {
  feature: DetectedPolygonFeature | null;
  onClose: () => void;
}

export const PolygonInspector: React.FC<PolygonInspectorProps> = ({ feature, onClose }) => {
  if (!feature) return null;

  const lat = feature.centroid_wgs84[0];
  const lon = feature.centroid_wgs84[1];

  return (
    <div className="absolute bottom-12 left-4 z-20 w-84 max-w-[340px] glass-panel-elevated rounded-lg p-3.5 shadow-2xl border border-tactical-cyan/60 animate-fadeIn">
      {/* Header */}
      <div className="flex items-start justify-between border-b border-border-subtle/80 pb-2">
        <div className="flex items-center space-x-2">
          <div className="p-1 rounded bg-tactical-cyan/15 text-tactical-cyan border border-tactical-cyan/40">
            <Crosshair className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[10px] text-text-muted font-bold tracking-wider uppercase">FEATURE INSPECTOR</div>
            <div className="text-xs font-bold text-text-primary leading-tight truncate max-w-[210px]">
              {feature.tactical_class}
            </div>
          </div>
        </div>

        <button
          type="button"
          onClick={onClose}
          className="text-text-muted hover:text-text-primary p-1 rounded hover:bg-panel transition-colors"
          title="Deselect feature"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
        {/* Confidence */}
        <div className="bg-void/80 rounded p-2 border border-border-subtle">
          <div className="text-[10px] text-text-muted font-medium uppercase tracking-wider">CONFIDENCE</div>
          <div className="telemetry-text text-sm font-bold text-tactical-emerald mt-0.5">
            {(feature.confidence * 100).toFixed(1)}%
          </div>
        </div>

        {/* Surface Area */}
        <div className="bg-void/80 rounded p-2 border border-border-subtle">
          <div className="text-[10px] text-text-muted font-medium uppercase tracking-wider">GROUND AREA</div>
          <div className="telemetry-text text-sm font-bold text-text-mono-cyan mt-0.5 truncate">
            {feature.area_sq_meters.toLocaleString()} <span className="text-[10px] font-normal">m²</span>
          </div>
          <div className="telemetry-text text-[10px] text-text-muted">
            ({feature.area_hectares.toFixed(3)} ha)
          </div>
        </div>
      </div>

      {/* Geodetic Coordinates Section */}
      <div className="mt-2.5 bg-void/80 rounded p-2 border border-border-subtle space-y-1.5 telemetry-text text-xs">
        {/* WGS 84 */}
        <div className="flex items-center justify-between text-text-secondary text-[11px]">
          <span className="flex items-center space-x-1 text-text-muted">
            <MapPin className="w-3 h-3 text-tactical-cyan" />
            <span>WGS 84:</span>
          </span>
          <span className="text-text-primary font-medium">
            {lat.toFixed(6)}° N, {lon.toFixed(6)}° E
          </span>
        </div>

        {/* MGRS */}
        <div className="flex items-center justify-between text-text-secondary text-[11px]">
          <span className="flex items-center space-x-1 text-text-muted">
            <Compass className="w-3 h-3 text-tactical-amber" />
            <span>MGRS:</span>
          </span>
          <span className="text-tactical-amber font-semibold tracking-wide">
            {feature.centroid_mgrs}
          </span>
        </div>
      </div>

      {/* UUID Tag */}
      <div className="mt-2 flex items-center justify-between text-[10px] text-text-muted telemetry-text pt-1.5 border-t border-border-subtle/50">
        <span>FEATURE ID:</span>
        <span className="truncate max-w-[170px] text-text-secondary" title={feature.feature_id}>
          {feature.feature_id}
        </span>
      </div>
    </div>
  );
};
