'use client';

import React from 'react';
import {
  Shield,
  Crosshair,
  Compass,
  Radio,
  Clock,
  Layers,
  Sparkles
} from 'lucide-react';

interface HeaderBarProps {
  currentLat: number;
  currentLon: number;
  currentMgrs: string;
  sensorName?: string;
  gsdText?: string;
  activePresetTitle?: string;
}

export const HeaderBar: React.FC<HeaderBarProps> = ({
  currentLat,
  currentLon,
  currentMgrs,
  sensorName = 'SENTINEL-2 L2A',
  gsdText = '10M GSD',
  activePresetTitle = 'Strategic Airbase Sector 4'
}) => {
  return (
    <header className="bg-surface/95 border-b border-border-subtle backdrop-blur-md px-4 py-2 z-30 flex items-center justify-between">
      {/* Brand & Mission Identification */}
      <div className="flex items-center space-x-3.5">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded bg-tactical-cyan/15 border border-tactical-cyan/40 flex items-center justify-center">
            <Shield className="w-4 h-4 text-tactical-cyan" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-sm font-black tracking-widest text-text-primary uppercase font-mono">
                GEODELTA
              </span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-tactical-crimson/20 border border-tactical-crimson/50 text-tactical-crimson font-bold telemetry-text">
                GEOINT
              </span>
            </div>
            <div className="text-[10px] text-text-muted tracking-wider uppercase font-medium">
              Automated Satellite Change Detection Platform
            </div>
          </div>
        </div>

        <span className="text-border-active text-xs">│</span>

        {/* Top Military Classification Marker */}
        <div className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded bg-tactical-amber/15 border border-tactical-amber/40 text-tactical-amber text-[10px] font-bold tracking-widest telemetry-text uppercase">
          <span className="w-1.5 h-1.5 rounded-full bg-tactical-amber animate-ping" />
          <span>RESTRICTED // GEOINT ASSESSMENT // FOR OFFICIAL USE ONLY</span>
        </div>
      </div>

      {/* Reticle Telemetry HUD: [LAT: ...] [LON: ...] [MGRS: ...] */}
      <div className="flex items-center space-x-2.5">
        <div className="hidden md:flex items-center space-x-2 bg-void/80 border border-border-subtle px-2.5 py-1 rounded">
          <Crosshair className="w-3.5 h-3.5 text-tactical-cyan animate-pulse" />
          <div className="text-[11px] telemetry-text font-bold text-text-mono-cyan flex items-center space-x-2">
            <span>LAT: {currentLat.toFixed(4)}° N</span>
            <span className="text-border-active">│</span>
            <span>LON: {currentLon.toFixed(4)}° E</span>
            <span className="text-border-active">│</span>
            <span className="text-tactical-amber">MGRS: {currentMgrs}</span>
          </div>
        </div>

        {/* Sensor & Resolution Indicator */}
        <div className="hidden lg:flex items-center space-x-1.5 px-2.5 py-1 rounded bg-panel border border-border-subtle text-[11px] telemetry-text text-text-secondary">
          <Radio className="w-3 h-3 text-tactical-emerald" />
          <span>{sensorName} ({gsdText})</span>
        </div>

        {/* Live System Indicator */}
        <div className="inline-flex items-center space-x-1.5 px-2 py-1 rounded bg-tactical-emerald/15 border border-tactical-emerald/40 text-tactical-emerald text-[10px] font-bold tracking-wider telemetry-text">
          <span className="w-1.5 h-1.5 rounded-full bg-tactical-emerald animate-pulse" />
          <span>OPERATIONAL</span>
        </div>
      </div>
    </header>
  );
};
