'use client';

import React from 'react';
import {
  Cpu,
  Radio,
  Crosshair,
  Layers,
  Activity,
  CheckCircle2
} from 'lucide-react';
import { PipelineStageInfo } from '../types/geoint';

interface StatusBarProps {
  cursorLat?: number;
  cursorLon?: number;
  cursorMgrs?: string;
  nativeGsd?: number;
  stageInfo?: PipelineStageInfo;
  gpuMemoryMb?: number;
  gpuTotalMb?: number;
  isWsConnected?: boolean;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  cursorLat = 25.0456,
  cursorLon = 75.1289,
  cursorMgrs = '43R EH 12890 04560',
  nativeGsd = 10.0,
  stageInfo,
  gpuMemoryMb = 4200,
  gpuTotalMb = 8192,
  isWsConnected = true
}) => {
  const gpuPercent = Math.round((gpuMemoryMb / gpuTotalMb) * 100);

  const getStageBadge = (stage?: string) => {
    switch (stage) {
      case 'COG_STREAMING':
        return { label: '1. COG BYTE-STREAMING', color: 'text-tactical-cyan' };
      case 'ECC_ALIGNMENT':
        return { label: '2. SUB-PIXEL ECC ALIGNMENT', color: 'text-tactical-amber' };
      case 'SIAMESE_INFERENCE':
        return { label: '3. SIAMESE CROSS-ATTENTION', color: 'text-tactical-indigo' };
      case 'VECTORIZATION':
        return { label: '4. MORPHOLOGICAL VECTORIZATION', color: 'text-tactical-cyan' };
      case 'COMPLETED':
        return { label: 'PIPELINE COMPLETE', color: 'text-tactical-emerald' };
      default:
        return { label: 'SYSTEM READY', color: 'text-text-muted' };
    }
  };

  const badge = getStageBadge(stageInfo?.stage);

  return (
    <footer className="h-8 bg-surface border-t border-border-subtle px-4 flex items-center justify-between text-xs telemetry-text select-none z-30">
      {/* Left: Reticle Telemetry */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-1.5 text-text-muted">
          <Crosshair className="w-3.5 h-3.5 text-tactical-cyan" />
          <span className="text-text-secondary">RETICLE:</span>
        </div>
        <span className="text-text-mono-cyan font-medium">
          {cursorLat.toFixed(5)}° N, {cursorLon.toFixed(5)}° E
        </span>
        <span className="text-border-active">│</span>
        <span className="text-tactical-amber font-medium">
          MGRS: {cursorMgrs}
        </span>
        <span className="text-border-active">│</span>
        <span className="text-text-muted">
          GSD: <span className="text-text-primary">{nativeGsd.toFixed(1)}m</span>/px
        </span>
      </div>

      {/* Middle: Active Pipeline Stage */}
      <div className="hidden md:flex items-center space-x-2">
        <span className="text-text-muted">STAGE:</span>
        <span className={`font-bold ${badge.color}`}>
          {badge.label}
        </span>
        {stageInfo?.percent !== undefined && stageInfo.stage !== 'IDLE' && stageInfo.stage !== 'COMPLETED' && (
          <span className="text-text-secondary">({stageInfo.percent}%)</span>
        )}
      </div>

      {/* Right: Hardware & WebSocket Status */}
      <div className="flex items-center space-x-4">
        {/* GPU VRAM Indicator */}
        <div className="flex items-center space-x-1.5 text-text-secondary">
          <Cpu className="w-3.5 h-3.5 text-tactical-cyan" />
          <span>
            VRAM: {(gpuMemoryMb / 1024).toFixed(1)} / {(gpuTotalMb / 1024).toFixed(1)} GB ({gpuPercent}%)
          </span>
        </div>

        <span className="text-border-active">│</span>

        {/* WebSocket Connection Indicator */}
        <div className="flex items-center space-x-1.5">
          <Radio className={`w-3.5 h-3.5 ${isWsConnected ? 'text-tactical-emerald animate-pulse' : 'text-tactical-crimson'}`} />
          <span className={isWsConnected ? 'text-tactical-emerald font-bold' : 'text-tactical-crimson font-bold'}>
            {isWsConnected ? 'WSS: ONLINE' : 'WSS: OFFLINE'}
          </span>
        </div>
      </div>
    </footer>
  );
};
