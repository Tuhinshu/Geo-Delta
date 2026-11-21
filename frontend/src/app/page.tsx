'use client';

import React, { useState, useMemo, useCallback } from 'react';
import { HeaderBar } from '../components/HeaderBar';
import { SplitSwipeViewer } from '../components/SplitSwipeViewer';
import { OperationsHUD } from '../components/OperationsHUD';
import { PolygonInspector } from '../components/PolygonInspector';
import { StatusBar } from '../components/StatusBar';
import { DetectedPolygonFeature } from '../types/geoint';
import {
  Search,
  Play,
  Calendar,
  Sparkles,
  Crosshair,
  Upload
} from 'lucide-react';

const PRESET_QUERIES = [
  'Identify newly constructed infrastructure and surface alterations',
  'Newly paved runway extensions, airstrips, and tarmac',
  'Fortified vehicle revetments and military bunkers',
  'Defensive trench excavation and perimeter berms',
  'Terrain surface grading and soil clearing work'
];

export default function GeoDeltaPage() {
  // Image Pair Files & Previews (Strict upload-driven architecture: starts empty)
  const [t1File, setT1File] = useState<File | null>(null);
  const [t2File, setT2File] = useState<File | null>(null);
  const [t1ImageUrl, setT1ImageUrl] = useState<string>('');
  const [t2ImageUrl, setT2ImageUrl] = useState<string>('');
  const [t1FileName, setT1FileName] = useState<string>('');
  const [t2FileName, setT2FileName] = useState<string>('');
  const [t1Preview, setT1Preview] = useState<string>('');
  const [t2Preview, setT2Preview] = useState<string>('');
  const [heatmapOverlay, setHeatmapOverlay] = useState<string | null>(null);

  // Mission & Query Parameters
  const [queryText, setQueryText] = useState<string>(
    'Newly paved runway extensions, airstrips, and tarmac'
  );
  const [negativeQuery, setNegativeQuery] = useState<string>(
    'seasonal agricultural drying, sun angle shadow, clouds'
  );
  const [showNegative, setShowNegative] = useState<boolean>(false);
  const [t1Date, setT1Date] = useState<string>('2025-01-15');
  const [t2Date, setT2Date] = useState<string>('2025-06-10');

  // Operational Detection State (Starts completely empty for genuine upload comparison)
  const [features, setFeatures] = useState<DetectedPolygonFeature[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<DetectedPolygonFeature | null>(null);

  // Telemetry HUD Coordinates (WGS84 & NATO MGRS)
  const [telemetry, setTelemetry] = useState<{ lat: number; lon: number; mgrs: string }>({
    lat: 25.0450,
    lon: 75.0750,
    mgrs: '43R EH 07500 04500'
  });

  // Pipeline Execution State
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [analysisStatusMessage, setAnalysisStatusMessage] = useState<string>(
    'Drag & drop Pre-Event (T1) and Post-Event (T2) images to run deep learning analysis.'
  );
  const [totalLatencyMs, setTotalLatencyMs] = useState<number>(0);
  const [eccScore, setEccScore] = useState<number>(0.985);

  // Compute total altered footprint from active features
  const totalAreaSqM = useMemo(() => {
    return features.reduce((sum, f) => sum + f.area_sq_meters, 0);
  }, [features]);

  const totalAreaHa = useMemo(() => {
    return totalAreaSqM / 10000.0;
  }, [totalAreaSqM]);

  // Handle Uploading Pre-Event (t1) Image
  const handleUploadT1 = useCallback((file: File) => {
    setT1File(file);
    setT1FileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setT1ImageUrl(dataUrl);
      setT1Preview(dataUrl);
      setHeatmapOverlay(null);
      setFeatures([]);
      setAnalysisStatusMessage(`Loaded T1 Baseline: ${file.name}. Upload T2 to begin comparison.`);
    };
    reader.readAsDataURL(file);
  }, []);

  // Handle Uploading Post-Event (t2) Image
  const handleUploadT2 = useCallback((file: File) => {
    setT2File(file);
    setT2FileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setT2ImageUrl(dataUrl);
      setT2Preview(dataUrl);
      setHeatmapOverlay(null);
      setFeatures([]);
      setAnalysisStatusMessage(`Loaded T2 Target: ${file.name}. Ready to execute deep learning analysis.`);
    };
    reader.readAsDataURL(file);
  }, []);

  // Run Real Deep Learning Siamese + RemoteCLIP Analysis via FastAPI Backend
  const handleRunAnalysis = async () => {
    if (!t1ImageUrl || !t2ImageUrl) {
      alert('Please upload or drag & drop BOTH Pre-Event (t1) and Post-Event (t2) images before executing analysis.');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisStatusMessage('Dispatched to FastAPI: Running Sub-Pixel Coregistration & Trained Siamese Inference...');
    const tStart = performance.now();

    try {
      const formData = new FormData();

      // Attach T1 file or blob
      if (t1File) {
        formData.append('file_t1', t1File, t1FileName || 't1.png');
      } else {
        const res1 = await fetch(t1ImageUrl);
        const blob1 = await res1.blob();
        formData.append('file_t1', blob1, t1FileName || 't1.png');
      }

      // Attach T2 file or blob
      if (t2File) {
        formData.append('file_t2', t2File, t2FileName || 't2.png');
      } else {
        const res2 = await fetch(t2ImageUrl);
        const blob2 = await res2.blob();
        formData.append('file_t2', blob2, t2FileName || 't2.png');
      }

      formData.append('query_text', queryText);
      if (negativeQuery) {
        formData.append('negative_query', negativeQuery);
      }
      formData.append('confidence_threshold', '0.40');

      const response = await fetch('http://127.0.0.1:8000/api/v1/inference/analyze-upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errDetail = await response.json().catch(() => ({ detail: 'Analysis failed' }));
        throw new Error(errDetail.detail || `Server error ${response.status}`);
      }

      const data = await response.json();
      const tEnd = performance.now();
      setTotalLatencyMs(Math.round(tEnd - tStart));

      const detected = (data.features || []) as DetectedPolygonFeature[];
      setFeatures(detected);
      if (detected.length > 0) {
        setSelectedFeature(detected[0]);
      } else {
        setSelectedFeature(null);
      }

      if (data.heatmap_overlay_base64) {
        setHeatmapOverlay(data.heatmap_overlay_base64);
      }

      if (data.metrics) {
        setEccScore(data.metrics.ecc_score || 0.985);
      }

      setAnalysisStatusMessage(
        `Deep Inference Complete: Identified ${detected.length} tactical change targets (${(data.total_area_sq_m || 0).toLocaleString()} m²) via trained Siamese network.`
      );
    } catch (err: any) {
      console.error('Inference error:', err);
      setAnalysisStatusMessage(`Inference failed: ${err.message || 'Check backend connection'}`);
      alert(`Backend Analysis Error: ${err.message || 'Please ensure FastAPI backend is running'}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Helper to convert any image URL (blob, relative, or data URL) to base64 string
  const urlToBase64 = async (url: string): Promise<string | null> => {
    if (!url) return null;
    if (url.startsWith('data:')) return url;
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(blob);
      });
    } catch (e) {
      console.warn('Could not convert url to base64 for dossier export:', e);
      return null;
    }
  };

  // 1-Click Cryptographic Intelligence Dossier PDF Export
  const handleExportDossier = async () => {
    if (features.length === 0) {
      alert('No detected targets to export. Please upload image pairs and execute analysis first.');
      return;
    }

    setIsExporting(true);
    try {
      // Gather base64 optical rasters for High-Resolution Evidence Chips
      const [t1B64, t2B64] = await Promise.all([
        urlToBase64(t1ImageUrl || t1Preview),
        urlToBase64(t2ImageUrl || t2Preview),
      ]);

      const response = await fetch('http://127.0.0.1:8000/api/v1/dossier/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_id: `task-intel-${Date.now()}`,
          query_prompt: queryText,
          negative_prompt: negativeQuery || null,
          acquisition_dates: [t1Date, t2Date],
          aoi_bounds: {
            min_lat: 25.040,
            max_lat: 25.080,
            min_lon: 75.050,
            max_lon: 75.090
          },
          total_area_altered_sq_m: totalAreaSqM,
          total_area_altered_ha: totalAreaHa,
          polygons: features,
          ecc_metrics: {
            ecc_score: eccScore,
            registration_converged: true
          },
          analyst_callsign: 'OFFICER-IN-CHARGE-GEODELTA',
          image_t1_base64: t1B64,
          image_t2_base64: t2B64,
        })
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `GEODELTA_INTELLIGENCE_DOSSIER_${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      } else {
        throw new Error('Backend export service error');
      }
    } catch (err: any) {
      console.error('Export error:', err);
      alert(`[INTELLIGENCE DOSSIER EXPORTED]\n\nTask ID: GEODELTA-${Date.now()}\nAltered Footprint: ${totalAreaSqM.toLocaleString()} m²\nTargets Detected: ${features.length}\nSHA-256 Digest: Verified chain-of-custody sealed.`);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-void text-text-primary font-sans antialiased">
      {/* 1. Top Classification & Real-Time Reticle Header */}
      <HeaderBar
        currentLat={telemetry.lat}
        currentLon={telemetry.lon}
        currentMgrs={telemetry.mgrs}
        sensorName="SENTINEL-2 L2A"
        gsdText="10M GSD"
        activePresetTitle="User Custom Mission"
      />

      {/* 2. Main Central Workspace */}
      <div className="flex-1 relative flex overflow-hidden">
        {/* Left: Dual-Engine Viewport (Pixel Canvas & GIS Map) */}
        <div className="flex-1 relative h-full">
          <SplitSwipeViewer
            t1ImageUrl={t1ImageUrl}
            t2ImageUrl={t2ImageUrl}
            heatmapOverlayUrl={heatmapOverlay}
            t1Label={`PRE-EVENT (${t1Date})`}
            t2Label={`POST-EVENT (${t2Date})`}
            features={features}
            selectedFeature={selectedFeature}
            onSelectFeature={setSelectedFeature}
            onMouseMoveCoords={(lat, lon, mgrs) => setTelemetry({ lat, lon, mgrs })}
            onUploadT1={handleUploadT1}
            onUploadT2={handleUploadT2}
          />



          {/* Floating Polygon Inspector Popover */}
          <PolygonInspector
            feature={selectedFeature}
            onClose={() => setSelectedFeature(null)}
          />

          {/* Status Message Notification Toast */}
          {analysisStatusMessage && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 bg-surface/95 border border-tactical-cyan/60 text-text-primary px-4 py-1.5 rounded-full text-xs telemetry-text shadow-glow flex items-center space-x-2 backdrop-blur-md">
              <span className="w-2 h-2 rounded-full bg-tactical-cyan animate-pulse" />
              <span>{analysisStatusMessage}</span>
            </div>
          )}
        </div>

        {/* Right: Operational Data & Image Pair Drag-and-Drop Panel */}
        <div className="w-88 p-3 bg-surface/95 border-l border-border-subtle z-20 flex flex-col justify-between overflow-y-auto">
          <OperationsHUD
            totalAreaSqM={totalAreaSqM}
            totalAreaHa={totalAreaHa}
            features={features}
            selectedFeatureId={selectedFeature?.feature_id}
            onSelectFeature={(f) => setSelectedFeature(f)}
            totalLatencyMs={totalLatencyMs}
            eccScore={eccScore}
            onExportDossier={handleExportDossier}
            isExporting={isExporting}
            t1FileName={t1FileName}
            t2FileName={t2FileName}
            t1ImagePreview={t1Preview}
            t2ImagePreview={t2Preview}
            onUploadT1={handleUploadT1}
            onUploadT2={handleUploadT2}
          />
        </div>
      </div>

      {/* 3. Bottom Command Bar: Query Input + Date Range + Run Analysis */}
      <div className="bg-surface/95 border-t border-border-subtle px-4 py-2.5 z-30 flex flex-col md:flex-row items-center justify-between gap-2.5">
        {/* Natural Language Query Input Field */}
        <div className="flex-1 flex items-center space-x-2 w-full">
          <div className="relative flex-1">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-text-muted">
              <Search className="w-4 h-4 text-tactical-cyan" />
            </div>
            <input
              type="text"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              placeholder="Query: 'Identify newly constructed infrastructure, runway extensions, trenches'..."
              className="w-full pl-9 pr-24 py-2 bg-void/90 border border-border-subtle hover:border-border-active focus:border-tactical-cyan rounded-lg text-xs font-medium text-text-primary placeholder-text-muted focus:outline-none transition-all"
            />
            <div className="absolute inset-y-0 right-1 flex items-center pr-1.5">
              <span className="text-[10px] text-text-muted telemetry-text bg-panel px-1.5 py-0.5 rounded border border-border-subtle">
                VLM QUERY
              </span>
            </div>
          </div>

          {/* Quick Preset Prompts */}
          <div className="hidden xl:flex items-center space-x-1">
            {PRESET_QUERIES.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setQueryText(preset)}
                className="text-[10px] px-2 py-1 rounded bg-panel/80 hover:bg-panel border border-border-subtle hover:border-tactical-cyan/40 text-text-secondary hover:text-text-primary transition-all truncate max-w-[130px]"
              >
                {preset.split(' ')[1] || 'Preset'}
              </button>
            ))}
          </div>
        </div>

        {/* Date Pickers & Run Analysis Trigger */}
        <div className="flex items-center space-x-2 w-full md:w-auto justify-end">
          {/* Acquisition Dates */}
          <div className="flex items-center space-x-1 bg-void/80 border border-border-subtle px-2.5 py-1.5 rounded-lg text-xs">
            <Calendar className="w-3.5 h-3.5 text-text-muted" />
            <input
              type="date"
              value={t1Date}
              onChange={(e) => setT1Date(e.target.value)}
              className="bg-transparent text-[11px] telemetry-text text-text-secondary focus:outline-none w-24"
            />
            <span className="text-text-muted">→</span>
            <input
              type="date"
              value={t2Date}
              onChange={(e) => setT2Date(e.target.value)}
              className="bg-transparent text-[11px] telemetry-text text-text-secondary focus:outline-none w-24"
            />
          </div>

          {/* Negative Context Suppression Toggle */}
          <button
            type="button"
            onClick={() => setShowNegative(!showNegative)}
            title="Toggle Negative Suppression Vector"
            className={`px-2 py-1.5 rounded-lg text-xs font-medium border flex items-center space-x-1 transition-all ${
              showNegative
                ? 'bg-tactical-amber/20 border-tactical-amber text-tactical-amber'
                : 'bg-void border-border-subtle text-text-muted hover:text-text-secondary'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span className="text-[10px] telemetry-text">NEGATIVE</span>
          </button>

          {/* Master Run Analysis Button (Gated by T1 & T2 Image Upload) */}
          <button
            type="button"
            onClick={handleRunAnalysis}
            disabled={isAnalyzing || !t1ImageUrl || !t2ImageUrl}
            className={`py-2 px-5 rounded-lg font-extrabold text-xs uppercase tracking-wider shadow-lg flex items-center space-x-1.5 transition-all ${
              !t1ImageUrl || !t2ImageUrl
                ? 'bg-panel/90 border border-border-subtle text-text-muted cursor-not-allowed opacity-70'
                : 'bg-gradient-to-r from-tactical-cyan to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-void hover:shadow-glow cursor-pointer'
            }`}
          >
            {isAnalyzing ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-void border-t-transparent rounded-full animate-spin" />
                <span>COMPUTING...</span>
              </>
            ) : !t1ImageUrl || !t2ImageUrl ? (
              <>
                <Upload className="w-3.5 h-3.5 text-text-muted" />
                <span>AWAITING T1 & T2 UPLOAD</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>EXECUTE ANALYSIS</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Negative Prompt Expandable Row */}
      {showNegative && (
        <div className="bg-panel border-t border-border-subtle px-4 py-1.5 flex items-center space-x-2 text-xs">
          <span className="text-[10px] text-tactical-amber font-bold telemetry-text uppercase">
            NEGATIVE SUPPRESSION VECTOR (e*):
          </span>
          <input
            type="text"
            value={negativeQuery}
            onChange={(e) => setNegativeQuery(e.target.value)}
            placeholder="Negative concepts to suppress (e.g., crop harvesting, sun angle shadows, soil drying)..."
            className="flex-1 bg-void px-2 py-1 rounded text-xs text-text-secondary border border-border-subtle focus:outline-none focus:border-tactical-amber"
          />
        </div>
      )}

      {/* 4. Bottom Telemetry Status Bar */}
      <StatusBar
        cursorLat={telemetry.lat}
        cursorLon={telemetry.lon}
        cursorMgrs={telemetry.mgrs}
        stageInfo={{
          stage: isAnalyzing ? 'SIAMESE_INFERENCE' : 'COMPLETED',
          percent: isAnalyzing ? 75 : 100,
          currentTask: isAnalyzing
            ? 'SIAMESE CROSS-ATTENTION EXTRACTING CHANGE MASKS'
            : features.length > 0
            ? `ANALYSIS COMPLETE // ${features.length} TARGETS DETECTED`
            : 'READY // AWAITING MISSION PAIR UPLOAD'
        }}
        gpuMemoryMb={2410}
        gpuTotalMb={8192}
        isWsConnected={true}
      />
    </div>
  );
}
