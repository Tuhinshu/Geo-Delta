import React from 'react';
import type { Metadata } from 'next';
import 'maplibre-gl/dist/maplibre-gl.css';
import './globals.css';

export const metadata: Metadata = {
  title: 'GeoDelta | Automated GEOINT Intelligence Platform',
  description:
    'Semantic Retrieval and Multi-Temporal Satellite Change Analysis for Defence Intelligence Operations (MoD SIH26227).',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full">
      <body className="bg-void text-text-primary antialiased h-screen flex flex-col overflow-hidden">
        {/* Top Military Classification Ribbon */}
        <header className="w-full bg-[#f59e0b1f] border-b border-[#f59e0b59] py-0.5 px-4 text-center select-none z-50 shrink-0">
          <span className="telemetry-text text-[11px] font-bold text-[#fbbf24] tracking-widest uppercase">
            RESTRICTED // GEOINT ASSESSMENT // FOR OFFICIAL USE ONLY // SIH26227
          </span>
        </header>
        <main className="flex-1 flex flex-col relative overflow-hidden min-h-0">
          {children}
        </main>
      </body>
    </html>
  );
}
