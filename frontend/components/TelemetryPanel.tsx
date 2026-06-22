"use client";
import { useAnomalyStore } from "@/app/providers";

export default function TelemetryPanel() {
  const { selectedNodeId, telemetryPanel, dispatchState } = useAnomalyStore();

  if (!selectedNodeId || dispatchState === "resolved" || telemetryPanel.delayMinutes === 0) {
    return (
      <div className="argus-panel p-6 w-full min-h-32 flex flex-col items-center justify-center text-argus-text-muted font-mono text-sm gap-2">
        <svg className="w-6 h-6 text-argus-text-muted/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span>NO ACTIVE ANOMALIES</span>
      </div>
    );
  }

  return (
    <div className="argus-panel p-6 w-full flex flex-col gap-5">
      <div className="flex items-center justify-between border-b border-argus-border pb-3">
        <h3 className="text-xs font-bold uppercase tracking-widest text-argus-text">
          Network Impact
        </h3>
        <div className="w-2 h-2 rounded-full bg-argus-orange animate-pulse"></div>
      </div>
      
      <div className="bg-black/20 rounded p-4 border border-argus-border">
        <span className="text-[10px] uppercase tracking-widest text-argus-text-muted block mb-1">
          Predicted Delay
        </span>
        <div className="flex items-baseline gap-1">
          <span className="text-4xl font-mono font-medium text-argus-orange tracking-tight">
            +{telemetryPanel.delayMinutes.toFixed(1)}
          </span>
          <span className="text-sm font-mono text-argus-orange/70">
            min
          </span>
        </div>
      </div>
      
      {telemetryPanel.affectedStreets.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-[10px] font-bold text-argus-text-muted uppercase tracking-widest">
            Affected Corridors
          </span>
          <div className="flex flex-wrap gap-2">
            {telemetryPanel.affectedStreets.map((street) => (
              <span key={street} className="text-xs font-mono bg-argus-surface-hover px-2.5 py-1.5 rounded text-argus-text border border-argus-border">
                {street}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
