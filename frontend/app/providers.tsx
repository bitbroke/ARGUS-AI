"use client";

import React, { createContext, useState, useContext, ReactNode } from "react";
import { AnomalyStoreContextType, AnomalyNode, GraphNode, GraphEdge, RippleResponse, TelemetryPanelData } from "../lib/types";

const AnomalyStoreContext = createContext<AnomalyStoreContextType | undefined>(undefined);

export function AnomalyStoreProvider({ children }: { children: ReactNode }) {
  const [anomalies, setAnomalies] = useState<AnomalyNode[]>([]);
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>([]);
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [rippleData, setRippleData] = useState<RippleResponse | null>(null);
  const [dispatchState, setDispatchState] = useState<"idle" | "dispatching" | "resolved">("idle");
  const [telemetryPanel, setTelemetryPanel] = useState<TelemetryPanelData>({ delayMinutes: 0, affectedStreets: [] });

  const value = {
    anomalies,
    graphNodes,
    graphEdges,
    selectedNodeId,
    rippleData,
    dispatchState,
    telemetryPanel,
    setAnomalies,
    setGraphNodes,
    setGraphEdges,
    setSelectedNodeId,
    setRippleData,
    setDispatchState,
    setTelemetryPanel,
  };

  return (
    <AnomalyStoreContext.Provider value={value}>
      {children}
    </AnomalyStoreContext.Provider>
  );
}

export function useAnomalyStore() {
  const context = useContext(AnomalyStoreContext);
  if (context === undefined) {
    throw new Error("useAnomalyStore must be used within an AnomalyStoreProvider");
  }
  return context;
}
