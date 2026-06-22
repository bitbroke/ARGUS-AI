import { AnomalyNode, GraphNode, GraphEdge, RippleResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = {
  async checkHealth() {
    const res = await fetch(`${API_URL}/api/health`);
    return res.json();
  },

  async getAnomalies() {
    const res = await fetch(`${API_URL}/api/anomalies`);
    if (!res.ok) throw new Error("Failed to fetch anomalies");
    return res.json() as Promise<{ anomalies: AnomalyNode[], timestamp: number, frame_index: number }>;
  },

  async getSimulation() {
    const res = await fetch(`${API_URL}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        latitude: 12.9255,
        longitude: 77.6186,
        hour_of_day: 9,
        day_of_week: 1
      })
    });
    if (!res.ok) throw new Error("Failed to fetch simulation");
    return res.json() as Promise<{ nodes: any[], metadata: any }>;
  },

  async getGraph() {
    const res = await fetch(`${API_URL}/api/graph`);
    if (!res.ok) throw new Error("Failed to fetch graph");
    return res.json() as Promise<{ nodes: GraphNode[], edges: GraphEdge[], corridor_name: string, bbox: any }>;
  },

  async getRipple(nodeId: string) {
    const res = await fetch(`${API_URL}/api/ripple`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: nodeId }),
    });
    if (!res.ok) throw new Error("Failed to calculate ripple");
    return res.json() as Promise<RippleResponse>;
  },

  async dispatchAnomaly(nodeId: string) {
    const res = await fetch(`${API_URL}/api/dispatch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: nodeId }),
    });
    if (!res.ok) throw new Error("Failed to dispatch");
    return res.json() as Promise<{ success: boolean, node_id: string, reset_edges: GraphEdge[], message: string }>;
  }
};
