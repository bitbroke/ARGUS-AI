export interface AnomalyNode {
  node_id: string;
  lat: number;
  lon: number;
  status: "anomaly" | "healthy" | "dispatched";
  confidence: number;
  detected_at: number;
  label: string;
  bbox_x: number;
  bbox_y: number;
  bbox_w: number;
  bbox_h: number;
}

export interface GraphNode {
  node_id: string;
  lat: number;
  lon: number;
  street_name: string;
  flow_state: "healthy" | "bottleneck" | "anomaly";
  delay_minutes: number;
}

export interface GraphEdge {
  edge_id: string;
  source_node_id: string;
  target_node_id: string;
  street_name: string;
  length_m: number;
  flow_state: "healthy" | "bottleneck" | "ripple";
  delay_minutes: number;
  color_rgb: [number, number, number];
}

export interface RippleResponse {
  origin_node_id: string;
  affected_edges: GraphEdge[];
  affected_nodes: GraphNode[];
  total_delay_minutes: number;
  affected_street_names: string[];
  calculation_method: string;
}

export interface TelemetryPanelData {
  delayMinutes: number;
  affectedStreets: string[];
}

export interface AnomalyStoreContextType {
  anomalies: AnomalyNode[];
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
  selectedNodeId: string | null;
  rippleData: RippleResponse | null;
  dispatchState: "idle" | "dispatching" | "resolved";
  telemetryPanel: TelemetryPanelData;
  setAnomalies: (anomalies: AnomalyNode[]) => void;
  setGraphNodes: (nodes: GraphNode[]) => void;
  setGraphEdges: (edges: GraphEdge[]) => void;
  setSelectedNodeId: (id: string | null) => void;
  setRippleData: (data: RippleResponse | null) => void;
  setDispatchState: (state: "idle" | "dispatching" | "resolved") => void;
  setTelemetryPanel: (data: TelemetryPanelData) => void;
}
