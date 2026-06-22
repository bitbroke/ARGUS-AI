"use client";
import { useEffect, useState } from "react";
import ReactMap from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import DeckGL from "@deck.gl/react";
import { PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import { useAnomalyStore } from "@/app/providers";
import { api } from "@/lib/api";

// Map initial state — centered on Koramangala, Bangalore
const INITIAL_VIEW_STATE = {
  longitude: 77.6186,
  latitude: 12.9255,
  zoom: 14.5,
  pitch: 50,
  bearing: -15,
};

export default function CommandMap() {
  const {
    anomalies,
    graphEdges,
    graphNodes,
    setAnomalies,
    setGraphEdges,
    setGraphNodes,
    selectedNodeId,
    setSelectedNodeId,
    dispatchState,
    setTelemetryPanel
  } = useAnomalyStore();

  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

  // Initial load — fetch the graph topology from the backend
  useEffect(() => {
    async function init() {
      try {
        const data = await api.getGraph();
        setGraphNodes(data.nodes);
        setGraphEdges(data.edges);
        
        // Center map on actual graph bbox
        if (data.bbox && data.bbox.min_lat) {
          const lat = (data.bbox.min_lat + data.bbox.max_lat) / 2;
          const lon = (data.bbox.min_lon + data.bbox.max_lon) / 2;
          setViewState(v => ({ ...v, latitude: lat, longitude: lon }));
        }
      } catch (e) {
        console.error("Failed to load graph", e);
      }
    }
    init();
  }, []);

  // Polling loop — ask the ST-GCN predictions and live YOLO anomalies, then merge
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        // 1. Fetch simulation anomalies
        let simAnomalies: any[] = [];
        try {
          const simRes = await api.getSimulation();
          simAnomalies = simRes.nodes
            .filter((n: any) => n.congested)
            .map((n: any) => ({
              node_id: n.node_id,
              lat: n.lat,
              lon: n.lon,
              status: 'anomaly' as const,
              predicted_volume: n.predicted_volume
            }));
        } catch (e) {
          console.error("Failed to poll simulation", e);
        }

        // 2. Fetch live YOLO anomalies
        let liveAnomalies: any[] = [];
        try {
          const liveRes = await api.getAnomalies();
          liveAnomalies = (liveRes.anomalies || []).map((a: any) => ({
            ...a,
            status: 'anomaly' as const
          }));
        } catch (e) {
          console.error("Failed to poll live anomalies", e);
        }

        // 3. Merge anomalies by node_id to avoid duplicates
        const mergedMap = new Map<string, any>();
        
        simAnomalies.forEach((a) => mergedMap.set(a.node_id, a));
        liveAnomalies.forEach((a) => {
          mergedMap.set(a.node_id, {
            ...mergedMap.get(a.node_id),
            ...a
          });
        });

        setAnomalies(Array.from(mergedMap.values()));
      } catch (e) {
        console.error("Failed in anomalies merging loop", e);
      }
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Deck.gl Layers
  const edgesLayer = new PathLayer({
    id: "street-edges",
    data: graphEdges,
    pickable: false,
    widthMinPixels: 3,
    widthMaxPixels: 8,
    getWidth: 5,
    getPath: (d: any) => d.path,
    getColor: (d: any) => d.color_rgb || [88, 166, 255],
    capRounded: true,
    jointRounded: true,
    transitions: {
      getColor: { type: "interpolation", duration: 2000 }
    },
    updateTriggers: {
      getColor: graphEdges.map((e: any) => e.color_rgb?.join(',')).join('|')
    }
  });

  const nodesLayer = new ScatterplotLayer({
    id: "anomaly-nodes",
    data: anomalies.filter((a: any) => a.status === 'anomaly'),
    pickable: true,
    radiusMinPixels: 8,
    radiusMaxPixels: 20,
    getPosition: (d: any) => [d.lon, d.lat],
    getRadius: (d: any) => (dispatchState === 'dispatching' && d.node_id === selectedNodeId ? 20 : 15),
    getFillColor: [247, 120, 186],
    stroked: true,
    getLineColor: [230, 237, 243, 120],
    lineWidthMinPixels: 1,
    onClick: async ({ object }: any) => {
      if (object && dispatchState === 'idle') {
        setSelectedNodeId(object.node_id);
        
        // Fly to node
        setViewState(v => ({
          ...v,
          longitude: object.lon,
          latitude: object.lat,
          zoom: 16,
          transitionDuration: 1000
        }));

        try {
          const res = await api.getRipple(object.node_id);
          
          // Update edges colors from ripple calculation
          if (res.affected_edges) {
            setGraphEdges(res.affected_edges);
          }
          setTelemetryPanel({
            delayMinutes: res.total_delay_minutes || 0,
            affectedStreets: res.affected_street_names || []
          });
        } catch (e) {
          console.error("Ripple failed", e);
        }
      }
    },
    updateTriggers: {
      getRadius: dispatchState,
      getFillColor: dispatchState
    }
  });

  return (
    <DeckGL
      initialViewState={viewState}
      controller={true}
      layers={[edgesLayer, nodesLayer]}
      onViewStateChange={({ viewState }: any) => setViewState(viewState)}
    >
      <ReactMap
        attributionControl={false}
        mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        reuseMaps
        preventStyleDiffing
      />
    </DeckGL>
  );
}
