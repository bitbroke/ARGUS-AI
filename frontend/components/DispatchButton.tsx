"use client";
import { useAnomalyStore } from "@/app/providers";
import { api } from "@/lib/api";

export default function DispatchButton() {
  const { selectedNodeId, dispatchState, setDispatchState, setSelectedNodeId } = useAnomalyStore();

  const handleDispatch = async () => {
    if (!selectedNodeId || dispatchState !== "idle") return;

    setDispatchState("dispatching");
    
    // Simulate API call and 3s animation sequence
    try {
      await api.dispatchAnomaly(selectedNodeId);
      
      // Wait for "dispatching" phase (2.5s total)
      setTimeout(() => {
        setDispatchState("resolved");
        
        // Wait for "resolved" phase (500ms) and reset
        setTimeout(() => {
          setDispatchState("idle");
          setSelectedNodeId(null);
        }, 500);

      }, 2500);
      
    } catch (e) {
      console.error(e);
      setDispatchState("idle");
    }
  };

  if (!selectedNodeId || dispatchState === "resolved") return null;

  return (
    <button
      onClick={handleDispatch}
      disabled={dispatchState !== "idle"}
      className={`
        argus-panel px-8 py-4 font-bold tracking-[0.2em] uppercase transition-all duration-300 flex items-center justify-center gap-3 min-w-[240px]
        ${dispatchState === "idle" 
          ? "bg-argus-surface-hover border-argus-cyan text-argus-cyan hover:bg-argus-cyan hover:text-argus-bg hover:shadow-[0_0_20px_rgba(88,166,255,0.4)]" 
          : "bg-argus-bg text-argus-text-muted border-argus-border opacity-70 cursor-not-allowed"
        }
      `}
    >
      {dispatchState === "idle" ? (
        <span>DISPATCH UNIT</span>
      ) : (
        <>
          <svg className="animate-spin h-5 w-5 text-argus-text-muted" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span>DISPATCHING...</span>
        </>
      )}
    </button>
  );
}
