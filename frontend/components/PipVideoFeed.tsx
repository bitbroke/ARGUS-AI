"use client";

export default function PipVideoFeed() {
  return (
    <div className="argus-panel overflow-hidden w-full h-48 relative group">
      {/* 
        For MVP, we serve loop.mp4 directly from public directory.
        Since video plays locally, latency is 0. 
      */}
      <video 
        autoPlay 
        muted 
        loop 
        playsInline
        className="w-full h-full object-cover opacity-80 mix-blend-screen group-hover:opacity-100 transition-opacity"
        src="/loop.mp4" 
      />
      
      {/* Overlay controls */}
      <div className="absolute inset-0 border-[4px] border-argus-surface/50 pointer-events-none"></div>
      
      <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm px-2 py-1 rounded text-[10px] font-bold font-mono text-argus-text tracking-wider border border-argus-border">
        YOLOv8 DETECTOR
      </div>
      <div className="absolute top-2 right-2 flex items-center gap-2 bg-black/60 backdrop-blur-sm px-2 py-1 rounded border border-argus-border">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
        </span>
        <span className="text-[10px] font-mono font-bold text-argus-text">LIVE</span>
      </div>
      
      {/* Reticle UI */}
      <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
        <div className="w-12 h-12 border border-argus-cyan/30 rounded-full flex items-center justify-center">
          <div className="w-1 h-1 bg-argus-cyan rounded-full"></div>
        </div>
      </div>
    </div>
  );
}
