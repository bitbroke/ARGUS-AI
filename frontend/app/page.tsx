import dynamic from 'next/dynamic';
import PipVideoFeed from '@/components/PipVideoFeed';
import TelemetryPanel from '@/components/TelemetryPanel';
import DispatchButton from '@/components/DispatchButton';

const CommandMap = dynamic(() => import('@/components/CommandMap'), {
  ssr: false,
});

export default function Home() {
  return (
    <main className="relative w-screen h-screen overflow-hidden bg-argus-bg flex flex-col">
      {/* Top Header Bar */}
      <header className="h-14 bg-argus-surface border-b border-argus-border flex items-center justify-between px-6 z-20 shadow-md">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 rounded overflow-hidden border border-argus-cyan flex items-center justify-center">
            <img src="/logo.png" alt="Argus AI" className="w-full h-full object-cover" />
          </div>
          <h1 className="text-xl font-bold tracking-widest text-argus-text">ARGUS AI</h1>
          <span className="text-xs font-mono text-argus-text-muted ml-2 px-2 py-0.5 rounded bg-white/5 border border-white/10">v2.1</span>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-argus-green animate-pulse"></div>
            <span className="text-xs font-mono text-argus-text-muted">SYSTEM ONLINE</span>
          </div>
          <div className="h-6 w-px bg-argus-border"></div>
          <div className="text-xs font-mono text-argus-text-muted">ST-GCN + MAMBA</div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 relative flex">
        {/* Map Layer */}
        <div className="absolute inset-0 z-0">
          <CommandMap />
        </div>

        {/* Floating UI - Left Panel */}
        <div className="pointer-events-none absolute left-6 top-6 bottom-6 w-80 z-10 flex flex-col gap-6">
          <div className="pointer-events-auto">
            <PipVideoFeed />
          </div>
          <div className="pointer-events-auto">
            <TelemetryPanel />
          </div>
        </div>

        {/* Floating UI - Bottom Center */}
        <div className="pointer-events-none absolute bottom-8 left-0 right-0 z-10 flex justify-center">
          <div className="pointer-events-auto">
            <DispatchButton />
          </div>
        </div>
      </div>
    </main>
  );
}
