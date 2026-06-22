import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        'ucc-bg': '#0B0F19',
        'ucc-cyan': '#00F3FF',
        'ucc-magenta': '#FF00FF',
        'ucc-orange': '#FF5E00',
        'ucc-surface': 'rgba(11, 15, 25, 0.7)',
        // Argus brand colors
        'argus-bg': '#0D1117',
        'argus-surface': '#161B22',
        'argus-surface-hover': '#21262D',
        'argus-border': '#30363D',
        'argus-cyan': '#58A6FF',
        'argus-magenta': '#F778BA',
        'argus-orange': '#F0883E',
        'argus-green': '#3FB950',
        'argus-text': '#E6EDF3',
        'argus-text-muted': '#8B949E',
      },
    },
  },
  plugins: [],
};
export default config;
