/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  transpilePackages: ['react-map-gl', '@vis.gl/react-maplibre', '@vis.gl/react-mapbox'],
};

export default nextConfig;
