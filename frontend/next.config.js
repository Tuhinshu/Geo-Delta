/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  swcMinify: true,
  transpilePackages: ['@deck.gl/core', '@deck.gl/layers', '@deck.gl/geo-layers', '@deck.gl/react'],
};

module.exports = nextConfig;
