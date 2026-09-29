import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        void: '#06080e',
        surface: '#0b0f19',
        panel: {
          DEFAULT: '#111827',
          elevated: '#162032',
        },
        border: {
          subtle: '#1f2937',
          active: '#374151',
          accent: '#06b6d4',
        },
        tactical: {
          crimson: '#ef4444',
          amber: '#f59e0b',
          emerald: '#10b981',
          cyan: '#06b6d4',
          indigo: '#6366f1',
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 15px #06b6d440',
        'glow-crimson': '0 0 15px #ef444459',
      },
    },
  },
  plugins: [],
};

export default config;
