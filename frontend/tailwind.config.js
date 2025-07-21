/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        blue: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8', // Added for hover states
          200: '#bfdbfe',
        },
        purple: {
          50: '#f5f3ff',
          600: '#7c3aed',
          700: '#6d28d9', // Added for hover states
        },
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
        },
        red: {
          300: '#fca5a5',
        },
        yellow: {
          300: '#fde047',
        },
      },
    },
  },
  plugins: [],
}
