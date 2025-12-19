/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // <--- Fondamentale per leggere AIDetectorInterface.jsx
  ],
  theme: {
    extend: {
      // Qui puoi aggiungere estensioni se necessario
    },
  },
  plugins: [],
}