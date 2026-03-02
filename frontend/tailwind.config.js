/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        './app/**/*.{js,ts,jsx,tsx,mdx}',
        './components/**/*.{js,ts,jsx,tsx,mdx}',
        './lib/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                mono: ['IBM Plex Mono', 'monospace'],
            },
            colors: {
                background: 'var(--background)',
                foreground: 'var(--foreground)',
                card: 'var(--card)',
                'card-foreground': 'var(--card-foreground)',
                border: 'var(--border)',
                muted: 'var(--muted)',
                'muted-foreground': 'var(--muted-foreground)',
                primary: 'var(--primary)',
                'primary-foreground': 'var(--primary-foreground)',
                accent: 'var(--accent)',
                'accent-foreground': 'var(--accent-foreground)',
                ring: 'var(--ring)',
                success: 'var(--success)',
                warning: 'var(--warning)',
                danger: 'var(--danger)',
            },
            keyframes: {
                pulseSoft: {
                    '0%, 100%': { opacity: '1' },
                    '50%': { opacity: '0.5' },
                },
                slideUp: {
                    from: { opacity: '0', transform: 'translateY(8px)' },
                    to: { opacity: '1', transform: 'translateY(0)' },
                },
                fadeIn: {
                    from: { opacity: '0' },
                    to: { opacity: '1' },
                },
            },
            animation: {
                'pulse-soft': 'pulseSoft 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'slide-up': 'slideUp 0.3s ease-out forwards',
                'fade-in': 'fadeIn 0.2s ease-out forwards',
            },
        },
    },
    plugins: [],
};
