import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "RETINA-Q | Quantum Clinical Diagnostic System",
    description:
        "High-Precision Clinical Retinal Diagnosis using Hybrid Quantum-Classical Neural Networks.",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <body className="antialiased font-sans bg-slate-50 text-slate-900">
                <div className="min-h-screen flex flex-col">
                    {/* Header: Clean clinical header */}
                    <header className="sticky top-0 z-50 bg-white border-b border-slate-200 shadow-sm">
                        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 sm:gap-0">
                            <a href="/" className="flex items-center gap-4">
                                {/* Clinical logo boxed */}
                                <div className="w-10 h-10 rounded-lg bg-sky-500 flex items-center justify-center shadow-md">
                                    <svg
                                        width="20"
                                        height="20"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="#ffffff"
                                        strokeWidth="2.5"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                    >
                                        <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                                        <rect x="2" y="2" width="20" height="20" rx="4" />
                                    </svg>
                                </div>
                                <div className="flex flex-col">
                                    <h1 className="text-xl font-bold tracking-tight text-slate-900 leading-none">RETINA-Q</h1>
                                    <p className="text-[11px] font-semibold text-teal-600 mt-1 uppercase tracking-wider">
                                        Clinical AI Engine
                                    </p>
                                </div>
                            </a>

                            <nav className="flex items-center gap-1 sm:gap-2">
                                <a
                                    href="/"
                                    className="px-3 py-2 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-md transition-colors"
                                >
                                    Diagnosis
                                </a>
                                <a
                                    href="/dashboard"
                                    className="px-3 py-2 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-md transition-colors"
                                >
                                    Metrics
                                </a>
                                <div className="ml-2 flex items-center gap-2 px-3 py-1.5 bg-emerald-50 border border-emerald-200 rounded-full">
                                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse-soft"></div>
                                    <span className="text-[11px] font-semibold text-emerald-700 uppercase tracking-wide">
                                        System Online
                                    </span>
                                </div>
                            </nav>
                        </div>
                    </header>

                    {/* Main Content */}
                    <main className="flex-1 overflow-x-hidden">{children}</main>

                    {/* Footer: Clean footer */}
                    <footer className="bg-white border-t border-slate-200 mt-auto">
                        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
                            <p>© 2026 RETINA-Q Clinical AI Systems. All rights reserved.</p>
                            <p className="font-mono bg-slate-100 text-slate-600 px-2 py-1 rounded">v2.2.0-CLINICAL</p>
                        </div>
                    </footer>
                </div>
            </body>
        </html>
    );
}
