import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "RETINA-Q | Quantum-Enhanced Retinal Diagnosis",
    description:
        "Hybrid Quantum-Classical Multi-Modal Retinal Disease Diagnosis System using OCT and Fundus imaging with AI explainability.",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className="dark">
            <body className="antialiased">
                <div className="min-h-screen flex flex-col bg-background text-foreground">
                    {/* Header */}
                    <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-md">
                        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                            <a href="/" className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-sm bg-foreground flex items-center justify-center">
                                    <svg
                                        width="20"
                                        height="20"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="var(--background)"
                                        strokeWidth="2"
                                        strokeLinecap="square"
                                        strokeLinejoin="miter"
                                    >
                                        <rect x="3" y="3" width="18" height="18" />
                                        <circle cx="12" cy="12" r="3" />
                                        <line x1="3" y1="12" x2="6" y2="12" />
                                        <line x1="18" y1="12" x2="21" y2="12" />
                                        <line x1="12" y1="3" x2="12" y2="6" />
                                        <line x1="12" y1="18" x2="12" y2="21" />
                                    </svg>
                                </div>
                                <div className="flex flex-col">
                                    <h1 className="text-lg font-black tracking-tighter uppercase text-foreground leading-none">RETINA-Q</h1>
                                    <p className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground mt-1">
                                        Diagnostic Engine
                                    </p>
                                </div>
                            </a>

                            <nav className="flex items-center gap-2">
                                <a
                                    href="/"
                                    className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-all"
                                >
                                    Diagnose
                                </a>
                                <a
                                    href="/dashboard"
                                    className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-all"
                                >
                                    Dashboard
                                </a>
                                <div className="ml-6 flex items-center gap-2 px-3 py-1 bg-muted border border-border rounded-sm">
                                    <div className="w-1.5 h-1.5 rounded-full bg-foreground animate-pulse"></div>
                                    <span className="text-[10px] uppercase font-bold tracking-widest text-foreground">
                                        Online
                                    </span>
                                </div>
                            </nav>
                        </div>
                    </header>

                    {/* Main Content */}
                    <main className="flex-1">{children}</main>

                    {/* Footer */}
                    <footer className="border-t border-border mt-auto">
                        <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between text-xs text-muted-foreground">
                            <p className="uppercase tracking-widest">© 2025 RETINA-Q Diagnostic AI</p>
                            <p className="font-mono">v1.1.0-B&W</p>
                        </div>
                    </footer>
                </div>
            </body>
        </html>
    );
}
