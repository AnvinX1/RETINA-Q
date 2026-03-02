import type { Metadata } from "next";
import "./globals.css";
import "katex/dist/katex.min.css";

export const metadata: Metadata = {
    title: "RETINA-Q | Quantum Retinal Diagnostic System",
    description:
        "Hybrid Quantum-Classical Neural Network for Retinal Disease Diagnosis — OCT & Fundus Analysis",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <body className="antialiased font-sans bg-white text-foreground">
                <div className="min-h-screen flex flex-col">
                    {/* Header */}
                    <header className="sticky top-0 z-50 bg-white border-b border-border">
                        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
                            <a href="/" className="flex items-center gap-3">
                                <div className="w-7 h-7 rounded bg-foreground flex items-center justify-center">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5">
                                        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z" />
                                        <circle cx="12" cy="12" r="3" />
                                    </svg>
                                </div>
                                <div className="flex items-baseline gap-2">
                                    <span className="text-sm font-bold tracking-tight font-mono">RETINA-Q</span>
                                    <span className="text-[10px] font-mono text-muted-foreground">v2.0.0</span>
                                </div>
                            </a>

                            <nav className="flex items-center gap-1">
                                <a
                                    href="/"
                                    className="px-3 py-1.5 text-xs font-medium font-mono text-muted-foreground hover:text-foreground hover:bg-muted rounded transition-colors"
                                >
                                    diagnose
                                </a>
                                <a
                                    href="/dashboard"
                                    className="px-3 py-1.5 text-xs font-medium font-mono text-muted-foreground hover:text-foreground hover:bg-muted rounded transition-colors"
                                >
                                    system
                                </a>
                                <a
                                    href="/docs"
                                    className="px-3 py-1.5 text-xs font-medium font-mono text-muted-foreground hover:text-foreground hover:bg-muted rounded transition-colors"
                                >
                                    docs
                                </a>
                                <div className="ml-3 flex items-center gap-1.5 px-2 py-1 border border-border rounded text-[10px] font-mono text-muted-foreground">
                                    <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-soft"></div>
                                    online
                                </div>
                            </nav>
                        </div>
                    </header>

                    <main className="flex-1 overflow-x-hidden">{children}</main>

                    <footer className="border-t border-border mt-auto">
                        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-4 flex items-center justify-between text-[10px] font-mono text-muted-foreground">
                            <p>&copy; 2026 RETINA-Q &mdash; Quantum Clinical AI</p>
                            <p>PennyLane + PyTorch + Next.js</p>
                        </div>
                    </footer>
                </div>
            </body>
        </html>
    );
}
