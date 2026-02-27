"use client";

import { useState, useEffect } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface SystemStatus {
    status: string;
    service: string;
    version: string;
}

interface StatCard {
    label: string;
    value: string;
    sub: string;
    icon: React.ReactNode;
    color: string;
}

export default function DashboardPage() {
    const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
    const [statusError, setStatusError] = useState(false);

    useEffect(() => {
        fetch(`${API_BASE}/health`)
            .then((r) => r.json())
            .then(setSystemStatus)
            .catch(() => setStatusError(true));
    }, []);

    const stats: StatCard[] = [
        {
            label: "OCT Model",
            value: "8-Qubit",
            sub: "Quantum Circuit Classifier",
            icon: (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="12" cy="12" r="3" />
                    <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
                </svg>
            ),
            color: "bg-muted text-foreground",
        },
        {
            label: "Fundus Model",
            value: "4-Qubit",
            sub: "EfficientNet-B0 Hybrid",
            icon: (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z" />
                    <circle cx="12" cy="12" r="3" />
                </svg>
            ),
            color: "bg-muted text-foreground",
        },
        {
            label: "Segmentation",
            value: "U-Net",
            sub: "BCE + Dice + Tversky Loss",
            icon: (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <path d="M3 12h18M12 3v18" />
                </svg>
            ),
            color: "bg-muted text-foreground",
        },
        {
            label: "Explainability",
            value: "Grad-CAM",
            sub: "Feature Importance + Grad-CAM",
            icon: (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                    <path d="M2 17l10 5 10-5" />
                    <path d="M2 12l10 5 10-5" />
                </svg>
            ),
            color: "bg-muted text-foreground",
        },
    ];

    const performanceTargets = [
        { metric: "OCT Accuracy", target: "≥ 92%", icon: "🎯" },
        { metric: "Fundus Accuracy", target: "≥ 93%", icon: "🎯" },
        { metric: "Dice Score", target: "≥ 0.90", icon: "📐" },
        { metric: "ROC-AUC", target: "≥ 0.95", icon: "📈" },
        { metric: "Inference Time", target: "< 2 sec", icon: "⚡" },
    ];

    const architecture = [
        {
            title: "OCT Pipeline",
            steps: [
                "Feature Extraction (64 features)",
                "8-Qubit Quantum Circuit",
                "Classical Dense Layers",
                "Binary Output + Heatmap",
            ],
        },
        {
            title: "Fundus Pipeline",
            steps: [
                "EfficientNet-B0 Backbone",
                "Adaptive Pooling",
                "4-Qubit Quantum Layer",
                "Feature Concatenation",
                "Binary Output + Grad-CAM",
            ],
        },
        {
            title: "Segmentation Pipeline",
            steps: [
                "Green Channel Extraction",
                "CLAHE Enhancement",
                "U-Net Inference",
                "Morphological Post-Processing",
                "Mask Output",
            ],
        },
    ];

    return (
        <div className="max-w-7xl mx-auto px-6 py-10">
            {/* Header */}
            <div className="mb-10 animate-slide-up">
                <h2 className="text-4xl font-black uppercase tracking-tighter text-foreground mb-3">
                    System Dashboard
                </h2>
                <p className="text-muted-foreground uppercase tracking-widest text-xs">
                    Monitor RETINA-Q system status, model architectures, and performance
                    targets.
                </p>
            </div>

            {/* System Status */}
            <div
                className="glass-card p-6 mb-8 animate-slide-up"
                style={{ animationDelay: "0.05s" }}
            >
                <div className="flex items-center justify-between border-b border-border pb-4 mb-4">
                    <div className="flex items-center gap-4">
                        <div
                            className={`w-2 h-2 rounded-full ${systemStatus ? "bg-foreground animate-pulse" : statusError ? "bg-red-500" : "bg-muted-foreground animate-pulse"
                                }`}
                        ></div>
                        <div>
                            <p className="font-bold uppercase tracking-wider text-foreground text-sm">
                                {systemStatus
                                    ? `${systemStatus.service} v${systemStatus.version}`
                                    : statusError
                                        ? "Backend Offline"
                                        : "Connecting..."}
                            </p>
                            <p className="text-xs uppercase tracking-widest text-muted-foreground mt-1">
                                {systemStatus
                                    ? "All systems operational"
                                    : statusError
                                        ? "Unable to reach backend API"
                                        : "Checking system status..."}
                            </p>
                        </div>
                    </div>
                    <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground border border-border px-2 py-1">
                        {API_BASE}
                    </span>
                </div>
            </div>

            {/* Model Cards */}
            <div
                className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-10 animate-slide-up"
                style={{ animationDelay: "0.1s" }}
            >
                {stats.map((s) => (
                    <div
                        key={s.label}
                        className="glass-card p-5 hover:bg-muted transition-all group"
                    >
                        <div
                            className={`w-10 h-10 rounded-sm ${s.color} flex items-center justify-center mb-4 transition-colors`}
                        >
                            {s.icon}
                        </div>
                        <p className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">{s.label}</p>
                        <p className="text-xl font-bold font-mono text-foreground mb-0.5">{s.value}</p>
                        <p className="text-xs text-muted-foreground">{s.sub}</p>
                    </div>
                ))}
            </div>

            <div className="grid lg:grid-cols-2 gap-8">
                {/* Performance Targets */}
                <div
                    className="glass-card p-6 animate-slide-up"
                    style={{ animationDelay: "0.15s" }}
                >
                    <h3 className="font-bold text-foreground uppercase tracking-widest text-sm mb-5 border-b border-border pb-3">
                        Performance Targets
                    </h3>
                    <div className="space-y-4">
                        {performanceTargets.map((p) => (
                            <div
                                key={p.metric}
                                className="flex items-center justify-between p-3 bg-background border border-border"
                            >
                                <div className="flex items-center gap-3">
                                    <span className="text-lg opacity-80 grayscale">{p.icon}</span>
                                    <span className="text-xs uppercase tracking-wider text-foreground">{p.metric}</span>
                                </div>
                                <span className="font-mono font-bold text-foreground text-sm">
                                    {p.target}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Architecture Pipelines */}
                <div
                    className="glass-card p-6 animate-slide-up flex flex-col"
                    style={{ animationDelay: "0.2s" }}
                >
                    <h3 className="font-bold text-foreground uppercase tracking-widest text-sm mb-5 border-b border-border pb-3">
                        Architecture Pipelines
                    </h3>
                    <div className="space-y-8 flex-1">
                        {architecture.map((pipeline) => (
                            <div key={pipeline.title}>
                                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground mb-3">
                                    {pipeline.title}
                                </p>
                                <div className="flex flex-wrap gap-2">
                                    {pipeline.steps.map((step, i) => (
                                        <div key={i} className="flex items-center gap-2">
                                            <span className="px-3 py-1.5 text-[10px] uppercase font-mono tracking-widest border border-border bg-muted text-foreground">
                                                {step}
                                            </span>
                                            {i < pipeline.steps.length - 1 && (
                                                <svg
                                                    width="12"
                                                    height="12"
                                                    viewBox="0 0 24 24"
                                                    fill="none"
                                                    stroke="currentColor"
                                                    strokeWidth="2"
                                                    className="text-muted-foreground flex-shrink-0"
                                                >
                                                    <polyline points="9 18 15 12 9 6" />
                                                </svg>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* API Endpoints */}
            <div
                className="glass-card p-6 mt-8 animate-slide-up"
                style={{ animationDelay: "0.25s" }}
            >
                <h3 className="font-bold text-foreground uppercase tracking-widest text-sm mb-5 border-b border-border pb-3">API Endpoints</h3>
                <div className="grid sm:grid-cols-3 gap-4">
                    {[
                        {
                            method: "POST",
                            path: "/api/predict/oct",
                            desc: "OCT classification with 8-qubit quantum circuit",
                        },
                        {
                            method: "POST",
                            path: "/api/predict/fundus",
                            desc: "Fundus classification + conditional segmentation",
                        },
                        {
                            method: "POST",
                            path: "/api/segment",
                            desc: "Standalone U-Net macular segmentation",
                        },
                    ].map((ep) => (
                        <div
                            key={ep.path}
                            className="p-4 bg-background border border-border"
                        >
                            <div className="flex items-center gap-2 mb-3">
                                <span className="px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest bg-foreground text-background">
                                    {ep.method}
                                </span>
                                <code className="text-[10px] uppercase font-bold tracking-widest text-foreground font-mono">
                                    {ep.path}
                                </code>
                            </div>
                            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{ep.desc}</p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
