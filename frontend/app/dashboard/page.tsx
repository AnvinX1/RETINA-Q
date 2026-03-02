"use client";

import { useState, useEffect } from "react";
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
    ResponsiveContainer, AreaChart, Area,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface SystemStatus {
    status: string;
    service: string;
    version: string;
}

const modelSpecs = [
    {
        name: "OCT Classifier",
        type: "Quantum",
        qubits: 8,
        layers: 8,
        params: "8×8×2 = 128 quantum params",
        input: "64 statistical features",
        output: "Normal / CSR",
    },
    {
        name: "Fundus Classifier",
        type: "Hybrid",
        qubits: 4,
        layers: 6,
        params: "~4.0M (EfficientNet) + 48 quantum",
        input: "224×224 RGB image",
        output: "Healthy / CSCR",
    },
    {
        name: "U-Net Segmentation",
        type: "Classical",
        qubits: 0,
        layers: 9,
        params: "~31M parameters",
        input: "256×256 green-channel CLAHE",
        output: "Binary mask",
    },
];

const performanceTargets = [
    { metric: "OCT Accuracy", target: "≥ 92%", current: 94 },
    { metric: "Fundus Accuracy", target: "≥ 93%", current: 95 },
    { metric: "Dice Score", target: "≥ 0.90", current: 91 },
    { metric: "ROC-AUC", target: "≥ 0.95", current: 96 },
    { metric: "Inference Time", target: "< 2s", current: 85 },
];

const quantumCircuitData = [
    { layer: "L1", depth: 2, entanglement: 8 },
    { layer: "L2", depth: 4, entanglement: 8 },
    { layer: "L3", depth: 6, entanglement: 8 },
    { layer: "L4", depth: 8, entanglement: 8 },
    { layer: "L5", depth: 10, entanglement: 8 },
    { layer: "L6", depth: 12, entanglement: 8 },
    { layer: "L7", depth: 14, entanglement: 8 },
    { layer: "L8", depth: 16, entanglement: 8 },
];

const pipelineSteps = [
    {
        title: "OCT Pipeline",
        steps: [
            { name: "Feature Extraction", detail: "64 features: gradient, histogram, LBP, texture, moments" },
            { name: "Pre-Network", detail: "Linear(64→32) → ReLU → BN → Dropout → Linear(32→8) → Tanh" },
            { name: "Quantum Circuit", detail: "8-qubit, 8-layer variational circuit with CNOT ring entanglement" },
            { name: "Post-Network", detail: "Linear(8→32) → ReLU → BN → Dropout → Linear(32→16) → Linear(16→1)" },
            { name: "Explainability", detail: "Gradient-based feature importance → 8×8 spatial heatmap" },
        ],
    },
    {
        title: "Fundus Pipeline",
        steps: [
            { name: "EfficientNet-B0", detail: "Pretrained backbone → 1,280-dim feature extraction" },
            { name: "Reduction", detail: "Linear(1280→64) → ReLU → Dropout → Linear(64→4) → Tanh" },
            { name: "Quantum Layer", detail: "4-qubit, 6-layer circuit with RY/RZ rotations + CNOT ring" },
            { name: "Fusion", detail: "Concatenate classical(4) + quantum(4) = 8-dim" },
            { name: "Classifier", detail: "Linear(8→32) → ReLU → BN → Linear(32→16) → Linear(16→1)" },
            { name: "Grad-CAM", detail: "Hook into _conv_head → gradient-weighted class activation" },
        ],
    },
    {
        title: "Segmentation Pipeline",
        steps: [
            { name: "Preprocessing", detail: "Green channel extraction → CLAHE (clip=2.0, grid=8×8)" },
            { name: "U-Net Encoder", detail: "4 levels: 1→64→128→256→512 with MaxPool2d" },
            { name: "Bottleneck", detail: "ConvBlock(512→1024) with double convolution" },
            { name: "U-Net Decoder", detail: "4 levels with skip connections + ConvTranspose2d" },
            { name: "Post-process", detail: "Binarize → largest connected component → morphological close/open" },
        ],
    },
];

const endpoints = [
    { method: "POST", path: "/api/predict/oct", desc: "OCT quantum classification" },
    { method: "POST", path: "/api/predict/fundus", desc: "Fundus hybrid classification + segmentation" },
    { method: "POST", path: "/api/segment", desc: "Standalone U-Net macular segmentation" },
    { method: "GET", path: "/api/jobs/{id}", desc: "Poll async job status" },
    { method: "GET", path: "/api/jobs/{id}/stream", desc: "SSE real-time job updates" },
    { method: "POST", path: "/api/feedback", desc: "Doctor verdict + correction logging" },
    { method: "GET", path: "/health", desc: "System health check" },
];

export default function DashboardPage() {
    const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
    const [statusError, setStatusError] = useState(false);

    useEffect(() => {
        fetch(`${API_BASE}/health`)
            .then((r) => r.json())
            .then(setSystemStatus)
            .catch(() => setStatusError(true));
    }, []);

    return (
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-6 space-y-6">
            {/* Header */}
            <div className="animate-slide-up border-b border-border pb-4">
                <div className="flex items-center gap-2 mb-1">
                    <span className="text-accent font-mono text-xs font-bold">$</span>
                    <h2 className="text-lg font-bold font-mono tracking-tight">retina-q system</h2>
                </div>
                <p className="text-xs text-muted-foreground font-mono pl-4">
                    Model specifications, architecture details, and system status
                </p>
            </div>

            {/* Status bar */}
            <div className="t-card p-4 animate-slide-up" style={{ animationDelay: "0.05s" }}>
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full ${systemStatus ? "bg-accent animate-pulse-soft" : statusError ? "bg-danger" : "bg-muted-foreground animate-pulse-soft"}`}></div>
                        <div className="font-mono text-xs">
                            <span className="font-bold">
                                {systemStatus ? `${systemStatus.service} v${systemStatus.version}` : statusError ? "Backend offline" : "Connecting..."}
                            </span>
                            <span className="text-muted-foreground ml-2">
                                {systemStatus ? "all systems operational" : statusError ? "unable to reach backend" : "checking..."}
                            </span>
                        </div>
                    </div>
                    <span className="text-[9px] font-mono text-muted-foreground border border-border px-2 py-0.5 rounded">{API_BASE}</span>
                </div>
            </div>

            {/* Model specs */}
            <div className="grid md:grid-cols-3 gap-4 animate-slide-up" style={{ animationDelay: "0.1s" }}>
                {modelSpecs.map((m) => (
                    <div key={m.name} className="t-card p-4 hover:bg-muted transition-colors">
                        <div className="flex items-center justify-between mb-3">
                            <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">{m.name}</p>
                            <span className="text-[9px] font-mono px-1.5 py-0.5 bg-muted border border-border rounded">{m.type}</span>
                        </div>
                        {m.qubits > 0 && (
                            <p className="text-2xl font-mono font-bold">{m.qubits}<span className="text-xs text-muted-foreground ml-1">qubits</span></p>
                        )}
                        {m.qubits === 0 && (
                            <p className="text-2xl font-mono font-bold">U-Net</p>
                        )}
                        <div className="mt-3 space-y-1 text-[10px] font-mono text-muted-foreground">
                            <div className="flex justify-between"><span>layers</span><span className="text-foreground">{m.layers}</span></div>
                            <div className="flex justify-between"><span>params</span><span className="text-foreground">{m.params}</span></div>
                            <div className="flex justify-between"><span>input</span><span className="text-foreground">{m.input}</span></div>
                            <div className="flex justify-between"><span>output</span><span className="text-foreground">{m.output}</span></div>
                        </div>
                    </div>
                ))}
            </div>

            <div className="grid lg:grid-cols-2 gap-6 items-stretch">
                {/* Performance targets */}
                <div className="t-card p-4 animate-slide-up flex flex-col" style={{ animationDelay: "0.15s" }}>
                    <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-4">Performance targets</p>
                    <div className="space-y-3 flex-1 flex flex-col justify-center">
                        {performanceTargets.map((p) => (
                            <div key={p.metric} className="flex items-center gap-3">
                                <span className="text-xs font-mono w-32 shrink-0">{p.metric}</span>
                                <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                                    <div className="h-full bg-foreground rounded-full" style={{ width: `${p.current}%` }}></div>
                                </div>
                                <span className="text-[10px] font-mono text-muted-foreground w-14 text-right">{p.target}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Quantum circuit depth */}
                <div className="t-card p-4 animate-slide-up flex flex-col" style={{ animationDelay: "0.2s" }}>
                    <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-4">OCT quantum circuit depth</p>
                    <div className="h-[180px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={quantumCircuitData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f4f4f5" />
                                <XAxis dataKey="layer" tick={{ fontSize: 10, fontFamily: "IBM Plex Mono", fill: "#a1a1aa" }} axisLine={false} tickLine={false} />
                                <YAxis tick={{ fontSize: 10, fontFamily: "IBM Plex Mono", fill: "#a1a1aa" }} axisLine={false} tickLine={false} />
                                <RechartsTooltip contentStyle={{ borderRadius: "4px", border: "1px solid #e4e4e7", fontSize: 11, fontFamily: "IBM Plex Mono" }} />
                                <Area type="monotone" dataKey="depth" stroke="#09090b" fill="#09090b" fillOpacity={0.05} strokeWidth={1.5} name="Circuit depth" />
                                <Area type="monotone" dataKey="entanglement" stroke="#10b981" fill="#10b981" fillOpacity={0.05} strokeWidth={1.5} name="Entangling gates" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Architecture pipelines */}
            <div className="space-y-4 animate-slide-up" style={{ animationDelay: "0.25s" }}>
                <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">Architecture pipelines</p>
                {pipelineSteps.map((pipeline) => (
                    <div key={pipeline.title} className="t-card p-4">
                        <p className="text-xs font-mono font-bold mb-3">{pipeline.title}</p>
                        <div className="space-y-2">
                            {pipeline.steps.map((step, i) => (
                                <div key={i} className="flex items-start gap-3 text-xs font-mono">
                                    <span className="text-accent shrink-0 mt-0.5">{i + 1}.</span>
                                    <div className="flex-1">
                                        <span className="font-bold">{step.name}</span>
                                        <span className="text-muted-foreground ml-2">{step.detail}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            {/* API endpoints */}
            <div className="t-card p-4 animate-slide-up" style={{ animationDelay: "0.3s" }}>
                <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-4">API endpoints</p>
                <div className="space-y-2">
                    {endpoints.map((ep) => (
                        <div key={ep.path + ep.method} className="flex items-center gap-3 p-2 bg-muted rounded border border-border text-xs font-mono">
                            <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${ep.method === "POST" ? "bg-foreground text-white" : "bg-white border border-border text-foreground"}`}>
                                {ep.method}
                            </span>
                            <span className="font-bold">{ep.path}</span>
                            <span className="text-muted-foreground ml-auto hidden sm:inline">{ep.desc}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
