"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
    ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    Radar, AreaChart, Area, PieChart, Pie, Cell,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ImageType = "oct" | "fundus";

interface SegmentationResult {
    mask_base64: string;
    overlay_base64: string;
    mask_area_ratio: number;
}

interface PredictionResult {
    image_type: string;
    prediction: string;
    confidence: number;
    probability: number;
    heatmap_base64?: string;
    gradcam_base64?: string;
    feature_importance?: number[];
    segmentation?: SegmentationResult | null;
}

const ValidationSteps = [
    { id: "upload", label: "Ingesting scan data", cmd: "load --stream" },
    { id: "loading_image", label: "Preprocessing & normalization", cmd: "preprocess --clahe" },
    { id: "quantum_inference", label: "Quantum circuit inference", cmd: "qml.execute()" },
    { id: "unet_inference", label: "Feature extraction", cmd: "model.forward()" },
    { id: "cross_val", label: "Generating explainability maps", cmd: "gradcam.compute()" },
];

export default function DiagnosePage() {
    const [imageType, setImageType] = useState<ImageType>("oct");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dragOver, setDragOver] = useState(false);
    const [activeTab, setActiveTab] = useState<"classification" | "heatmap" | "segmentation">("classification");
    const fileRef = useRef<HTMLInputElement>(null);

    const [jobId, setJobId] = useState<string | null>(null);
    const [processingStep, setProcessingStep] = useState<string>("");
    const [currentStepIndex, setCurrentStepIndex] = useState(0);

    const [feedbackSent, setFeedbackSent] = useState(false);
    const [showCorrectionDropdown, setShowCorrectionDropdown] = useState(false);
    const [feedbackNote, setFeedbackNote] = useState("");
    const [segLoading, setSegLoading] = useState(false);

    useEffect(() => {
        if (loading && processingStep) {
            const idx = ValidationSteps.findIndex((s) => s.id === processingStep);
            if (idx !== -1) setCurrentStepIndex(idx);
        } else if (!loading) {
            setCurrentStepIndex(0);
        }
    }, [loading, processingStep]);

    const handleFile = useCallback((file: File) => {
        if (!file.type.startsWith("image/")) {
            setError("Please upload an image file (JPEG or PNG)");
            return;
        }
        setSelectedFile(file);
        setPreview(URL.createObjectURL(file));
        setResult(null);
        setError(null);
        setJobId(null);
        setFeedbackSent(false);
        setShowCorrectionDropdown(false);
        setCurrentStepIndex(0);
    }, []);

    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            setDragOver(false);
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        },
        [handleFile],
    );

    const listenForResult = (jid: string) => {
        setProcessingStep("upload");
        const eventSource = new EventSource(`${API_BASE}/api/jobs/${jid}/stream`);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.status === "pending") {
                    setProcessingStep("upload");
                } else if (data.status === "processing") {
                    const step = data.step || "unknown";
                    setProcessingStep(step);
                    if (step === "unet_inference") {
                        setTimeout(() => setProcessingStep("cross_val"), 1500);
                    }
                } else if (data.status === "complete") {
                    eventSource.close();
                    setResult(data.result as PredictionResult);
                    setActiveTab("classification");
                    setLoading(false);
                } else if (data.status === "failed") {
                    eventSource.close();
                    setError(data.error || "Inference failed on the worker");
                    setLoading(false);
                }
            } catch {
                /* ignore malformed events */
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
            setTimeout(() => pollResult(jid), 2000);
        };
    };

    const pollResult = async (jid: string) => {
        try {
            const res = await fetch(`${API_BASE}/api/jobs/${jid}`);
            const data = await res.json();
            if (data.status === "complete") {
                setResult(data.result as PredictionResult);
                setActiveTab("classification");
                setLoading(false);
            } else if (data.status === "failed") {
                setError(data.error || "Inference failed");
                setLoading(false);
            } else {
                setTimeout(() => pollResult(jid), 2000);
            }
        } catch {
            setError("Lost connection to backend");
            setLoading(false);
        }
    };

    const handleSubmit = async () => {
        if (!selectedFile) return;
        setLoading(true);
        setError(null);
        setResult(null);
        setFeedbackSent(false);
        setShowCorrectionDropdown(false);
        setProcessingStep("upload");

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);
            const endpoint =
                imageType === "oct"
                    ? `${API_BASE}/api/predict/oct?async_mode=true`
                    : `${API_BASE}/api/predict/fundus?async_mode=true`;
            const res = await fetch(endpoint, { method: "POST", body: formData });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Prediction failed");
            }
            const data = await res.json();
            if (data.job_id) {
                setJobId(data.job_id);
                listenForResult(data.job_id);
            } else {
                setResult(data as PredictionResult);
                setActiveTab("classification");
                setLoading(false);
            }
        } catch (err: any) {
            setError(err.message || "An unexpected error occurred");
            setLoading(false);
        }
    };

    const submitFeedback = async (verdict: "accept" | "reject", correction?: string) => {
        if (!jobId) return;
        try {
            await fetch(`${API_BASE}/api/feedback`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    job_id: jobId,
                    doctor_verdict: verdict,
                    correction: correction || null,
                    notes: feedbackNote || null,
                }),
            });
            setFeedbackSent(true);
            setShowCorrectionDropdown(false);
        } catch {
            /* non-critical */
        }
    };

    const correctionOptions = imageType === "oct" ? ["Normal", "CSR"] : ["Healthy", "CSCR"];
    const isDisease = result?.prediction === "CSR" || result?.prediction === "CSCR" || result?.prediction === "DISEASED";

    const runSegmentation = async () => {
        if (!selectedFile || segLoading) return;
        setSegLoading(true);
        try {
            const formData = new FormData();
            formData.append("file", selectedFile);
            const res = await fetch(`${API_BASE}/api/segment`, { method: "POST", body: formData });
            if (!res.ok) throw new Error("Segmentation failed");
            const seg = await res.json();
            setResult((prev) => prev ? { ...prev, segmentation: seg } : prev);
            setActiveTab("segmentation");
        } catch {
            setError("Segmentation request failed");
        } finally {
            setSegLoading(false);
        }
    };

    // Build dynamic feature importance data for radar chart
    const featureRadarData = result?.feature_importance
        ? [
            { subject: "Gradient", A: Math.round(result.feature_importance.slice(0, 16).reduce((a, b) => a + b, 0) / 16 * 100) },
            { subject: "Histogram", A: Math.round(result.feature_importance.slice(16, 32).reduce((a, b) => a + b, 0) / 16 * 100) },
            { subject: "LBP", A: Math.round(result.feature_importance.slice(32, 48).reduce((a, b) => a + b, 0) / 16 * 100) },
            { subject: "Texture", A: Math.round(result.feature_importance.slice(48, 56).reduce((a, b) => a + b, 0) / 8 * 100) },
            { subject: "Moments", A: Math.round(result.feature_importance.slice(56, 64).reduce((a, b) => a + b, 0) / 8 * 100) },
        ]
        : null;

    // Build bar chart data from top-N feature importances
    const featureBarData = result?.feature_importance
        ? result.feature_importance.map((v, i) => ({ idx: i, value: +(v * 100).toFixed(1) })).sort((a, b) => b.value - a.value).slice(0, 16)
        : null;

    // Confidence pie chart
    const confidencePieData = result
        ? [
            { name: "Confidence", value: +(result.confidence * 100).toFixed(1) },
            { name: "Uncertainty", value: +((1 - result.confidence) * 100).toFixed(1) },
        ]
        : null;

    return (
        <div className="w-full max-w-[1440px] mx-auto px-4 sm:px-6 py-5 space-y-4">
            {/* Header */}
            <div className="animate-slide-up border-b border-border pb-3">
                <div className="flex items-center gap-2 mb-1">
                    <span className="text-accent font-mono text-xs font-bold">$</span>
                    <h2 className="text-lg font-bold font-mono tracking-tight">retina-q diagnose</h2>
                </div>
                <p className="text-xs text-muted-foreground font-mono pl-4">
                    Hybrid quantum-classical retinal disease classification &bull; OCT &amp; Fundus pipelines
                </p>
            </div>

            {/* Input strip — compact horizontal bar */}
            <div className="grid grid-cols-12 gap-3 animate-slide-up" style={{ animationDelay: "0.05s" }}>
                {/* Modality */}
                <div className="col-span-12 sm:col-span-3 lg:col-span-2 t-card p-3">
                    <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">Modality</p>
                    <div className="space-y-1.5">
                        {(["oct", "fundus"] as const).map((type) => (
                            <button
                                key={type}
                                onClick={() => { setImageType(type); setResult(null); }}
                                className={`w-full p-2 rounded border text-left transition-all font-mono ${imageType === type
                                    ? "border-foreground bg-foreground text-white"
                                    : "border-border bg-white hover:bg-muted text-foreground"
                                    }`}
                            >
                                <div className="text-[11px] font-bold">{type === "oct" ? "OCT Scan" : "Fundus"}</div>
                                <div className={`text-[9px] ${imageType === type ? "text-zinc-400" : "text-muted-foreground"}`}>
                                    {type === "oct" ? "8-qubit QC" : "EfficientNet + 4Q"}
                                </div>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Upload zone */}
                <div className="col-span-12 sm:col-span-6 lg:col-span-8">
                    <div
                        className={`drop-zone p-3 cursor-pointer h-full flex items-center justify-center ${dragOver ? "drag-over" : ""}`}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                        onClick={() => fileRef.current?.click()}
                    >
                        <input
                            ref={fileRef}
                            type="file"
                            accept="image/*"
                            className="hidden"
                            onChange={(e) => { const file = e.target.files?.[0]; if (file) handleFile(file); }}
                        />
                        {preview ? (
                            <div className="bg-muted rounded p-2 border border-border w-full">
                                <img src={preview} alt="Selected retinal image" className="rounded max-h-36 mx-auto object-contain" />
                                <p className="text-center mt-1 text-[10px] font-mono text-muted-foreground truncate">{selectedFile?.name}</p>
                            </div>
                        ) : (
                            <div className="text-center py-4">
                                <div className="w-10 h-10 rounded bg-muted flex items-center justify-center mx-auto mb-2 text-muted-foreground">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                        <polyline points="17 8 12 3 7 8" />
                                        <line x1="12" y1="3" x2="12" y2="15" />
                                    </svg>
                                </div>
                                <p className="text-sm font-medium">Drop retinal scan here</p>
                                <p className="text-[10px] text-muted-foreground font-mono mt-1">JPEG / PNG &bull; max 10 MB</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Execute + pipeline status */}
                <div className="col-span-12 sm:col-span-3 lg:col-span-2 flex flex-col gap-2">
                    <button
                        onClick={handleSubmit}
                        disabled={!selectedFile || loading}
                        className={`w-full flex items-center justify-center gap-2 py-2.5 text-xs font-bold font-mono uppercase tracking-widest rounded transition-all ${!selectedFile || loading
                            ? "bg-muted text-muted-foreground cursor-not-allowed border border-border"
                            : "bg-foreground text-white hover:bg-zinc-800"
                            }`}
                    >
                        {loading ? (
                            <span className="cursor">processing</span>
                        ) : (
                            <>
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polygon points="5 3 19 12 5 21 5 3" /></svg>
                                execute
                            </>
                        )}
                    </button>

                    {loading && (
                        <div className="t-card p-2 animate-fade-in flex-1">
                            <div className="space-y-1 font-mono text-[10px]">
                                {ValidationSteps.map((step, idx) => {
                                    const isPast = idx < currentStepIndex;
                                    const isCurrent = idx === currentStepIndex;
                                    return (
                                        <div key={step.id} className={`flex items-center gap-1 ${isPast ? "text-muted-foreground" : isCurrent ? "text-foreground" : "text-zinc-300"}`}>
                                            <span className={`w-3 text-center text-[9px] ${isPast ? "text-accent" : isCurrent ? "text-accent animate-pulse-soft" : ""}`}>
                                                {isPast ? "✓" : isCurrent ? "▶" : "·"}
                                            </span>
                                            <span className={isCurrent ? "font-medium" : ""}>{step.label}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {!loading && !result && (
                        <div className="t-card p-2 flex-1">
                            <div className="space-y-1 font-mono text-[10px] text-muted-foreground">
                                <div>1. Select modality</div>
                                <div>2. Upload scan</div>
                                <div>3. Execute</div>
                            </div>
                        </div>
                    )}

                    {error && (
                        <div className="t-card p-2 border-danger/30 bg-red-50 animate-fade-in">
                            <p className="text-[10px] font-mono text-danger truncate">[ERR] {error}</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Results — full width below input */}
            {result ? (
                <div className="space-y-4 animate-slide-up" style={{ animationDelay: "0.1s" }}>
                    {/* Verdict row: 8/4 split */}
                    <div className="grid grid-cols-12 gap-3">
                        <div className="col-span-12 lg:col-span-8 t-card p-4">
                            <div className="flex items-center justify-between mb-3 pb-3 border-b border-border">
                                <div>
                                    <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">Diagnostic result</p>
                                    <p className="text-sm font-mono font-bold mt-0.5">{result.image_type} Pipeline</p>
                                </div>
                                <div className={`px-3 py-1.5 rounded text-xs font-mono font-bold ${isDisease ? "bg-red-50 text-danger border border-red-200" : "bg-emerald-50 text-accent border border-emerald-200"}`}>
                                    {result.prediction}
                                </div>
                            </div>
                            <div className="grid grid-cols-3 gap-4 items-center">
                                <div>
                                    <div className="flex justify-between text-[10px] font-mono text-muted-foreground uppercase mb-1.5">
                                        <span>confidence</span>
                                        <span className="text-foreground font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="confidence-bar">
                                        <div className="confidence-fill" style={{ width: `${result.confidence * 100}%` }}></div>
                                    </div>
                                </div>
                                <div>
                                    <p className="text-[10px] font-mono text-muted-foreground uppercase mb-1.5">probability</p>
                                    <p className="text-lg font-mono font-bold">{result.probability.toFixed(4)}</p>
                                </div>
                                {confidencePieData && (
                                    <div className="flex items-center gap-3 justify-end">
                                        <div className="w-12 h-12">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <PieChart>
                                                    <Pie data={confidencePieData} dataKey="value" cx="50%" cy="50%" innerRadius={14} outerRadius={22} strokeWidth={0}>
                                                        <Cell fill="#09090b" />
                                                        <Cell fill="#e4e4e7" />
                                                    </Pie>
                                                </PieChart>
                                            </ResponsiveContainer>
                                        </div>
                                        <div className="text-[9px] font-mono text-muted-foreground space-y-0.5">
                                            <div className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-sm bg-foreground"></span>{confidencePieData[0].value}%</div>
                                            <div className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-sm bg-zinc-200"></span>{confidencePieData[1].value}%</div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Feedback */}
                        <div className="col-span-12 lg:col-span-4 t-card p-4">
                            <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">Physician feedback</p>
                            {feedbackSent ? (
                                <div className="text-center py-2 bg-emerald-50 rounded border border-emerald-200 animate-fade-in">
                                    <p className="text-xs font-mono font-bold text-accent">Recorded</p>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    <input
                                        type="text"
                                        placeholder="Clinical notes (optional)"
                                        value={feedbackNote}
                                        onChange={(e) => setFeedbackNote(e.target.value)}
                                        className="w-full px-3 py-1.5 text-xs font-mono bg-muted border border-border rounded text-foreground placeholder:text-zinc-400 focus:outline-none focus:ring-1 focus:ring-foreground"
                                    />
                                    <div className="flex gap-2">
                                        <button onClick={() => submitFeedback("accept")} className="flex-1 px-2 py-1.5 border border-border text-xs font-mono hover:bg-muted rounded transition-colors">
                                            Concur
                                        </button>
                                        <button onClick={() => setShowCorrectionDropdown(true)} className="flex-1 px-2 py-1.5 bg-foreground text-white text-xs font-mono rounded hover:bg-zinc-800 transition-colors">
                                            Override
                                        </button>
                                    </div>
                                    {showCorrectionDropdown && (
                                        <div className="border border-border p-2 rounded bg-muted animate-fade-in">
                                            <p className="text-[10px] font-mono text-muted-foreground uppercase mb-1">Correct label:</p>
                                            <div className="grid grid-cols-2 gap-1.5">
                                                {correctionOptions.map((label) => (
                                                    <button key={label} onClick={() => submitFeedback("reject", label)} className="px-2 py-1 border border-border rounded bg-white text-xs font-mono hover:bg-muted transition-colors">
                                                        {label}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Tabs — full width content area */}
                    <div className="t-card overflow-hidden">
                        <div className="flex border-b border-border">
                            {(["classification", "heatmap", "segmentation"] as const).map((tab) => {
                                const disabled =
                                    (tab === "heatmap" && !result.heatmap_base64 && !result.gradcam_base64) ||
                                    (tab === "segmentation" && imageType === "oct");
                                return (
                                    <button
                                        key={tab}
                                        disabled={disabled}
                                        onClick={() => {
                                            if (tab === "segmentation" && !result.segmentation) {
                                                runSegmentation();
                                            } else {
                                                setActiveTab(tab);
                                            }
                                        }}
                                        className={`flex-1 py-2.5 text-[10px] font-mono uppercase tracking-widest transition-colors border-b-2
                                            ${activeTab === tab ? "border-foreground text-foreground bg-white font-bold" : "border-transparent text-muted-foreground hover:text-foreground hover:bg-muted"}
                                            ${disabled ? "opacity-30 cursor-not-allowed" : ""}`}
                                    >
                                        {tab === "segmentation" && segLoading ? "loading..." : tab}
                                    </button>
                                );
                            })}
                        </div>

                        <div className="p-4">
                            {activeTab === "classification" && (
                                <div className="space-y-4">
                                    <div className="p-3 bg-muted rounded border border-border">
                                        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">What the AI analyzed</p>
                                        <p className="text-xs text-foreground leading-relaxed">
                                            {result.image_type === "OCT" ? (
                                                <>The model extracted <strong>64 statistical features</strong> from the OCT scan — gradient magnitudes, histogram distributions, local binary patterns, texture variance, and statistical moments. These were reduced to <strong>8 dimensions</strong> and processed through an <strong>8-qubit quantum circuit</strong> with {8} variational layers.</>
                                            ) : (
                                                <>The fundus image was processed through <strong>EfficientNet-B0</strong> producing <strong>1,280 deep features</strong>, compressed to <strong>4 dimensions</strong> and fed into a <strong>4-qubit quantum circuit</strong> with {6} variational layers. The quantum output was concatenated with classical features for final classification.</>
                                            )}
                                        </p>
                                    </div>

                                    {/* Charts — side by side */}
                                    {featureRadarData && featureBarData && (
                                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                            <div>
                                                <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">Feature group importance</p>
                                                <div className="h-[220px] w-full">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={featureRadarData}>
                                                            <PolarGrid stroke="#e4e4e7" />
                                                            <PolarAngleAxis dataKey="subject" tick={{ fill: "#71717a", fontSize: 10, fontFamily: "IBM Plex Mono" }} />
                                                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                                            <Radar name="Importance" dataKey="A" stroke="#09090b" fill="#09090b" fillOpacity={0.1} strokeWidth={1.5} />
                                                        </RadarChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                            <div>
                                                <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">Top 16 feature activations</p>
                                                <div className="h-[220px] w-full">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <BarChart data={featureBarData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f4f4f5" />
                                                            <XAxis dataKey="idx" tick={{ fontSize: 9, fontFamily: "IBM Plex Mono", fill: "#a1a1aa" }} axisLine={false} tickLine={false} />
                                                            <YAxis tick={{ fontSize: 9, fontFamily: "IBM Plex Mono", fill: "#a1a1aa" }} axisLine={false} tickLine={false} />
                                                            <RechartsTooltip contentStyle={{ borderRadius: "4px", border: "1px solid #e4e4e7", fontSize: 11, fontFamily: "IBM Plex Mono" }} />
                                                            <Bar dataKey="value" fill="#09090b" radius={[2, 2, 0, 0]} />
                                                        </BarChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {!result.feature_importance && (
                                        <div className="p-3 bg-muted rounded border border-border">
                                            <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">Model decision factors</p>
                                            <p className="text-xs text-foreground leading-relaxed">
                                                The Grad-CAM visualization (Heatmap tab) highlights which spatial regions most influenced the classification. Brighter regions indicate stronger activation.
                                            </p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {activeTab === "heatmap" && (
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                    <div className="border border-border rounded overflow-hidden bg-muted flex justify-center p-2">
                                        <img
                                            src={`data:image/png;base64,${result.heatmap_base64 || result.gradcam_base64}`}
                                            alt="Explainability heatmap"
                                            className="max-w-full max-h-72 object-contain"
                                        />
                                    </div>
                                    <div className="space-y-3">
                                        <div className="p-2 bg-muted rounded border border-border">
                                            <span className="text-[10px] font-mono font-bold uppercase tracking-widest">
                                                {result.image_type === "OCT" ? "Feature importance heatmap" : "Grad-CAM activation map"}
                                            </span>
                                        </div>
                                        <div className="p-3 bg-muted rounded border border-border">
                                            <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-1">Interpretation</p>
                                            <p className="text-xs text-foreground leading-relaxed">
                                                {result.image_type === "OCT" ? (
                                                    <>64-dimensional feature importance mapped to an 8&times;8 spatial grid. <strong>Red/warm regions</strong> show features with highest gradient magnitude relative to model output.</>
                                                ) : (
                                                    <>Gradient-weighted activation of EfficientNet-B0&apos;s final conv layer. <strong>Red/warm areas</strong> are regions the model focused on most for classification.</>
                                                )}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {activeTab === "segmentation" && (
                                <div>
                                    {segLoading ? (
                                        <div className="flex flex-col items-center justify-center py-8 text-center">
                                            <div className="w-8 h-8 rounded bg-muted flex items-center justify-center mb-2 text-muted-foreground animate-pulse-soft">
                                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M9 21V9" /></svg>
                                            </div>
                                            <p className="text-xs font-mono text-muted-foreground cursor">Running U-Net segmentation</p>
                                        </div>
                                    ) : result.segmentation ? (
                                        <div className="grid grid-cols-12 gap-4">
                                            <div className="col-span-12 lg:col-span-4 border border-border rounded overflow-hidden p-2 bg-muted">
                                                <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2 text-center">Binary mask</p>
                                                <img src={`data:image/png;base64,${result.segmentation.mask_base64}`} alt="Mask" className="w-full object-contain rounded" />
                                            </div>
                                            <div className="col-span-12 lg:col-span-4 border border-border rounded overflow-hidden p-2 bg-muted">
                                                <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2 text-center">Overlay</p>
                                                <img src={`data:image/png;base64,${result.segmentation.overlay_base64}`} alt="Overlay" className="w-full object-contain rounded" />
                                            </div>
                                            <div className="col-span-12 lg:col-span-4 space-y-3">
                                                <div className="p-3 bg-muted rounded border border-border flex justify-between items-center">
                                                    <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Anomalous area</span>
                                                    <span className="font-mono text-sm font-bold">{(result.segmentation.mask_area_ratio * 100).toFixed(2)}%</span>
                                                </div>
                                                <div className="p-3 bg-muted rounded border border-border">
                                                    <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-1">About</p>
                                                    <p className="text-xs text-foreground leading-relaxed">
                                                        U-Net processes green channel with CLAHE enhancement. Post-processing includes connected component analysis and morphological refinement.
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="flex flex-col items-center justify-center py-8 text-center">
                                            <p className="text-xs font-mono text-muted-foreground">Segmentation not available for OCT scans</p>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            ) : (
                /* Idle — pipeline info full width */
                <div className="t-card p-4 animate-slide-up" style={{ animationDelay: "0.1s" }}>
                    <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-3">Pipeline architecture</p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs">
                        {[
                            { label: "OCT", desc: "64 features → 8-qubit QC (8 layers) → dense classifier", tag: "PennyLane" },
                            { label: "Fundus", desc: "EfficientNet-B0 → 4-qubit QC (6 layers) → concat → classifier", tag: "Hybrid" },
                            { label: "Segment", desc: "Green channel → CLAHE → U-Net → morphological postprocess", tag: "PyTorch" },
                        ].map((item) => (
                            <div key={item.label} className="flex items-start gap-3 p-3 bg-muted rounded border border-border">
                                <span className="text-accent font-bold shrink-0">{item.label}</span>
                                <span className="text-muted-foreground flex-1">{item.desc}</span>
                                <span className="text-[9px] px-1.5 py-0.5 bg-white border border-border rounded shrink-0">{item.tag}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
