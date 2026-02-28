"use client";

import { useState, useCallback, useRef } from "react";

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

    // ── Async job state ────────────────────────────────────
    const [jobId, setJobId] = useState<string | null>(null);
    const [processingStep, setProcessingStep] = useState<string>("");

    // ── Doctor feedback state ──────────────────────────────
    const [feedbackSent, setFeedbackSent] = useState(false);
    const [showCorrectionDropdown, setShowCorrectionDropdown] = useState(false);
    const [feedbackNote, setFeedbackNote] = useState("");

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
    }, []);

    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            setDragOver(false);
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        },
        [handleFile]
    );

    // ── SSE Listener ───────────────────────────────────────
    const listenForResult = (jid: string) => {
        setProcessingStep("Connecting to inference worker...");
        const eventSource = new EventSource(`${API_BASE}/api/jobs/${jid}/stream`);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.status === "pending") {
                    setProcessingStep("Queued — waiting for GPU worker...");
                } else if (data.status === "processing") {
                    const step = data.step || "unknown";
                    const labels: Record<string, string> = {
                        loading_image: "Loading image...",
                        quantum_inference: "Running quantum circuit inference...",
                        unet_inference: "Running U-Net segmentation...",
                    };
                    setProcessingStep(labels[step] || `Processing (${step})...`);
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
                // Ignore malformed events
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
            // Fallback: poll once
            setTimeout(() => pollResult(jid), 2000);
        };
    };

    // ── Polling fallback ───────────────────────────────────
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
                // Still processing, poll again
                setTimeout(() => pollResult(jid), 2000);
            }
        } catch {
            setError("Lost connection to backend");
            setLoading(false);
        }
    };

    // ── Submit handler ─────────────────────────────────────
    const handleSubmit = async () => {
        if (!selectedFile) return;
        setLoading(true);
        setError(null);
        setResult(null);
        setFeedbackSent(false);
        setShowCorrectionDropdown(false);

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
                // Async mode — listen for result via SSE
                setJobId(data.job_id);
                listenForResult(data.job_id);
            } else {
                // Sync fallback
                setResult(data as PredictionResult);
                setActiveTab("classification");
                setLoading(false);
            }
        } catch (err: any) {
            setError(err.message || "An unexpected error occurred");
            setLoading(false);
        }
    };

    // ── Feedback submission ────────────────────────────────
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
            // Silent fail — feedback is non-critical
        }
    };

    const correctionOptions =
        imageType === "oct" ? ["Normal", "CSR"] : ["Healthy", "CSCR"];

    const isDisease =
        result?.prediction === "CSR" || result?.prediction === "CSCR";

    return (
        <div className="max-w-7xl mx-auto px-6 py-10">
            {/* Hero Section */}
            <div className="text-center mb-12 animate-slide-up">
                <h2 className="text-4xl font-extrabold gradient-text mb-3">
                    Retinal Image Diagnosis
                </h2>
                <p className="text-muted-foreground max-w-2xl mx-auto">
                    Upload an OCT or Fundus image for quantum-enhanced AI classification
                    with explainability and automated macular segmentation.
                </p>
            </div>

            <div className="grid lg:grid-cols-2 gap-8">
                {/* Left Column — Upload */}
                <div className="space-y-6 animate-slide-up" style={{ animationDelay: "0.1s" }}>
                    {/* Image Type Selector */}
                    <div className="glass-card mb-6 p-6">
                        <label className="text-sm font-semibold text-muted-foreground mb-3 block uppercase tracking-wider">
                            Image Type
                        </label>
                        <div className="grid grid-cols-2 gap-3">
                            {(["oct", "fundus"] as const).map((type) => (
                                <button
                                    key={type}
                                    onClick={() => {
                                        setImageType(type);
                                        setResult(null);
                                    }}
                                    className={`p-4 rounded-xl border text-left transition-all ${imageType === type
                                        ? "border-foreground bg-foreground text-background"
                                        : "border-border bg-card text-muted-foreground hover:border-foreground/50 hover:text-foreground"
                                        }`}
                                >
                                    <div className="font-semibold text-sm mb-1">
                                        {type === "oct" ? "OCT Scan" : "Fundus Image"}
                                    </div>
                                    <div className="text-xs opacity-60">
                                        {type === "oct"
                                            ? "8-qubit quantum classifier"
                                            : "EfficientNet + 4-qubit layer"}
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Drop Zone */}
                    <div
                        className={`glass-card p-8 drop-zone cursor-pointer ${dragOver ? "drag-over" : ""
                            }`}
                        onDragOver={(e) => {
                            e.preventDefault();
                            setDragOver(true);
                        }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                        onClick={() => fileRef.current?.click()}
                    >
                        <input
                            ref={fileRef}
                            type="file"
                            accept="image/*"
                            className="hidden"
                            onChange={(e) => {
                                const file = e.target.files?.[0];
                                if (file) handleFile(file);
                            }}
                        />

                        {preview ? (
                            <div className="image-viewer p-2 bg-background">
                                <img
                                    src={preview}
                                    alt="Selected retinal image"
                                    className="rounded-lg max-h-64 mx-auto object-contain"
                                />
                                <p className="text-center mt-3 text-sm text-muted-foreground font-mono">
                                    {selectedFile?.name}
                                </p>
                            </div>
                        ) : (
                            <div className="text-center py-10">
                                <div className="w-16 h-16 rounded-full border border-border bg-muted flex items-center justify-center mx-auto mb-6">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-foreground">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                        <polyline points="17 8 12 3 7 8" />
                                        <line x1="12" y1="3" x2="12" y2="15" />
                                    </svg>
                                </div>
                                <p className="text-foreground font-medium mb-2">
                                    Click or drag image here
                                </p>
                                <p className="text-xs text-muted-foreground uppercase tracking-widest">
                                    JPEG, PNG up to 10MB
                                </p>
                            </div>
                        )}
                    </div>

                    {/* Analyze Button */}
                    <button
                        onClick={handleSubmit}
                        disabled={!selectedFile || loading}
                        className="btn-primary w-full flex items-center justify-center gap-2"
                    >
                        {loading ? (
                            <>
                                <div className="spinner !w-5 !h-5 !border-2"></div>
                                <span>{processingStep || "Dispatching to worker..."}</span>
                            </>
                        ) : (
                            <>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="11" cy="11" r="8" />
                                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                                </svg>
                                <span>Run Quantum Diagnosis</span>
                            </>
                        )}
                    </button>

                    {error && (
                        <div className="p-4 border border-red-900 bg-red-950/20 rounded-md">
                            <p className="text-red-500 text-sm font-mono">{error}</p>
                        </div>
                    )}
                </div>

                {/* Right Column — Results */}
                <div className="space-y-6 animate-slide-up" style={{ animationDelay: "0.2s" }}>
                    {result ? (
                        <>
                            {/* Classification Result */}
                            <div className="glass-card p-6 mb-6">
                                <div className="flex items-center justify-between mb-6 border-b border-border pb-4">
                                    <h3 className="font-semibold text-foreground tracking-wide uppercase text-sm">
                                        Diagnosis Result
                                    </h3>
                                    <span
                                        className={`px-3 py-1 rounded-sm text-xs font-bold uppercase tracking-widest ${isDisease
                                            ? "bg-foreground text-background"
                                            : "border border-foreground text-foreground"
                                            }`}
                                    >
                                        {result.prediction}
                                    </span>
                                </div>

                                {/* Confidence bar */}
                                <div className="mb-6">
                                    <div className="flex justify-between text-sm mb-2">
                                        <span className="text-muted-foreground uppercase text-xs tracking-wider">Confidence Level</span>
                                        <span className="font-mono text-foreground">
                                            {(result.confidence * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="confidence-bar">
                                        <div
                                            className="confidence-fill"
                                            style={{ width: `${result.confidence * 100}%` }}
                                        ></div>
                                    </div>
                                </div>

                                {/* Probability */}
                                <div className="flex items-center justify-between text-sm">
                                    <span className="text-muted-foreground uppercase text-xs tracking-wider">Raw Probability</span>
                                    <span className="font-mono text-muted-foreground">
                                        {result.probability.toFixed(4)}
                                    </span>
                                </div>
                            </div>

                            {/* ── Doctor Feedback Panel ──────────────── */}
                            <div className="glass-card p-6">
                                <h3 className="font-semibold text-foreground tracking-wide uppercase text-sm mb-4 border-b border-border pb-3">
                                    Clinical Feedback
                                </h3>

                                {feedbackSent ? (
                                    <div className="text-center py-4">
                                        <div className="w-10 h-10 rounded-full bg-foreground text-background flex items-center justify-center mx-auto mb-3">
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                                <polyline points="20 6 9 17 4 12" />
                                            </svg>
                                        </div>
                                        <p className="text-sm text-foreground font-medium">Feedback Recorded</p>
                                        <p className="text-xs text-muted-foreground mt-1">Thank you. This will improve future predictions.</p>
                                    </div>
                                ) : (
                                    <div className="space-y-4">
                                        <p className="text-xs text-muted-foreground">
                                            Does this diagnosis align with your clinical assessment?
                                        </p>

                                        {/* Notes input */}
                                        <input
                                            type="text"
                                            placeholder="Optional clinical notes..."
                                            value={feedbackNote}
                                            onChange={(e) => setFeedbackNote(e.target.value)}
                                            className="w-full px-3 py-2 text-sm bg-background border border-border rounded-md text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-foreground"
                                        />

                                        <div className="flex gap-3">
                                            <button
                                                onClick={() => submitFeedback("accept")}
                                                className="flex-1 px-4 py-2.5 border border-foreground text-foreground text-sm font-medium hover:bg-foreground hover:text-background transition-all uppercase tracking-wider"
                                            >
                                                ✓ Accept
                                            </button>
                                            <button
                                                onClick={() => setShowCorrectionDropdown(true)}
                                                className="flex-1 px-4 py-2.5 bg-foreground text-background text-sm font-medium hover:opacity-80 transition-all uppercase tracking-wider"
                                            >
                                                ✗ Reject
                                            </button>
                                        </div>

                                        {/* Correction Dropdown */}
                                        {showCorrectionDropdown && (
                                            <div className="border border-border p-4 bg-background space-y-3 animate-slide-up">
                                                <p className="text-xs text-muted-foreground uppercase tracking-wider">
                                                    Select the correct diagnosis:
                                                </p>
                                                <div className="grid grid-cols-2 gap-2">
                                                    {correctionOptions.map((label) => (
                                                        <button
                                                            key={label}
                                                            onClick={() => submitFeedback("reject", label)}
                                                            className="px-4 py-2 border border-border text-sm text-foreground hover:bg-foreground hover:text-background transition-all font-mono"
                                                        >
                                                            {label}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            {/* Tabs */}
                            <div className="flex gap-2">
                                {(["classification", "heatmap", "segmentation"] as const).map(
                                    (tab) => {
                                        const disabled =
                                            (tab === "heatmap" &&
                                                !result.heatmap_base64 &&
                                                !result.gradcam_base64) ||
                                            (tab === "segmentation" && !result.segmentation);
                                        return (
                                            <button
                                                key={tab}
                                                disabled={disabled}
                                                onClick={() => setActiveTab(tab)}
                                                className={`px-4 py-2 text-sm border-b-2 transition-all capitalize ${activeTab === tab
                                                    ? "border-foreground text-foreground"
                                                    : "border-transparent text-muted-foreground hover:text-foreground"
                                                    } ${disabled ? "opacity-30 cursor-not-allowed" : ""}`}
                                            >
                                                {tab}
                                            </button>
                                        );
                                    }
                                )}
                            </div>

                            {/* Tab Content */}
                            <div className="glass-card p-6 mt-4">
                                {activeTab === "classification" && (
                                    <div className="space-y-6">
                                        <h4 className="text-xs tracking-widest uppercase font-semibold text-muted-foreground border-b border-border pb-2">
                                            Classification Details
                                        </h4>
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="p-4 bg-background border border-border">
                                                <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wider">
                                                    Image Type
                                                </p>
                                                <p className="text-foreground font-mono">
                                                    {result.image_type}
                                                </p>
                                            </div>
                                            <div className="p-4 bg-background border border-border">
                                                <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wider">Model</p>
                                                <p className="text-foreground text-sm font-mono">
                                                    {result.image_type === "OCT"
                                                        ? "8-Qubit Quantum"
                                                        : "EfficientNet + 4Q"}
                                                </p>
                                            </div>
                                        </div>
                                        {result.feature_importance && (
                                            <div className="pt-4 border-t border-border">
                                                <p className="text-xs tracking-widest uppercase font-semibold text-muted-foreground mb-4">
                                                    Feature Importance (top 10)
                                                </p>
                                                <div className="space-y-3">
                                                    {result.feature_importance
                                                        .map((v, i) => ({ value: v, index: i }))
                                                        .sort((a, b) => b.value - a.value)
                                                        .slice(0, 10)
                                                        .map((f) => (
                                                            <div key={f.index} className="flex items-center gap-4">
                                                                <span className="text-xs text-muted-foreground font-mono w-8">
                                                                    F{f.index}
                                                                </span>
                                                                <div className="flex-1 confidence-bar">
                                                                    <div
                                                                        className="confidence-fill"
                                                                        style={{ width: `${f.value * 100}%` }}
                                                                    ></div>
                                                                </div>
                                                                <span className="text-xs text-foreground font-mono w-10 text-right">
                                                                    {(f.value * 100).toFixed(0)}%
                                                                </span>
                                                            </div>
                                                        ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {activeTab === "heatmap" && (
                                    <div className="space-y-4">
                                        <h4 className="text-xs tracking-widest uppercase font-semibold text-muted-foreground border-b border-border pb-2">
                                            {result.image_type === "OCT"
                                                ? "Feature Importance Heatmap"
                                                : "Grad-CAM Visualization"}
                                        </h4>
                                        <div className="image-viewer p-2 bg-background">
                                            <img
                                                src={`data:image/png;base64,${result.heatmap_base64 || result.gradcam_base64
                                                    }`}
                                                alt="Explainability heatmap"
                                                className="w-full object-contain max-h-80"
                                            />
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-4 leading-relaxed">
                                            {result.image_type === "OCT"
                                                ? "Brighter regions indicate features with higher contribution to the classification decision."
                                                : "Grad-CAM highlights the retinal regions most influential to the model's prediction."}
                                        </p>
                                    </div>
                                )}

                                {activeTab === "segmentation" && result.segmentation && (
                                    <div className="space-y-4">
                                        <h4 className="text-xs tracking-widest uppercase font-semibold text-muted-foreground border-b border-border pb-2">
                                            Macular Segmentation
                                        </h4>
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="image-viewer p-2 bg-background">
                                                <p className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2 text-center">
                                                    Binary Mask
                                                </p>
                                                <img
                                                    src={`data:image/png;base64,${result.segmentation.mask_base64}`}
                                                    alt="Segmentation mask"
                                                    className="w-full object-contain"
                                                />
                                            </div>
                                            <div className="image-viewer p-2 bg-background">
                                                <p className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2 text-center">Overlay</p>
                                                <img
                                                    src={`data:image/png;base64,${result.segmentation.overlay_base64}`}
                                                    alt="Segmentation overlay"
                                                    className="w-full object-contain"
                                                />
                                            </div>
                                        </div>
                                        <div className="mt-4 p-4 bg-background border border-border flex justify-between items-center">
                                            <p className="text-xs tracking-wider uppercase text-muted-foreground">
                                                Affected Area Ratio
                                            </p>
                                            <p className="font-mono text-foreground">
                                                {(result.segmentation.mask_area_ratio * 100).toFixed(2)}%
                                            </p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </>
                    ) : (
                        /* Placeholder */
                        <div className="glass-card p-12 flex flex-col items-center justify-center text-center h-full min-h-[400px]">
                            <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-6">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-foreground">
                                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                                    <polyline points="21 15 16 10 5 21"></polyline>
                                </svg>
                            </div>
                            <h3 className="text-sm uppercase tracking-widest font-semibold text-foreground mb-2">
                                Ready to Analyze
                            </h3>
                            <p className="text-sm text-muted-foreground max-w-xs">
                                Upload a retinal image and select the scan type to begin.
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
