"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

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

// Dummy data for clinical dashboard monitoring graphs
const performanceData = [
    { time: '08:00', throughput: 45, accuracy: 94 },
    { time: '09:00', throughput: 52, accuracy: 95 },
    { time: '10:00', throughput: 38, accuracy: 94 },
    { time: '11:00', throughput: 65, accuracy: 96 },
    { time: '12:00', throughput: 48, accuracy: 95 },
    { time: '13:00', throughput: 70, accuracy: 97 },
    { time: '14:00', throughput: 55, accuracy: 96 },
];

const featureConfidenceData = [
    { subject: 'Macular Thickness', A: 95, fullMark: 100 },
    { subject: 'Foveal Contour', A: 88, fullMark: 100 },
    { subject: 'Drusen Deposits', A: 75, fullMark: 100 },
    { subject: 'Vascular Pattern', A: 92, fullMark: 100 },
    { subject: 'Optic Disc', A: 85, fullMark: 100 },
];

const ValidationSteps = [
    { id: 'upload', label: 'Ingesting Scan Data' },
    { id: 'loading_image', label: 'Preprocessing & Filtering' },
    { id: 'quantum_inference', label: 'Quantum Circuit Inference' },
    { id: 'unet_inference', label: 'Feature Extraction & Segmentation' },
    { id: 'cross_val', label: 'Cross-referencing Clinical Baseline' },
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

    // ── Async job state ────────────────────────────────────
    const [jobId, setJobId] = useState<string | null>(null);
    const [processingStep, setProcessingStep] = useState<string>("");
    const [currentStepIndex, setCurrentStepIndex] = useState(0);

    // ── Doctor feedback state ──────────────────────────────
    const [feedbackSent, setFeedbackSent] = useState(false);
    const [showCorrectionDropdown, setShowCorrectionDropdown] = useState(false);
    const [feedbackNote, setFeedbackNote] = useState("");

    useEffect(() => {
        if (loading && processingStep) {
            const exactStepIdx = ValidationSteps.findIndex(s => s.id === processingStep);
            if (exactStepIdx !== -1) setCurrentStepIndex(exactStepIdx);
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
        [handleFile]
    );

    // ── SSE Listener ───────────────────────────────────────
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
                    // Artificial progression logic for visual validation layers
                    if (step === 'unet_inference') {
                        setTimeout(() => {
                            setProcessingStep("cross_val");
                        }, 1500);
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
                // Ignore malformed events
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
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
        result?.prediction === "CSR" || result?.prediction === "CSCR" || result?.prediction === "DISEASED";

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

            {/* Header / Hero Section */}
            <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 animate-fade-in border-b border-slate-200 pb-6">
                <div>
                    <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Clinical Diagnosis Pipeline</h2>
                    <p className="text-slate-500 mt-2 font-medium max-w-2xl">
                        High-precision multi-modal retinal disease diagnosis powered by hybrid quantum-classical neural networks.
                    </p>
                </div>
                <div className="inline-flex items-center gap-2 bg-sky-50 text-sky-700 px-3 py-1.5 rounded-full border border-sky-100 text-sm font-semibold">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="12" r="10" /><path d="m9 12 2 2 4-4" /></svg>
                    Enterprise Node
                </div>
            </div>

            <div className="grid lg:grid-cols-2 gap-8">
                {/* ── Left Column: Input ──────────────── */}
                <div className="space-y-6 animate-slide-up" style={{ animationDelay: "0.1s" }}>

                    {/* Image Type Selector */}
                    <div className="med-card p-6">
                        <h3 className="text-sm font-semibold tracking-wide text-slate-500 uppercase mb-4">1. Select Modality</h3>
                        <div className="grid grid-cols-2 gap-4">
                            {(["oct", "fundus"] as const).map((type) => (
                                <button
                                    key={type}
                                    onClick={() => {
                                        setImageType(type);
                                        setResult(null);
                                    }}
                                    className={`p-4 rounded-xl border text-left transition-all ${imageType === type
                                        ? "border-sky-500 bg-sky-50 ring-1 ring-sky-500 shadow-sm"
                                        : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                                        }`}
                                >
                                    <div className={`font-semibold text-base mb-1 ${imageType === type ? 'text-sky-900' : 'text-slate-800'}`}>
                                        {type === "oct" ? "OCT Scan" : "Fundus Image"}
                                    </div>
                                    <div className={`text-xs ${imageType === type ? 'text-sky-700' : 'text-slate-500'}`}>
                                        {type === "oct" ? "8-Qubit Quantum" : "EfficientNet + 4Q"}
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Drop Zone */}
                    <div
                        className={`drop-zone p-8 cursor-pointer ${dragOver ? "drag-over" : ""}`}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                        onClick={() => fileRef.current?.click()}
                    >
                        <h3 className="text-sm font-semibold tracking-wide text-slate-500 uppercase mb-4 text-center">2. Upload Source Imaging</h3>
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
                            <div className="bg-slate-50 rounded-lg p-2 border border-slate-200">
                                <img
                                    src={preview}
                                    alt="Selected retinal image"
                                    className="rounded-md max-h-64 mx-auto object-contain"
                                />
                                <p className="text-center mt-3 text-xs font-medium text-slate-600 truncate">
                                    {selectedFile?.name}
                                </p>
                            </div>
                        ) : (
                            <div className="text-center py-8">
                                <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center mx-auto mb-4 text-slate-400">
                                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                        <polyline points="17 8 12 3 7 8" />
                                        <line x1="12" y1="3" x2="12" y2="15" />
                                    </svg>
                                </div>
                                <p className="text-slate-800 font-semibold mb-1">Click to browse or drag file</p>
                                <p className="text-xs text-slate-500">Supported formats: JPEG, PNG (max 10MB)</p>
                            </div>
                        )}
                    </div>

                    {/* Action Region */}
                    <div>
                        <button
                            onClick={handleSubmit}
                            disabled={!selectedFile || loading}
                            className={`w-full flex items-center justify-center gap-2 py-3.5 text-base font-semibold rounded-xl text-white transition-all shadow-sm ${!selectedFile || loading ? 'bg-slate-400 cursor-not-allowed' : 'bg-sky-600 hover:bg-sky-700 hover:shadow-md hover:-translate-y-0.5'
                                }`}
                        >
                            {!loading && (
                                <>
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
                                    Execute Clinical Diagnosis
                                </>
                            )}
                            {loading && <span>Initializing Pipeline...</span>}
                        </button>

                        {/* Validation Progress Layers */}
                        {loading && (
                            <div className="mt-6 med-card p-6 animate-fade-in">
                                <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-4">Diagnostics Sequence Log</h4>
                                <div className="space-y-4">
                                    {ValidationSteps.map((step, idx) => {
                                        const isPast = idx < currentStepIndex;
                                        const isCurrent = idx === currentStepIndex;
                                        return (
                                            <div key={step.id} className={`flex items-center gap-3 ${isPast ? 'opacity-50' : isCurrent ? 'opacity-100' : 'opacity-30'}`}>
                                                <div className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${isPast ? 'bg-emerald-500 text-white' : isCurrent ? 'bg-sky-500 text-white animate-pulse-soft' : 'bg-slate-200 text-transparent'}`}>
                                                    {isPast && <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg>}
                                                </div>
                                                <span className={`text-sm font-medium ${isCurrent ? 'text-sky-700 font-semibold' : 'text-slate-700'}`}>{step.label}</span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        {error && (
                            <div className="mt-4 p-4 border border-rose-200 bg-rose-50 rounded-lg flex items-start gap-3">
                                <svg className="shrink-0 text-rose-500 mt-0.5" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>
                                <p className="text-rose-700 text-sm font-medium">{error}</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* ── Right Column: Output / Dashboard ──────────────── */}
                <div className="space-y-6 animate-slide-up" style={{ animationDelay: "0.2s" }}>
                    {result ? (
                        <>
                            {/* Classification Summary */}
                            <div className="med-card p-6">
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 pb-6 border-b border-slate-100">
                                    <div>
                                        <h3 className="text-lg font-bold text-slate-900 leading-tight">Diagnostic Summary</h3>
                                        <p className="text-sm font-medium text-slate-500">{result.image_type} Model Pipeline</p>
                                    </div>
                                    <div className={`px-4 py-2 rounded-lg text-sm font-bold shadow-sm self-start sm:self-auto ${isDisease ? "bg-rose-500 border border-rose-600 text-white" : "bg-emerald-500 border border-emerald-600 text-white"
                                        }`}
                                    >
                                        VERDICT: {result.prediction}
                                    </div>
                                </div>

                                <div className="grid sm:grid-cols-2 gap-6">
                                    <div>
                                        <div className="flex justify-between text-xs font-semibold text-slate-500 uppercase mb-2 mt-1">
                                            <span>Confidence Check</span>
                                            <span className="text-sky-600">{(result.confidence * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden w-full">
                                            <div className="h-full bg-sky-500" style={{ width: `${result.confidence * 100}%` }}></div>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-xs font-semibold text-slate-500 uppercase mb-1 mt-1">
                                            <span>Raw Prob Vector</span>
                                            <span className="text-slate-800 font-mono">{result.probability.toFixed(4)}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Detailed Tabs Area */}
                            <div className="med-card overflow-hidden">
                                <div className="flex overflow-x-auto border-b border-slate-200 bg-slate-50/50">
                                    {(["classification", "heatmap", "segmentation"] as const).map(
                                        (tab) => {
                                            const disabled =
                                                (tab === "heatmap" && !result.heatmap_base64 && !result.gradcam_base64) ||
                                                (tab === "segmentation" && !result.segmentation);
                                            return (
                                                <button
                                                    key={tab}
                                                    disabled={disabled}
                                                    onClick={() => setActiveTab(tab)}
                                                    className={`flex-1 min-w-[120px] py-3.5 text-xs font-semibold uppercase tracking-wide transition-colors border-b-2
                                                        ${activeTab === tab
                                                            ? "border-sky-500 text-sky-700 bg-white"
                                                            : "border-transparent text-slate-500 hover:bg-slate-100 hover:text-slate-700"
                                                        } ${disabled ? "opacity-40 cursor-not-allowed bg-slate-50 hover:bg-slate-50" : ""}`}
                                                >
                                                    {tab}
                                                </button>
                                            );
                                        }
                                    )}
                                </div>
                                <div className="p-6 bg-white">
                                    {activeTab === "classification" && (
                                        <div className="space-y-6">
                                            {/* Feature Importance via Recharts Radar */}
                                            {result.feature_importance ? (
                                                <div className="h-[250px] w-full">
                                                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-widest text-center mb-2">Feature Importance Vector</h4>
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={featureConfidenceData}>
                                                            <PolarGrid stroke="#e2e8f0" />
                                                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 10, fontWeight: 600 }} />
                                                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                                            <Radar name="Confidence" dataKey="A" stroke="#0ea5e9" fill="#0ea5e9" fillOpacity={0.3} />
                                                        </RadarChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            ) : (
                                                <p className="text-sm text-slate-500 text-center py-8">No feature vectors provided by model output.</p>
                                            )}
                                        </div>
                                    )}

                                    {activeTab === "heatmap" && (
                                        <div className="space-y-4">
                                            <div className="flex justify-between items-center bg-slate-50 p-3 rounded-lg border border-slate-100 mb-2">
                                                <span className="text-sm font-medium text-slate-700">{result.image_type === "OCT" ? "Quantum Attention Heatmap" : "Grad-CAM Activation"}</span>
                                                <span className="text-xs text-sky-600 bg-sky-100 px-2 py-1 rounded-md">Visual Verify</span>
                                            </div>
                                            <div className="border border-slate-200 rounded-lg overflow-hidden bg-black/5 flex justify-center">
                                                <img src={`data:image/png;base64,${result.heatmap_base64 || result.gradcam_base64}`} alt="Explainability heatmap" className="max-w-full max-h-72 object-contain mix-blend-multiply" />
                                            </div>
                                        </div>
                                    )}

                                    {activeTab === "segmentation" && result.segmentation && (
                                        <div className="space-y-4">
                                            <div className="grid sm:grid-cols-2 gap-4">
                                                <div className="border border-slate-200 rounded-lg overflow-hidden p-2 bg-slate-50">
                                                    <p className="text-xs font-semibold text-slate-500 uppercase mb-2 text-center">Binary Isolate</p>
                                                    <img src={`data:image/png;base64,${result.segmentation.mask_base64}`} alt="Mask" className="w-full object-contain rounded bg-black/5" />
                                                </div>
                                                <div className="border border-slate-200 rounded-lg overflow-hidden p-2 bg-slate-50">
                                                    <p className="text-xs font-semibold text-slate-500 uppercase mb-2 text-center">Spectral Overlay</p>
                                                    <img src={`data:image/png;base64,${result.segmentation.overlay_base64}`} alt="Overlay" className="w-full object-contain rounded bg-black/5" />
                                                </div>
                                            </div>
                                            <div className="mt-4 p-4 bg-sky-50 border border-sky-100 rounded-lg flex justify-between items-center text-sky-900">
                                                <span className="text-sm font-semibold">Anomalous Area Ratio</span>
                                                <span className="font-mono text-lg font-bold">{(result.segmentation.mask_area_ratio * 100).toFixed(2)}%</span>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* ── Clinical Feedback Panel ──────────────── */}
                            <div className="med-card p-6">
                                <h3 className="text-sm font-semibold text-slate-900 mb-4 pb-2 border-b border-slate-100">Physician Oversight</h3>

                                {feedbackSent ? (
                                    <div className="text-center py-4 bg-emerald-50 rounded-lg border border-emerald-100 animate-fade-in">
                                        <svg className="w-8 h-8 text-emerald-500 mx-auto mb-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="m9 12 2 2 4-4" /><circle cx="12" cy="12" r="10" /></svg>
                                        <p className="text-sm font-bold text-emerald-800">Verification Recorded</p>
                                        <p className="text-xs text-emerald-600 mt-1">Log stored to internal database.</p>
                                    </div>
                                ) : (
                                    <div className="space-y-4">
                                        <input
                                            type="text"
                                            placeholder="Append clinical notes to log..."
                                            value={feedbackNote}
                                            onChange={(e) => setFeedbackNote(e.target.value)}
                                            className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500/50"
                                        />

                                        <div className="flex flex-col sm:flex-row gap-3">
                                            <button onClick={() => submitFeedback("accept")} className="flex-1 px-4 py-2 border border-slate-300 text-slate-700 text-sm font-semibold hover:bg-slate-50 rounded-lg transition-colors">
                                                Concur with Model
                                            </button>
                                            <button onClick={() => setShowCorrectionDropdown(true)} className="flex-1 px-4 py-2 bg-slate-800 text-white text-sm font-semibold hover:bg-slate-900 rounded-lg transition-colors">
                                                Override Diagnosis
                                            </button>
                                        </div>

                                        {showCorrectionDropdown && (
                                            <div className="border border-slate-200 p-4 rounded-lg bg-slate-50 mt-4 animate-slide-up">
                                                <p className="text-xs font-semibold text-slate-500 uppercase mb-3">Ground Truth Override:</p>
                                                <div className="grid grid-cols-2 gap-2">
                                                    {correctionOptions.map((label) => (
                                                        <button key={label} onClick={() => submitFeedback("reject", label)} className="px-3 py-2 border border-slate-300 rounded-md bg-white text-sm font-semibold text-slate-700 hover:bg-slate-100 hover:border-slate-400 transition-colors">
                                                            {label}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </>
                    ) : (
                        /* Default Dashboard View (shown before upload) */
                        <div className="w-full h-full min-h-[500px] flex flex-col space-y-6">

                            <div className="med-card p-6 flex flex-col items-center justify-center text-center flex-1 bg-gradient-to-b from-white to-slate-50/50">
                                <div className="w-16 h-16 rounded-full bg-sky-50 flex items-center justify-center mb-4 text-sky-500 ring-4 ring-white shadow-sm">
                                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                                        <polyline points="21 15 16 10 5 21"></polyline>
                                    </svg>
                                </div>
                                <h3 className="text-lg font-bold text-slate-900">Awaiting Diagnostic Input</h3>
                                <p className="text-sm font-medium text-slate-500 mt-2 max-w-sm">Provide patient scans to initiate the quantum network prediction pipeline.</p>
                            </div>

                            {/* System Monitoring Dashboard Placeholder */}
                            <div className="med-card p-6">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="text-sm font-semibold text-slate-900">Node Performance Telemetry</h3>
                                    <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Live</span>
                                </div>
                                <div className="h-[200px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={performanceData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                            <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} />
                                            <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} />
                                            <RechartsTooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }} />
                                            <Line type="monotone" dataKey="throughput" stroke="#0ea5e9" strokeWidth={2} dot={false} activeDot={{ r: 4 }} name="Throughput (Img/hr)" />
                                            <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} dot={false} name="Accuracy (%)" />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
