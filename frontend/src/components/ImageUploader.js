"use client";

import { useState, useCallback } from "react";
import { createClient } from "@/lib/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";
import {
  UploadCloud,
  X,
  CheckCircle,
  AlertTriangle,
  BrainCircuit,
  Leaf,
  FlaskConical,
} from "lucide-react";

// ── Spinner ────────────────────────────────────────────────────────────────────
const Spinner = ({ size = 20 }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={size}
    height={size}
    viewBox="0 0 24 24"      // ← fixed: was "0 0 24" (missing height)
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="animate-spin"
  >
    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
  </svg>
);

// ── Severity colour helper ─────────────────────────────────────────────────────
const getSeverityColor = (label) => {
  switch (label) {
    case "Mild":     return "text-yellow-500";
    case "Moderate": return "text-orange-500";
    case "Severe":   return "text-red-600";
    default:         return "text-gray-500";
  }
};

const getSeverityBarColor = (pct) => {
  if (pct < 5)  return "bg-yellow-400";
  if (pct < 15) return "bg-orange-500";
  return "bg-red-600";
};

// ── Drag-and-drop zone ─────────────────────────────────────────────────────────
const DropZone = ({ preview, onFile, onClear }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && (dropped.type === "image/jpeg" || dropped.type === "image/png")) {
      onFile(dropped);
    } else {
      toast.error("Please upload a JPEG or PNG image.");
    }
  }, [onFile]);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`relative h-64 border-2 border-dashed rounded-xl flex items-center
        justify-center text-center transition-colors duration-200
        ${isDragging
          ? "border-green-500 bg-green-50"
          : preview
          ? "border-transparent bg-gray-100"
          : "border-gray-300 bg-gray-50/50 hover:border-gray-400"
        }`}
    >
      {preview ? (
        <>
          <img
            src={preview}
            alt="Leaf preview"
            className="max-h-60 object-contain rounded-lg"
          />
          <button
            onClick={onClear}
            aria-label="Remove image"
            className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white
              rounded-full p-1 shadow transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </>
      ) : (
        <div className="text-gray-400 space-y-2 pointer-events-none">
          <UploadCloud className="w-10 h-10 mx-auto text-gray-300" />
          <p className="text-sm font-medium">Drag & drop or click to upload</p>
          <p className="text-xs">JPEG / PNG supported</p>
        </div>
      )}
    </div>
  );
};

// ── Analysing steps indicator ──────────────────────────────────────────────────
const steps = [
  { label: "Classifying disease",   icon: FlaskConical },
  { label: "Segmenting leaf (SAM2)", icon: Leaf        },
  { label: "Measuring severity",     icon: BrainCircuit },
];

const LoadingSteps = ({ currentStep }) => (
  <div className="space-y-3 py-4">
    {steps.map((step, i) => {
      const Icon = step.icon;
      const done    = i < currentStep;
      const active  = i === currentStep;
      return (
        <div
          key={step.label}
          className={`flex items-center gap-3 text-sm transition-opacity duration-300
            ${i > currentStep ? "opacity-30" : "opacity-100"}`}
        >
          <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0
            ${done   ? "bg-green-100 text-green-600"
            : active ? "bg-blue-100 text-blue-600"
                     : "bg-gray-100 text-gray-400"}`}
          >
            {active ? <Spinner size={14} /> : <Icon className="w-4 h-4" />}
          </div>
          <span className={done ? "line-through text-gray-400" : active ? "font-medium" : ""}>
            {step.label}
          </span>
          {done && <CheckCircle className="w-4 h-4 text-green-500 ml-auto" />}
        </div>
      );
    })}
  </div>
);

// ── Main component ─────────────────────────────────────────────────────────────
export default function ImageUploader() {
  const [file,        setFile       ] = useState(null);
  const [preview,     setPreview    ] = useState(null);
  const [result,      setResult     ] = useState(null);
  const [isLoading,   setIsLoading  ] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [error,       setError      ] = useState(null);

  const supabase = createClient();

  const handleFile = useCallback((selected) => {
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  }, []);

  const handleFileInput = (e) => {
    const selected = e.target.files[0];
    if (selected) handleFile(selected);
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    const input = document.getElementById("leaf-image-input");
    if (input) input.value = "";
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsLoading(true);
    setLoadingStep(0);
    setError(null);
    setResult(null);

    try {
      // 1. Auth check
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error("You must be logged in to analyse images.");

      // 2. Upload to Supabase Storage
      const filePath = `${user.id}/${Date.now()}_${file.name}`;
      const { error: uploadError } = await supabase.storage
        .from("maize-images")
        .upload(filePath, file);
      if (uploadError) throw new Error(`Storage Error: ${uploadError.message}`);

      // 3. Call FastAPI backend
      setLoadingStep(1);
      const formData = new FormData();
      formData.append("file", file);

      const apiUrl = process.env.NEXT_PUBLIC_API_URL
        ? `${process.env.NEXT_PUBLIC_API_URL}/predict`
        : "http://127.0.0.1:8000/predict";

      const response = await fetch(apiUrl, { method: "POST", body: formData });

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail?.detail || `API error ${response.status}`);
      }

      setLoadingStep(2);
      const analysisResult = await response.json();
      if (analysisResult.error) throw new Error(analysisResult.error);

      // 4. Save to database
      const { error: insertError } = await supabase.from("analyses").insert({
        image_path:          filePath,
        prediction:          analysisResult.prediction,
        confidence:          analysisResult.confidence,
        severity_percentage: analysisResult.severity_percentage,
        severity_label:      analysisResult.severity_label,
      });
      if (insertError) throw new Error(`Database Error: ${insertError.message}`);

      setResult(analysisResult);
      toast.success("Analysis complete!");

    } catch (err) {
      console.error(err);
      const msg = err.message || "An unknown error occurred.";
      setError(msg);
      toast.error(msg);
    } finally {
      setIsLoading(false);
      setLoadingStep(0);
    }
  };

  const isHealthy     = result?.prediction === "Healthy";
  const ResultIcon    = isHealthy ? CheckCircle : AlertTriangle;
  const resultColor   = isHealthy ? "text-green-500" : "text-orange-500";
  const severityColor = result ? getSeverityColor(result.severity_label) : "";

  return (
    <div className="w-full max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">

      {/* ── Upload Card ───────────────────────────────────────────────────────── */}
      <Card className="transition-all hover:shadow-lg hover:-translate-y-1">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <UploadCloud className="w-5 h-5" />
            Upload Leaf Image
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <DropZone preview={preview} onFile={handleFile} onClear={handleClear} />

          <Input
            id="leaf-image-input"
            type="file"
            accept="image/jpeg,image/png"
            onChange={handleFileInput}
            className="cursor-pointer"
          />

          <Button
            onClick={handleAnalyze}
            disabled={!file || isLoading}
            className="w-full flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <><Spinner size={18} /> Analysing...</>
            ) : (
              "Analyse Image"
            )}
          </Button>

          {error && (
            <p className="text-red-500 text-sm text-center bg-red-50 rounded-lg p-3">
              {error}
            </p>
          )}
        </CardContent>
      </Card>

      {/* ── Results Card ──────────────────────────────────────────────────────── */}
      <Card className="transition-all hover:shadow-lg hover:-translate-y-1">
        <CardHeader>
          <CardTitle>Analysis Result</CardTitle>
        </CardHeader>
        <CardContent>

          {/* Loading state */}
          {isLoading && <LoadingSteps currentStep={loadingStep} />}

          {/* Results */}
          {!isLoading && result && (
            <div className="space-y-5">

              {/* Prediction + confidence */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <ResultIcon className={`w-10 h-10 flex-shrink-0 ${resultColor}`} />
                  <div>
                    <p className="text-xs text-gray-400 uppercase tracking-wide">
                      Prediction
                    </p>
                    <p className="text-2xl font-bold leading-tight">
                      {result.prediction.replace(/_/g, " ")}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xs text-gray-400 uppercase tracking-wide">
                    Confidence
                  </p>
                  <p className="text-xl font-semibold">{result.confidence}</p>
                </div>
              </div>

              {/* Severity (non-healthy only) */}
              {!isHealthy && (
                <div className="space-y-2 bg-gray-50 rounded-xl p-4">
                  <div className="flex justify-between items-center">
                    <p className="text-xs text-gray-400 uppercase tracking-wide">
                      Disease Severity
                    </p>
                    <span className={`text-sm font-semibold ${severityColor}`}>
                      {result.severity_label}
                    </span>
                  </div>
                  {/* Custom coloured progress bar */}
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                      className={`h-3 rounded-full transition-all duration-700
                        ${getSeverityBarColor(result.severity_percentage)}`}
                      style={{ width: `${result.severity_percentage}%` }}
                    />
                  </div>
                  <p className="text-right text-lg font-bold">
                    {result.severity_percentage}%
                  </p>
                </div>
              )}

              {/* SAM2 overlay */}
              {result.sam_mask_image && (
                <div className="space-y-2">
                  <p className="text-xs text-gray-400 uppercase tracking-wide flex items-center gap-1.5">
                    <BrainCircuit className="w-4 h-4" />
                    SAM2 Disease Overlay
                  </p>
                  <div className="rounded-xl overflow-hidden border border-gray-100 bg-gray-900">
                    <img
                      src={`data:image/jpeg;base64,${result.sam_mask_image}`}
                      alt="SAM2 disease segmentation overlay"
                      className="w-full h-auto"
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Empty state */}
          {!isLoading && !result && (
            <div className="flex flex-col items-center justify-center h-48 text-gray-400 gap-2">
              <BrainCircuit className="w-10 h-10 text-gray-200" />
              <p className="text-sm">Results will appear here after analysis.</p>
            </div>
          )}

        </CardContent>
      </Card>
    </div>
  );
}
