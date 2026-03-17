"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ArrowLeft,
  CheckCircle,
  AlertTriangle,
  BrainCircuit,
  Calendar,
  Percent,
  FlaskConical,
  Circle,
  Layers,
} from "lucide-react";
import Link from "next/link";

// ── Skeleton ───────────────────────────────────────────────────────────────────
const SkeletonBlock = ({ className }) => (
  <div className={`animate-pulse rounded-lg bg-gray-200 ${className}`} />
);

const SkeletonLoader = () => (
  <div className="space-y-8">
    <SkeletonBlock className="h-7 w-40" />
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      <SkeletonBlock className="h-96 w-full" />
      <div className="space-y-6">
        <SkeletonBlock className="h-12 w-3/4" />
        <SkeletonBlock className="h-8 w-1/2" />
        <SkeletonBlock className="h-10 w-full" />
        <SkeletonBlock className="h-8 w-1/3" />
        <SkeletonBlock className="h-8 w-2/3" />
      </div>
    </div>
  </div>
);

// ── Severity helpers (mirrors ImageUploader) ───────────────────────────────────
const getSeverityColor = (label) => {
  switch (label) {
    case "Mild": return "text-yellow-500";
    case "Moderate": return "text-orange-500";
    case "Severe": return "text-red-600";
    default: return "text-gray-400";
  }
};

const getSeverityBarColor = (pct) => {
  if (pct < 5) return "bg-yellow-400";
  if (pct < 15) return "bg-orange-500";
  return "bg-red-600";
};

// ── Stat row ───────────────────────────────────────────────────────────────────
const StatRow = ({ icon: Icon, label, children }) => (
  <div className="flex items-start gap-3">
    <div className="mt-0.5 w-8 h-8 rounded-lg bg-gray-100 flex items-center justify-center flex-shrink-0">
      <Icon className="w-4 h-4 text-gray-500" />
    </div>
    <div className="flex-1 min-w-0">
      <p className="text-xs text-gray-400 uppercase tracking-wide mb-0.5">{label}</p>
      {children}
    </div>
  </div>
);

// ── Main page ──────────────────────────────────────────────────────────────────
export default function AnalysisDetailPage() {
  const params = useParams();
  const id = params.id;
  const supabase = createClient();

  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAnalysis = async () => {
      if (!id) return;
      setLoading(true);

      const { data, error: fetchError } = await supabase
        .from("analyses")
        .select("*")
        .eq("id", id)
        .single();

      if (fetchError) {
        console.error("Error fetching analysis:", fetchError);
        setError("Failed to load analysis details.");
        setLoading(false);
        return;
      }

      // Signed URL — non-fatal if it fails
      const { data: urlData, error: urlError } = await supabase.storage
        .from("maize-images")
        .createSignedUrl(data.image_path, 3600);

      setAnalysis({
        ...data,
        signedImageUrl: urlError ? null : urlData.signedUrl,
      });
      setLoading(false);
    };

    fetchAnalysis();
  }, [id]);   // ← removed `supabase` from deps (stable client reference)

  // ── Loading ────────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="container mx-auto p-4 md:p-8 pt-24">
        <SkeletonLoader />
      </div>
    );
  }

  // ── Error ──────────────────────────────────────────────────────────────────
  if (error || !analysis) {
    return (
      <div className="container mx-auto p-4 md:p-8 pt-24 text-center">
        <BackLink />
        <div className="mt-12 flex flex-col items-center gap-3 text-gray-500">
          <AlertTriangle className="w-10 h-10 text-red-400" />
          <p className="text-red-500 font-medium">{error ?? "Analysis not found."}</p>
        </div>
      </div>
    );
  }

  const isHealthy = analysis.prediction === "Healthy";
  const ResultIcon = isHealthy ? CheckCircle : AlertTriangle;
  const resultColor = isHealthy ? "text-green-500" : "text-orange-500";
  const severityColor = getSeverityColor(analysis.severity_label);
  const showSeverity = !isHealthy && analysis.severity_percentage != null;
  const formattedDate = new Date(analysis.created_at).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });

  // ── Replace underscores for display ───────────────────────────────────────
  const predictionLabel = analysis.prediction?.replace(/_/g, " ") ?? "—";

  return (
    <div className="container mx-auto p-4 md:p-8 pt-24">
      <BackLink />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start mt-6">

        {/* ── Image Card ──────────────────────────────────────────────────── */}
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle className="text-base">Uploaded Image</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {analysis.signedImageUrl ? (
              <img
                src={analysis.signedImageUrl}
                alt={`Leaf image — ${predictionLabel}`}
                className="w-full h-auto object-cover"
              />
            ) : (
              <div className="w-full h-72 bg-gray-100 flex flex-col items-center
                justify-center gap-2 text-gray-400">
                <BrainCircuit className="w-8 h-8 text-gray-300" />
                <p className="text-sm">Image unavailable</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* ── Details Card ────────────────────────────────────────────────── */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Analysis Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">

            {/* Prediction */}
            <div className="flex items-center gap-4 p-4 rounded-xl bg-gray-50">
              <ResultIcon className={`w-12 h-12 flex-shrink-0 ${resultColor}`} />
              <div>
                <p className="text-xs text-gray-400 uppercase tracking-wide">Prediction</p>
                <p className="text-2xl font-bold leading-tight">{predictionLabel}</p>
              </div>
            </div>

            {/* Confidence */}
            <StatRow icon={FlaskConical} label="Confidence">
              <p className="text-xl font-semibold">{analysis.confidence}</p>
            </StatRow>

            {/* Severity — hidden for Healthy */}
            {showSeverity && (
              <StatRow icon={Percent} label="Disease Severity">
                <div className="space-y-2 mt-1">
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                      className={`h-3 rounded-full transition-all duration-700
                        ${getSeverityBarColor(analysis.severity_percentage)}`}
                      style={{ width: `${analysis.severity_percentage}%` }}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className={`text-sm font-semibold ${severityColor}`}>
                      {analysis.severity_label}
                    </span>
                    <span className="text-xl font-bold">
                      {analysis.severity_percentage}%
                    </span>
                  </div>

                  {/* Spot / Region breakdown */}
                  {(analysis.spot_count > 0 || analysis.region_count > 0) && (
                    <div className="grid grid-cols-2 gap-3 pt-2 border-t border-gray-200 mt-2">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-orange-400 flex-shrink-0" />
                        <div>
                          <p className="text-xs text-gray-400">Spots</p>
                          <p className="text-sm font-semibold">
                            {analysis.spot_count ?? 0} ({analysis.spot_severity_pct ?? 0}%)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-red-500 flex-shrink-0" />
                        <div>
                          <p className="text-xs text-gray-400">Regions</p>
                          <p className="text-sm font-semibold">
                            {analysis.region_count ?? 0} ({analysis.region_severity_pct ?? 0}%)
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </StatRow>
            )}

            {/* Date */}
            <StatRow icon={Calendar} label="Analysed On">
              <p className="text-base">{formattedDate}</p>
            </StatRow>

          </CardContent>
        </Card>
      </div>

      {/* ── SAM2 Overlay Card (full-width, below the grid) ──────────────── */}
      {analysis.sam_mask_image && (
        <Card className="mt-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <BrainCircuit className="w-5 h-5" />
              SAM2 Disease Overlay
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="rounded-b-xl overflow-hidden bg-gray-900">
              <img
                src={`data:image/jpeg;base64,${analysis.sam_mask_image}`}
                alt="SAM2 disease segmentation overlay"
                className="w-full h-auto"
              />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ── Back link ──────────────────────────────────────────────────────────────────
const BackLink = () => (
  <Link
    href="/dashboard"
    className="inline-flex items-center gap-2 text-sm text-gray-500
      hover:text-gray-900 transition-colors"
  >
    <ArrowLeft className="w-4 h-4" />
    Back to Dashboard
  </Link>
);
