"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  ArrowLeft,
  CheckCircle,
  AlertTriangle,
  BrainCircuit,
} from "lucide-react";
import Link from "next/link";

const SkeletonLoader = () => (
  <div className="animate-pulse">
    <div className="h-8 w-1/3 rounded bg-gray-300 mb-8"></div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      <div className="w-full h-96 rounded-lg bg-gray-300"></div>
      <div className="space-y-8">
        <div className="h-12 w-3/4 rounded bg-gray-300"></div>
        <div className="h-8 w-1/2 rounded bg-gray-300"></div>
        <div className="h-10 w-full rounded bg-gray-300"></div>
        <div className="h-8 w-1/2 rounded bg-gray-300"></div>
      </div>
    </div>
  </div>
);

export default function AnalysisDetailPage() {
  const params = useParams();
  const id = params.id;

  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const supabase = createClient();

  useEffect(() => {
    const fetchAnalysis = async () => {
      if (!id) return;

      setLoading(true);
      const { data, error } = await supabase
        .from("analyses")
        .select("*")
        .eq("id", id)
        .single();

      if (error) {
        console.error("Error fetching analysis:", error);
        setError("Failed to load analysis details.");
      } else {
        const { data: urlData, error: urlError } = await supabase.storage
          .from("maize-images")
          .createSignedUrl(data.image_path, 3600); // URL valid for 1 hour

        if (urlError) {
          setError("Failed to load image.");
          setAnalysis({ ...data, signedImageUrl: null });
        } else {
          setAnalysis({ ...data, signedImageUrl: urlData.signedUrl });
        }
      }
      setLoading(false);
    };

    fetchAnalysis();
  }, [id, supabase]);

  if (loading) {
    return (
      <div className="container mx-auto p-4 md:p-8 pt-24">
        <SkeletonLoader />
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="container mx-auto p-4 md:p-8 pt-24 text-center">
        <Link
          href="/dashboard"
          className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-800 mb-6 justify-center"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Dashboard
        </Link>
        <p className="text-red-500">{error || "Analysis not found."}</p>
      </div>
    );
  }

  const ResultIcon =
    analysis.prediction === "Healthy" ? CheckCircle : AlertTriangle;
  const resultColor =
    analysis.prediction === "Healthy" ? "text-green-500" : "text-orange-500";

  return (
    <div className="container mx-auto p-4 md:p-8 pt-24">
      <Link
        href="/dashboard"
        className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-800 mb-6"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Dashboard
      </Link>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
        <Card>
          <CardHeader>
            <CardTitle>Uploaded Image</CardTitle>
          </CardHeader>
          <CardContent>
            {analysis.signedImageUrl ? (
              <img
                src={analysis.signedImageUrl}
                alt="Analyzed leaf"
                className="w-full h-auto rounded-lg object-cover"
              />
            ) : (
              <div className="w-full h-96 rounded-lg bg-gray-200 flex items-center justify-center">
                Image not available
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Analysis Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center gap-3">
              <ResultIcon className={`w-12 h-12 ${resultColor}`} />
              <div>
                <p className="text-sm text-gray-500">Prediction</p>
                <p className="text-3xl font-bold">
                  {analysis.prediction?.replace("_", " ")}
                </p>
              </div>
            </div>
            <div>
              <p className="text-sm text-gray-500">Confidence</p>
              <p className="text-2xl font-semibold">{analysis.confidence}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Severity</p>
              <Progress
                value={analysis.severity_percentage}
                className="w-full mt-2 h-3"
              />
              <div className="flex justify-between items-center mt-1">
                <span className="font-medium text-gray-600">
                  {analysis.severity_label}
                </span>
                <span className="text-2xl font-semibold">
                  {analysis.severity_percentage}%
                </span>
              </div>
            </div>
            <div>
              <p className="text-sm text-gray-500">Analyzed On</p>
              <p className="text-lg">
                {new Date(analysis.created_at).toLocaleString()}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
