"use client";

import { useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  UploadCloud,
  X,
  CheckCircle,
  AlertTriangle,
  BrainCircuit,
} from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";

const Spinner = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
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

export default function ImageUploader() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const supabase = createClient();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const {
        data: { user },
      } = await supabase.auth.getUser();
      if (!user)
        throw new Error("Authentication Error: You must be logged in.");

      const filePath = `${user.id}/${Date.now()}_${file.name}`;
      const { error: uploadError } = await supabase.storage
        .from("maize-images")
        .upload(filePath, file);
      if (uploadError) throw new Error("Failed to upload image to storage.");

      const formData = new FormData();
      formData.append("file", file);
      const apiUrl = `${process.env.NEXT_PUBLIC_API_URL}/predict`;
      const response = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Analysis API request failed.");

      const analysisResult = await response.json();
      if (analysisResult.error) throw new Error(analysisResult.error);

      setResult(analysisResult);

      const {
        data: { publicUrl },
      } = supabase.storage.from("maize-images").getPublicUrl(filePath);

      const { error: insertError } = await supabase.from("analyses").insert({
        image_path: filePath,
        prediction: analysisResult.prediction,
        confidence: analysisResult.confidence,
        severity_percentage: analysisResult.severity_percentage,
        severity_label: analysisResult.severity_label,
        sam_mask_image: analysisResult.sam_mask_image, // <-- Save the new mask data
      });

      if (insertError) throw new Error(insertError.message);

      toast.success("Analysis complete and results saved!");
    } catch (error) {
      console.error("A critical error occurred in handleAnalyze:", error);
      setError(error.message);
      toast.error(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const ResultIcon =
    result?.prediction === "Healthy" ? CheckCircle : AlertTriangle;
  const resultColor =
    result?.prediction === "Healthy" ? "text-green-500" : "text-orange-500";

  return (
    <div className="w-full max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
      <Card className="transition-all hover:shadow-lg hover:-translate-y-1">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <UploadCloud className="w-6 h-6" /> Upload Leaf Image
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="h-64 border-2 border-dashed rounded-lg flex items-center justify-center text-center bg-gray-50/50">
              {preview ? (
                <div className="relative">
                  <img
                    src={preview}
                    alt="Image preview"
                    className="max-h-60 object-contain rounded-md"
                  />
                  <Button
                    variant="destructive"
                    size="icon"
                    className="absolute -top-2 -right-2 h-7 w-7 rounded-full"
                    onClick={handleClear}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ) : (
                <p className="text-gray-500">Image preview will appear here</p>
              )}
            </div>
            <Input
              id="picture"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="cursor-pointer"
            />
            <Button
              onClick={handleAnalyze}
              disabled={!file || isLoading}
              className="w-full flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Spinner /> Analyzing...
                </>
              ) : (
                "Analyze Image"
              )}
            </Button>
            {error && (
              <p className="text-red-500 text-sm text-center">{error}</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="transition-all hover:shadow-lg hover:-translate-y-1">
        <CardHeader>
          <CardTitle>Analysis Result</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex flex-col items-center justify-center h-full gap-4">
              <Spinner />
              <p className="text-gray-500">Loading results...</p>
            </div>
          ) : result ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <ResultIcon className={`w-10 h-10 ${resultColor}`} />
                  <div>
                    <p className="text-sm text-gray-500">Prediction</p>
                    <p className="text-2xl font-bold">
                      {result.prediction.replace("_", " ")}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-500">Confidence</p>
                  <p className="text-lg font-semibold">{result.confidence}</p>
                </div>
              </div>

              <div>
                <p className="text-sm text-gray-500">Severity</p>
                <div className="flex items-center gap-4">
                  <Progress
                    value={result.severity_percentage}
                    className="w-full"
                  />
                  <span className="font-bold text-lg">
                    {result.severity_percentage}%
                  </span>
                </div>
                <p className="text-right font-medium text-gray-600">
                  {result.severity_label}
                </p>
              </div>

              {/* --- NEW: Display SAM Mask Image --- */}
              {result.sam_mask_image && (
                <div>
                  <p className="text-sm text-gray-500 flex items-center gap-2 mb-2">
                    <BrainCircuit className="w-4 h-4" />
                    AI Detected Area (SAM Mask)
                  </p>
                  <div className="bg-gray-900 rounded-lg p-2">
                    <img
                      src={`data:image/png;base64,${result.sam_mask_image}`}
                      alt="SAM segmentation mask"
                      className="rounded-md w-full h-auto"
                    />
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <p>Results will be displayed here after analysis.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
