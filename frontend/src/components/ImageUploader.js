"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { UploadCloud, X, CheckCircle, AlertTriangle } from "lucide-react";
import { Progress } from "@/components/ui/progress";

export default function ImageUploader() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

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

    // --- START: Real API Call Logic ---
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error analyzing image:", error);
      setError("Failed to analyze image. Please try again.");
    } finally {
      setIsLoading(false);
    }
    // --- END: Real API Call Logic ---
  };

  const ResultIcon =
    result?.prediction === "Healthy" ? CheckCircle : AlertTriangle;
  const resultColor =
    result?.prediction === "Healthy" ? "text-green-500" : "text-orange-500";

  return (
    <div className="w-full max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <UploadCloud className="w-6 h-6" /> Upload Leaf Image
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="h-64 border-2 border-dashed rounded-lg flex items-center justify-center text-center">
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
              className="w-full"
            >
              {isLoading ? "Analyzing..." : "Analyze Image"}
            </Button>
            {error && (
              <p className="text-red-500 text-sm text-center">{error}</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Analysis Result</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <p>Loading results...</p>
            </div>
          ) : result ? (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <ResultIcon className={`w-10 h-10 ${resultColor}`} />
                <div>
                  <p className="text-sm text-gray-500">Prediction</p>
                  <p className="text-2xl font-bold">
                    {result.prediction.replace("_", " ")}
                  </p>
                </div>
              </div>
              <div>
                <p className="text-sm text-gray-500">Confidence</p>
                <p className="text-lg font-semibold">{result.confidence}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Severity</p>
                <Progress
                  value={result.severity_percentage}
                  className="w-full mt-2"
                />
                <p className="text-lg font-semibold text-right mt-1">
                  {result.severity_percentage}%
                </p>
              </div>
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
