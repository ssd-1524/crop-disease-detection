import ImageUploader from "@/components/ImageUploader";

export default function DashboardPage() {
  return (
    <div className="container mx-auto p-4 md:p-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl md:text-4xl font-bold">
          Disease & Severity Analysis
        </h1>
        <p className="text-gray-600 mt-2">
          Upload an image of a maize leaf to get an instant AI-powered analysis.
        </p>
      </div>
      <ImageUploader />
    </div>
  );
}
