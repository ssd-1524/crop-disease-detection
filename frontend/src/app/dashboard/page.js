import ImageUploader from "@/components/ImageUploader";
import HistorySidebar from "@/components/HistorySidebar";

export default function DashboardPage() {
  return (
    <div className="container mx-auto p-4 md:p-8 pt-24">
      <div className="text-center mb-12">
        <h1 className="text-3xl md:text-4xl font-bold">
          Disease & Severity Analysis
        </h1>
        <p className="text-gray-600 mt-2">
          Upload an image of a maize leaf to get an instant AI-powered analysis.
        </p>
      </div>

      {/* New Responsive Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <ImageUploader />
        </div>
        <div className="lg:col-span-1">
          <HistorySidebar />
        </div>
      </div>
    </div>
  );
}
