import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function HomePage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 text-center p-4">
      <h1 className="text-4xl md:text-5xl font-bold text-gray-800">
        Maize Crop Disease Detection
      </h1>
      <p className="mt-4 text-lg text-gray-600 max-w-xl">
        Upload an image of a maize leaf, and our AI will detect diseases like
        Blight, Common Rust, and Gray Leaf Spot.
      </p>
      <Link href="/login" className="mt-8">
        <Button size="lg">Get Started</Button>
      </Link>
    </div>
  );
}
