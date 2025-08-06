"use client";

import { useState, useEffect, useCallback } from "react";
import { createClient } from "@/lib/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { History } from "lucide-react";
import Link from "next/link";

const SkeletonLoader = () => (
  <div className="space-y-4">
    {[...Array(3)].map((_, i) => (
      <div key={i} className="flex items-center space-x-4 animate-pulse">
        <div className="h-16 w-16 rounded-lg bg-gray-300"></div>
        <div className="space-y-2">
          <div className="h-4 w-32 rounded bg-gray-300"></div>
          <div className="h-3 w-24 rounded bg-gray-300"></div>
        </div>
      </div>
    ))}
  </div>
);

export default function HistorySidebar() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const supabase = createClient();

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (user) {
      const { data: analyses, error } = await supabase
        .from("analyses")
        .select("*")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false });

      if (error) {
        console.error("Error fetching history:", error);
        setHistory([]);
      } else {
        // --- GENERATE SIGNED URLS ---
        const historyWithSignedUrls = await Promise.all(
          analyses.map(async (item) => {
            if (!item.image_path) return { ...item, signedImageUrl: null };

            const { data, error: urlError } = await supabase.storage
              .from("maize-images")
              .createSignedUrl(item.image_path, 3600); // URL valid for 1 hour

            if (urlError) {
              console.error("Error creating signed URL:", urlError);
              return { ...item, signedImageUrl: null };
            }
            return { ...item, signedImageUrl: data.signedUrl };
          })
        );
        setHistory(historyWithSignedUrls);
      }
    }
    setLoading(false);
  }, [supabase]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  return (
    // 1. Make the Card a flex container that fills the height of its grid cell
    <Card className="h-full flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <History className="w-6 h-6" />
          Analysis History
        </CardTitle>
      </CardHeader>
      {/* 2. Make the content area grow to fill the remaining space */}
      <CardContent className="flex-grow overflow-y-auto">
        {/* 3. Remove the fixed height from the ScrollArea */}
        <ScrollArea className="h-full pr-4">
          {loading ? (
            <SkeletonLoader />
          ) : history.length > 0 ? (
            <div className="space-y-2">
              {history.map((item) => (
                <Link key={item.id} href={`/dashboard/analysis/${item.id}`}>
                  <div className="flex items-center p-2 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer">
                    {item.signedImageUrl ? (
                      <img
                        src={item.signedImageUrl}
                        alt="Analyzed leaf"
                        className="w-16 h-16 rounded-md object-cover mr-4"
                      />
                    ) : (
                      <div className="w-16 h-16 rounded-md bg-gray-200 flex items-center justify-center mr-4 text-xs text-center">
                        No Image
                      </div>
                    )}
                    <div>
                      <p className="font-bold text-lg">
                        {item.prediction?.replace("_", " ") || "N/A"}
                      </p>
                      <p className="text-sm text-gray-500">
                        {new Date(item.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center mt-8">
              No past analyses found.
            </p>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
