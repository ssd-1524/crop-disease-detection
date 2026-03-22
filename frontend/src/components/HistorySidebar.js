"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { createClient } from "@/lib/supabase/client";
import { History, ChevronRight, Clock } from "lucide-react";
import Link from "next/link";

const T = {
  font: "'Outfit', system-ui, sans-serif",
  card: "#18181b", cardAlt: "#111113",
  b1: "#27272a", b2: "#3f3f46",
  t1: "#f4f4f5", t2: "#a1a1aa", t3: "#71717a", t4: "#52525b",
  emL: "#34d399", emBg: "#052014", emBd: "#14532d",
};

const DOT = { Healthy: "#34d399", Mild: "#fbbf24", Moderate: "#fb923c", Severe: "#f87171" };
const dot = (pred, label) => pred === "Healthy" ? DOT.Healthy : (DOT[label] ?? T.b2);

function Skeleton() {
  return (
    <>
      <style>{`@keyframes mh-pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }`}</style>
      {[...Array(4)].map((_, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 12px", animation: "mh-pulse 1.8s ease-in-out infinite" }}>
          <div style={{ width: 44, height: 44, borderRadius: 12, background: T.b2, flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <div style={{ height: 12, width: "65%", background: T.b2, borderRadius: 99, marginBottom: 8 }} />
            <div style={{ height: 10, width: "40%", background: T.b1, borderRadius: 99 }} />
          </div>
        </div>
      ))}
    </>
  );
}

export default function HistorySidebar() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const supabase = useMemo(() => createClient(), []);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    const { data: { user } } = await supabase.auth.getUser();
    if (user) {
      const { data: analyses, error } = await supabase
        .from("analyses").select("*").eq("user_id", user.id)
        .order("created_at", { ascending: false });
      if (!error) {
        const withUrls = await Promise.all(analyses.map(async (item) => {
          if (!item.image_path) return { ...item, signedImageUrl: null };
          const { data, error: e } = await supabase.storage.from("maize-images").createSignedUrl(item.image_path, 3600);
          return { ...item, signedImageUrl: e ? null : data.signedUrl };
        }));
        setHistory(withUrls);
      }
    }
    setLoading(false);
  }, [supabase]);

  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  return (
    <div style={{
      background: T.card, border: `1px solid ${T.b1}`, borderRadius: 16,
      overflow: "hidden", fontFamily: T.font,
      boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
    }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 20px", borderBottom: `1px solid ${T.b1}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 28, height: 28, borderRadius: 9, background: T.cardAlt, border: `1px solid ${T.b2}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            <History size={14} color={T.t3} strokeWidth={1.5} />
          </div>
          <div>
            <p style={{ margin: 0, fontSize: 14, fontWeight: 600, color: T.t1 }}>History</p>
            <p style={{ margin: 0, fontSize: 11, fontWeight: 300, color: T.t3 }}>Past analyses</p>
          </div>
        </div>
        {!loading && history.length > 0 && (
          <span style={{ fontSize: 11, fontWeight: 600, color: T.t4, background: T.cardAlt, border: `1px solid ${T.b2}`, padding: "3px 10px", borderRadius: 99 }}>
            {history.length}
          </span>
        )}
      </div>

      {/* List */}
      <div style={{ maxHeight: 520, overflowY: "auto", padding: "8px 12px" }}>
        {loading ? <Skeleton /> : history.length > 0 ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {history.map((item) => (
              <Link key={item.id} href={`/dashboard/analysis/${item.id}`}
                style={{ textDecoration: "none", color: "inherit", display: "block" }}>
                <div
                  style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 12px", borderRadius: 12, transition: "background 0.15s", cursor: "pointer" }}
                  onMouseEnter={e => e.currentTarget.style.background = T.b1}
                  onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                >
                  <div style={{ position: "relative", flexShrink: 0 }}>
                    {item.signedImageUrl
                      ? <img src={item.signedImageUrl} alt="Leaf" style={{ width: 44, height: 44, borderRadius: 12, objectFit: "cover", border: `1px solid ${T.b1}`, display: "block" }} />
                      : <div style={{ width: 44, height: 44, borderRadius: 12, background: T.cardAlt, border: `1px solid ${T.b2}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, color: T.t4 }}>N/A</div>
                    }
                    <div style={{ position: "absolute", bottom: -2, right: -2, width: 10, height: 10, borderRadius: "50%", background: dot(item.prediction, item.severity_label), border: `2px solid ${T.card}` }} />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ margin: 0, fontSize: 13, fontWeight: 500, color: T.t1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {item.prediction?.replace(/_/g, " ") || "—"}
                    </p>
                    <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 2 }}>
                      <Clock size={10} color={T.t4} />
                      <p style={{ margin: 0, fontSize: 11, color: T.t4, fontWeight: 300 }}>
                        {new Date(item.created_at).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}
                      </p>
                    </div>
                  </div>
                  <ChevronRight size={14} color={T.b2} />
                </div>
              </Link>
            ))}
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "48px 24px", gap: 12 }}>
            <div style={{ width: 48, height: 48, borderRadius: 16, background: T.cardAlt, border: `1px solid ${T.b1}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <History size={20} color={T.b2} strokeWidth={1.5} />
            </div>
            <div style={{ textAlign: "center" }}>
              <p style={{ margin: 0, fontSize: 14, color: T.t3, fontWeight: 300 }}>No analyses yet</p>
              <p style={{ margin: "4px 0 0", fontSize: 12, color: T.t4, fontWeight: 300 }}>Upload an image to get started</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}