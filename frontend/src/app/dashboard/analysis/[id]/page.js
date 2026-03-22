"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { ArrowLeft, CheckCircle, AlertTriangle, BrainCircuit, Calendar, FlaskConical, Microscope } from "lucide-react";
import Link from "next/link";

const T = {
  font: "'Outfit', system-ui, sans-serif",
  bg: "#09090b", card: "#18181b", cardAlt: "#111113",
  b1: "#27272a", b2: "#3f3f46",
  t1: "#f4f4f5", t2: "#a1a1aa", t3: "#71717a", t4: "#52525b",
  em: "#10b981", emL: "#34d399", emBg: "#052014", emBd: "#14532d",
  am: "#f59e0b", amBg: "#1c1107", amBd: "#78350f",
  or: "#f97316", orBg: "#1a0a02", orBd: "#7c2d12",
  rd: "#ef4444", rdBg: "#1c0606", rdBd: "#7f1d1d",
};

const SEV = {
  Mild: { bar: "#f59e0b", txt: "#f59e0b", bg: "#1c1107", bd: "#78350f" },
  Moderate: { bar: "#f97316", txt: "#f97316", bg: "#1a0a02", bd: "#7c2d12" },
  Severe: { bar: "#ef4444", txt: "#ef4444", bg: "#1c0606", bd: "#7f1d1d" },
  _: { bar: T.b2, txt: T.t3, bg: T.cardAlt, bd: T.b1 },
};

function Skel({ w = "100%", h = 16 }) {
  return <div style={{ width: w, height: h, borderRadius: 12, background: T.b2, opacity: 0.5 }} />;
}

function StatCard({ icon: Icon, label, children }) {
  return (
    <div style={{ background: T.cardAlt, border: `1px solid ${T.b1}`, borderRadius: 12, padding: "14px 16px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
        <Icon size={12} color={T.t4} strokeWidth={1.5} />
        <p style={{ margin: 0, fontSize: 10, fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase", color: T.t4 }}>{label}</p>
      </div>
      {children}
    </div>
  );
}

export default function AnalysisDetailPage() {
  const { id } = useParams();
  const supabase = createClient();
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const run = async () => {
      if (!id) return;
      setLoading(true);
      const { data, error: e } = await supabase.from("analyses").select("*").eq("id", id).single();
      if (e) { setError("Failed to load analysis."); setLoading(false); return; }
      const { data: u, error: ue } = await supabase.storage.from("maize-images").createSignedUrl(data.image_path, 3600);
      setAnalysis({ ...data, signedImageUrl: ue ? null : u.signedUrl });
      setLoading(false);
    };
    run();
  }, [id]);

  if (loading) return (
    <div style={{ minHeight: "100vh", background: T.bg, fontFamily: T.font }}>
      <div style={{ maxWidth: 960, margin: "0 auto", padding: "112px 24px 64px" }}>
        <Skel w={100} h={16} />
        <div style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: 16, marginTop: 32 }}>
          <Skel h={320} /><div style={{ display: "flex", flexDirection: "column", gap: 12 }}><Skel h={80} /><Skel h={60} /><Skel h={100} /><Skel h={48} /></div>
        </div>
      </div>
    </div>
  );

  if (error || !analysis) return (
    <div style={{ minHeight: "100vh", background: T.bg, fontFamily: T.font, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 20 }}>
      <div style={{ width: 56, height: 56, borderRadius: 18, background: T.rdBg, border: `1px solid ${T.rdBd}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <AlertTriangle size={24} color={T.rd} strokeWidth={1.5} />
      </div>
      <p style={{ margin: 0, fontSize: 14, color: T.t3, fontWeight: 300 }}>{error ?? "Analysis not found."}</p>
      <Link href="/dashboard" style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 13, color: T.t3, textDecoration: "none", fontWeight: 500 }}>
        <ArrowLeft size={13} /> Back to Dashboard
      </Link>
    </div>
  );

  const isHealthy = analysis.prediction === "Healthy";
  const pred = analysis.prediction?.replace(/_/g, " ") ?? "—";
  const sev = SEV[analysis.severity_label] ?? SEV._;
  const showSev = !isHealthy && analysis.severity_percentage != null;
  const date = new Date(analysis.created_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

  return (
    <div style={{ minHeight: "100vh", background: T.bg, fontFamily: T.font }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500;600;700&display=swap');`}</style>
      <div style={{ maxWidth: 960, margin: "0 auto", padding: "112px 24px 64px" }}>

        <Link href="/dashboard" style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: T.t4, textDecoration: "none", fontWeight: 500, marginBottom: 28 }}>
          <ArrowLeft size={13} /> Back to Dashboard
        </Link>

        <div style={{ marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: T.emL, boxShadow: "0 0 8px rgba(52,211,153,0.9)" }} />
            <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.18em", textTransform: "uppercase", color: T.emL }}>Analysis Report</span>
          </div>
          <h1 style={{ margin: "0 0 6px", fontSize: 30, fontWeight: 200, color: T.t1, letterSpacing: "-0.02em" }}>{pred}</h1>
          <p style={{ margin: 0, fontSize: 13, color: T.t4, fontWeight: 300 }}>{date}</p>
        </div>

        {/* 5-col grid */}
        <div style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: 16, alignItems: "start" }}>

          {/* Image */}
          <div style={{ background: T.card, border: `1px solid ${T.b1}`, borderRadius: 16, overflow: "hidden", boxShadow: "0 8px 32px rgba(0,0,0,0.4)" }}>
            <div style={{ padding: "14px 20px", borderBottom: `1px solid ${T.b1}` }}>
              <p style={{ margin: 0, fontSize: 10, fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase", color: T.t4 }}>Uploaded Image</p>
            </div>
            {analysis.signedImageUrl
              ? <img src={analysis.signedImageUrl} alt={pred} style={{ width: "100%", display: "block" }} />
              : <div style={{ height: 280, background: T.bg, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12 }}>
                <BrainCircuit size={32} color={T.b2} strokeWidth={1.5} />
                <p style={{ margin: 0, fontSize: 12, color: T.t4, fontWeight: 300 }}>Image unavailable</p>
              </div>
            }
          </div>

          {/* Details */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {/* Prediction */}
            <div style={{ padding: 16, borderRadius: 16, display: "flex", alignItems: "center", gap: 14, background: isHealthy ? T.emBg : "#1a0c02", border: `1px solid ${isHealthy ? T.emBd : "#7c3311"}` }}>
              <div style={{ width: 44, height: 44, borderRadius: 12, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", background: isHealthy ? "#0a3320" : "#2d1206", boxShadow: isHealthy ? "0 0 16px rgba(16,185,129,0.2)" : "none" }}>
                {isHealthy ? <CheckCircle size={20} color={T.emL} strokeWidth={1.5} /> : <AlertTriangle size={20} color="#fb923c" strokeWidth={1.5} />}
              </div>
              <div>
                <p style={{ margin: 0, fontSize: 10, fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase", color: T.t4 }}>Prediction</p>
                <p style={{ margin: "3px 0 0", fontSize: 18, fontWeight: 700, color: T.t1, lineHeight: 1.2 }}>{pred}</p>
              </div>
            </div>

            <StatCard icon={FlaskConical} label="Confidence">
              <p style={{ margin: 0, fontSize: 24, fontWeight: 200, color: T.t1, letterSpacing: "-0.02em" }}>{analysis.confidence}</p>
            </StatCard>

            {showSev && (
              <div style={{ background: T.card, border: `1px solid ${T.b1}`, borderRadius: 12, padding: "14px 16px" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <Microscope size={12} color={T.t4} strokeWidth={1.5} />
                    <p style={{ margin: 0, fontSize: 10, fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase", color: T.t4 }}>Severity</p>
                  </div>
                  <span style={{ fontSize: 11, fontWeight: 600, padding: "3px 10px", borderRadius: 99, background: sev.bg, color: sev.txt, border: `1px solid ${sev.bd}` }}>
                    {analysis.severity_label}
                  </span>
                </div>
                <div style={{ width: "100%", height: 6, background: T.b2, borderRadius: 99, overflow: "hidden" }}>
                  <div style={{ height: "100%", borderRadius: 99, background: sev.bar, width: `${analysis.severity_percentage}%`, transition: "width 0.7s ease" }} />
                </div>
                <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginTop: 8 }}>
                  <p style={{ margin: 0, fontSize: 11, color: T.t4, fontWeight: 300 }}>Affected area</p>
                  <p style={{ margin: 0, fontSize: 28, fontWeight: 200, color: sev.txt, letterSpacing: "-0.02em", lineHeight: 1 }}>
                    {analysis.severity_percentage}<span style={{ fontSize: 13, color: T.t4 }}>%</span>
                  </p>
                </div>
                {(analysis.spot_count > 0 || analysis.region_count > 0) && (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 12, paddingTop: 12, borderTop: `1px solid ${T.b1}` }}>
                    {[{ l: "Spots", n: analysis.spot_count ?? 0, p: analysis.spot_severity_pct ?? 0, dot: "#fbbf24" }, { l: "Regions", n: analysis.region_count ?? 0, p: analysis.region_severity_pct ?? 0, dot: "#f87171" }]
                      .map(({ l, n, p, dot }) => (
                        <div key={l} style={{ background: T.cardAlt, borderRadius: 10, padding: "10px 12px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                            <div style={{ width: 8, height: 8, borderRadius: "50%", background: dot }} />
                            <p style={{ margin: 0, fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: T.t4 }}>{l}</p>
                          </div>
                          <p style={{ margin: 0, fontSize: 16, fontWeight: 600, color: T.t2 }}>{n}</p>
                          <p style={{ margin: "2px 0 0", fontSize: 11, color: T.t4 }}>{p}% area</p>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            )}

            <StatCard icon={Calendar} label="Analysed On">
              <p style={{ margin: 0, fontSize: 13, color: T.t2, fontWeight: 300 }}>{date}</p>
            </StatCard>
          </div>
        </div>

        {/* SAM2 */}
        {analysis.sam_mask_image && (
          <div style={{ marginTop: 16, background: T.card, border: `1px solid ${T.b1}`, borderRadius: 16, overflow: "hidden", boxShadow: "0 8px 32px rgba(0,0,0,0.4)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "14px 20px", borderBottom: `1px solid ${T.b1}` }}>
              <div style={{ width: 24, height: 24, borderRadius: 8, background: T.emBg, border: `1px solid ${T.emBd}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                <BrainCircuit size={12} color={T.emL} strokeWidth={1.5} />
              </div>
              <p style={{ margin: 0, fontSize: 10, fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase", color: T.t4 }}>SAM2 Disease Segmentation Overlay</p>
            </div>
            <div style={{ background: T.bg }}>
              <img src={`data:image/jpeg;base64,${analysis.sam_mask_image}`} alt="SAM2" style={{ width: "100%", display: "block" }} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}