"use client";

import { useState, useCallback, useMemo, useRef } from "react";
import { createClient } from "@/lib/supabase/client";
import { toast } from "sonner";
import {
  UploadCloud, X, CheckCircle, AlertTriangle,
  BrainCircuit, Leaf, FlaskConical, Loader2, Microscope,
} from "lucide-react";

// ── Design tokens ──────────────────────────────────────────────────────────────
const T = {
  font: "'Outfit', system-ui, sans-serif",
  bg: "#09090b",
  card: "#18181b",
  cardAlt: "#111113",
  b1: "#27272a",   // subtle border
  b2: "#3f3f46",   // mid border
  t1: "#f4f4f5",   // text primary
  t2: "#a1a1aa",   // text secondary
  t3: "#71717a",   // text muted
  t4: "#52525b",   // text dim
  em: "#10b981",   // emerald
  emL: "#34d399",   // emerald light
  emBg: "#052014",
  emBd: "#14532d",
  am: "#f59e0b",   // amber
  amBg: "#1c1107",
  amBd: "#78350f",
  or: "#f97316",   // orange
  orBg: "#1a0a02",
  orBd: "#7c2d12",
  rd: "#ef4444",   // red
  rdBg: "#1c0606",
  rdBd: "#7f1d1d",
};

const SEV_MAP = {
  Mild: { bar: T.am, txt: T.am, bg: T.amBg, bd: T.amBd },
  Moderate: { bar: T.or, txt: T.or, bg: T.orBg, bd: T.orBd },
  Severe: { bar: T.rd, txt: T.rd, bg: T.rdBg, bd: T.rdBd },
  _: { bar: T.b2, txt: T.t3, bg: T.cardAlt, bd: T.b1 },
};
const getSev = (l) => SEV_MAP[l] ?? SEV_MAP._;

// ── Shared card shell ──────────────────────────────────────────────────────────
const card = {
  background: T.card,
  border: `1px solid ${T.b1}`,
  borderRadius: 16,
  padding: 20,
  boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
  fontFamily: T.font,
};

// ── Section header row ─────────────────────────────────────────────────────────
function SectionHead({ IconComp, iconColor, iconBg, iconBd, title, sub }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
      <div style={{
        width: 32, height: 32, borderRadius: 10, flexShrink: 0,
        background: iconBg, border: `1px solid ${iconBd}`,
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <IconComp size={15} color={iconColor} strokeWidth={1.5} />
      </div>
      <div>
        <p style={{ margin: 0, fontSize: 14, fontWeight: 600, color: T.t1 }}>{title}</p>
        <p style={{ margin: 0, fontSize: 11, fontWeight: 300, color: T.t3 }}>{sub}</p>
      </div>
    </div>
  );
}

// ── Loading steps ─────────────────────────────────────────────────────────────
const STEPS = [
  { label: "Classifying disease", sub: "Running CNN inference", icon: FlaskConical },
  { label: "Segmenting leaf", sub: "SAM2 segmentation model", icon: Leaf },
  { label: "Measuring severity", sub: "Pixel-level analysis", icon: Microscope },
];

function LoadingSteps({ current }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, padding: "4px 0" }}>
      {STEPS.map((step, i) => {
        const Icon = step.icon;
        const done = i < current, active = i === current, future = i > current;
        return (
          <div key={step.label} style={{
            display: "flex", alignItems: "center", gap: 12,
            padding: "10px 12px", borderRadius: 12,
            background: active ? T.emBg : "transparent",
            border: `1px solid ${active ? T.emBd : "transparent"}`,
            opacity: future ? 0.25 : 1,
          }}>
            <div style={{
              width: 32, height: 32, borderRadius: 10, flexShrink: 0,
              display: "flex", alignItems: "center", justifyContent: "center",
              background: done ? T.emBg : active ? T.em : T.cardAlt,
              border: `1px solid ${done ? T.emBd : active ? T.em : T.b1}`,
              boxShadow: active ? "0 0 12px rgba(16,185,129,0.4)" : "none",
            }}>
              {active
                ? <Loader2 size={14} color="#fff" style={{ animation: "mh-spin 0.8s linear infinite" }} />
                : done
                  ? <CheckCircle size={14} color={T.emL} strokeWidth={1.5} />
                  : <Icon size={14} color={T.t4} strokeWidth={1.5} />}
            </div>
            <div>
              <p style={{
                margin: 0, fontSize: 13,
                fontWeight: active ? 500 : 400,
                color: done ? T.t4 : active ? T.t1 : T.t4,
                textDecoration: done ? "line-through" : "none",
              }}>
                {step.label}
              </p>
              {active && (
                <p style={{ margin: 0, fontSize: 11, color: T.emL, fontWeight: 300, marginTop: 2 }}>
                  {step.sub}
                </p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Drop zone ─────────────────────────────────────────────────────────────────
function DropZone({ preview, onFile, onClear, inputRef }) {
  const [drag, setDrag] = useState(false);

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files[0];
    if (f && (f.type === "image/jpeg" || f.type === "image/png")) onFile(f);
    else toast.error("JPEG or PNG only.");
  }, [onFile]);

  const baseStyle = {
    position: "relative", display: "flex", alignItems: "center", justifyContent: "center",
    height: 240, borderRadius: 16, cursor: preview ? "default" : "pointer",
    border: `2px dashed ${drag ? T.em : T.b2}`,
    background: drag ? T.emBg : T.cardAlt,
    boxShadow: drag ? "0 0 28px rgba(16,185,129,0.12)" : "none",
    transition: "border-color 0.2s, background 0.2s, box-shadow 0.2s",
    overflow: "hidden",
  };
  if (preview) { baseStyle.border = `1px solid ${T.b1}`; baseStyle.cursor = "default"; }

  return (
    <div
      style={baseStyle}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      onClick={() => !preview && inputRef.current?.click()}
    >
      {preview ? (
        <>
          <img src={preview} alt="Preview" style={{ maxHeight: 220, borderRadius: 12, objectFit: "contain" }} />
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); onClear(); }}
            style={{
              position: "absolute", top: 12, right: 12,
              width: 28, height: 28, borderRadius: 8, border: `1px solid ${T.b2}`,
              background: T.card, cursor: "pointer", color: T.t3,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}
            onMouseEnter={e => { e.currentTarget.style.background = T.rdBg; e.currentTarget.style.color = T.rd; e.currentTarget.style.borderColor = T.rdBd; }}
            onMouseLeave={e => { e.currentTarget.style.background = T.card; e.currentTarget.style.color = T.t3; e.currentTarget.style.borderColor = T.b2; }}
          >
            <X size={14} />
          </button>
        </>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16, padding: "0 24px", pointerEvents: "none", textAlign: "center" }}>
          <div style={{
            width: 56, height: 56, borderRadius: 16, flexShrink: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
            background: drag ? T.emBg : T.card,
            border: `2px dashed ${drag ? T.em : T.b2}`,
          }}>
            <UploadCloud size={24} color={drag ? T.emL : T.t4} strokeWidth={1.5} />
          </div>
          <div>
            <p style={{ margin: 0, fontSize: 14, fontWeight: 500, color: T.t2 }}>
              Drop image here or{" "}
              <span style={{ color: T.emL }}>browse files</span>
            </p>
            <p style={{ margin: "4px 0 0", fontSize: 12, color: T.t4, fontWeight: 300 }}>
              JPEG · PNG supported
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main ───────────────────────────────────────────────────────────────────────
export default function ImageUploader() {
  const [file, setFile] = useState(null);
  const [prev, setPrev] = useState(null);
  const [res, setRes] = useState(null);
  const [load, setLoad] = useState(false);
  const [step, setStep] = useState(0);
  const [err, setErr] = useState(null);
  const inputRef = useRef(null);
  const supabase = useMemo(() => createClient(), []);

  const handleFile = useCallback((f) => {
    setFile(f); setPrev(URL.createObjectURL(f)); setRes(null); setErr(null);
  }, []);

  const handleClear = () => {
    setFile(null); setPrev(null); setRes(null); setErr(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoad(true); setStep(0); setErr(null); setRes(null);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error("You must be logged in.");
      const fp = `${user.id}/${Date.now()}_${file.name}`;
      const { error: ue } = await supabase.storage.from("maize-images").upload(fp, file);
      if (ue) throw new Error(`Upload failed: ${ue.message}`);
      setStep(1);
      const form = new FormData(); form.append("file", file);
      const url = process.env.NEXT_PUBLIC_API_URL ? `${process.env.NEXT_PUBLIC_API_URL}/predict` : "http://127.0.0.1:8000/predict";
      const r = await fetch(url, { method: "POST", body: form });
      if (!r.ok) { const d = await r.json().catch(() => { }); throw new Error(d?.detail || `API error ${r.status}`); }
      setStep(2);
      const data = await r.json();
      if (data.error) throw new Error(data.error);
      await supabase.from("analyses").insert({
        image_path: fp, prediction: data.prediction, confidence: data.confidence,
        severity_percentage: data.severity_percentage, severity_label: data.severity_label,
        sam_mask_image: data.sam_mask_image ?? null, spot_count: data.spot_count ?? 0,
        region_count: data.region_count ?? 0, spot_severity_pct: data.spot_severity_pct ?? 0,
        region_severity_pct: data.region_severity_pct ?? 0,
      });
      setRes(data); toast.success("Analysis complete.");
    } catch (e) {
      const msg = e.message || "Unknown error."; setErr(msg); toast.error(msg);
    } finally { setLoad(false); setStep(0); }
  };

  const isHealthy = res?.prediction === "Healthy";
  const sev = res ? getSev(res.severity_label) : null;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500;600;700&display=swap');
        @keyframes mh-spin { to { transform: rotate(360deg); } }
        @keyframes mh-fade { from { opacity:0; transform: translateY(4px); } to { opacity:1; transform: translateY(0); } }
      `}</style>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, fontFamily: T.font }}>

        {/* Upload card */}
        <div style={card}>
          <SectionHead IconComp={UploadCloud} iconColor={T.emL} iconBg={T.emBg} iconBd={T.emBd} title="Upload Image" sub="Maize leaf photo" />

          <DropZone preview={prev} onFile={handleFile} onClear={handleClear} inputRef={inputRef} />

          <input ref={inputRef} type="file" accept="image/jpeg,image/png"
            onChange={e => { if (e.target.files[0]) handleFile(e.target.files[0]); }}
            style={{ display: "none" }} />

          <button
            onClick={handleAnalyze}
            disabled={!file || load}
            style={{
              marginTop: 14, width: "100%", height: 44, borderRadius: 12, border: "none",
              background: !file || load ? "#10b98150" : T.em,
              color: "#fff", fontSize: 14, fontWeight: 600, cursor: !file || load ? "not-allowed" : "pointer",
              fontFamily: T.font, display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
              boxShadow: !file || load ? "none" : "0 0 24px rgba(16,185,129,0.3)",
              transition: "all 0.2s",
            }}
          >
            {load
              ? <><Loader2 size={16} style={{ animation: "mh-spin 0.8s linear infinite" }} /> Analysing…</>
              : "Analyse Image"
            }
          </button>

          {err && (
            <div style={{ marginTop: 12, display: "flex", alignItems: "flex-start", gap: 10, padding: "12px 14px", borderRadius: 12, background: T.rdBg, border: `1px solid ${T.rdBd}` }}>
              <AlertTriangle size={15} color={T.rd} style={{ flexShrink: 0, marginTop: 1 }} strokeWidth={1.5} />
              <p style={{ margin: 0, fontSize: 12, color: T.rd, fontWeight: 300, lineHeight: 1.5 }}>{err}</p>
            </div>
          )}
        </div>

        {/* Result card */}
        <div style={card}>
          <SectionHead IconComp={BrainCircuit} iconColor={T.t3} iconBg={T.cardAlt} iconBd={T.b1} title="Analysis Result" sub="AI diagnostic output" />

          {load && <LoadingSteps current={step} />}

          {!load && res && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14, animation: "mh-fade 0.25s ease-out" }}>

              {/* Prediction */}
              <div style={{
                display: "flex", alignItems: "center", gap: 14, padding: "14px 16px", borderRadius: 14,
                background: isHealthy ? T.emBg : "#1a0c02",
                border: `1px solid ${isHealthy ? T.emBd : "#7c3311"}`,
              }}>
                <div style={{
                  width: 44, height: 44, borderRadius: 12, flexShrink: 0,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  background: isHealthy ? "#0a3320" : "#2d1206",
                  boxShadow: isHealthy ? "0 0 16px rgba(16,185,129,0.25)" : "none",
                }}>
                  {isHealthy
                    ? <CheckCircle size={20} color={T.emL} strokeWidth={1.5} />
                    : <AlertTriangle size={20} color="#fb923c" strokeWidth={1.5} />}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ margin: 0, fontSize: 10, fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase", color: T.t4 }}>Prediction</p>
                  <p style={{ margin: "3px 0 0", fontSize: 20, fontWeight: 700, color: T.t1, lineHeight: 1.2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {res.prediction.replace(/_/g, " ")}
                  </p>
                </div>
                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <p style={{ margin: 0, fontSize: 10, fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase", color: T.t4 }}>Confidence</p>
                  <p style={{ margin: "3px 0 0", fontSize: 16, fontWeight: 600, color: T.t2 }}>{res.confidence}</p>
                </div>
              </div>

              {/* Severity */}
              {!isHealthy && res.severity_percentage != null && (
                <div style={{ padding: "14px 16px", borderRadius: 12, background: T.cardAlt, border: `1px solid ${T.b1}` }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                    <p style={{ margin: 0, fontSize: 10, fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase", color: T.t4 }}>Disease Severity</p>
                    <span style={{ fontSize: 11, fontWeight: 600, padding: "3px 10px", borderRadius: 99, background: sev.bg, color: sev.txt, border: `1px solid ${sev.bd}` }}>
                      {res.severity_label}
                    </span>
                  </div>
                  <div style={{ width: "100%", height: 6, background: T.b2, borderRadius: 99, overflow: "hidden" }}>
                    <div style={{ height: "100%", borderRadius: 99, background: sev.bar, width: `${res.severity_percentage}%`, transition: "width 0.7s cubic-bezier(0.4,0,0.2,1)" }} />
                  </div>
                  <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginTop: 8 }}>
                    <p style={{ margin: 0, fontSize: 11, color: T.t4, fontWeight: 300 }}>Affected leaf area</p>
                    <p style={{ margin: 0, fontSize: 28, fontWeight: 200, color: sev.txt, letterSpacing: "-0.02em", lineHeight: 1 }}>
                      {res.severity_percentage}<span style={{ fontSize: 13, color: T.t4 }}>%</span>
                    </p>
                  </div>
                  {(res.spot_count > 0 || res.region_count > 0) && (
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 12, paddingTop: 12, borderTop: `1px solid ${T.b1}` }}>
                      {[{ l: "Spots", n: res.spot_count, p: res.spot_severity_pct, dot: "#fbbf24" }, { l: "Regions", n: res.region_count, p: res.region_severity_pct, dot: "#f87171" }]
                        .map(({ l, n, p, dot }) => (
                          <div key={l} style={{ background: T.card, borderRadius: 10, padding: "10px 12px" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                              <div style={{ width: 8, height: 8, borderRadius: "50%", background: dot, flexShrink: 0 }} />
                              <p style={{ margin: 0, fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: T.t4 }}>{l}</p>
                            </div>
                            <p style={{ margin: 0, fontSize: 18, fontWeight: 600, color: T.t2 }}>{n}</p>
                            <p style={{ margin: "2px 0 0", fontSize: 11, color: T.t4 }}>{p}% coverage</p>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              )}

              {/* SAM2 */}
              {res.sam_mask_image && (
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
                    <BrainCircuit size={12} color={T.t4} strokeWidth={1.5} />
                    <p style={{ margin: 0, fontSize: 10, fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase", color: T.t4 }}>SAM2 Overlay</p>
                  </div>
                  <div style={{ borderRadius: 12, overflow: "hidden", border: `1px solid ${T.b1}` }}>
                    <img src={`data:image/jpeg;base64,${res.sam_mask_image}`} alt="SAM2" style={{ width: "100%", display: "block" }} />
                  </div>
                </div>
              )}
            </div>
          )}

          {!load && !res && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 200, gap: 16 }}>
              <div style={{ width: 64, height: 64, borderRadius: 20, background: T.cardAlt, border: `1px solid ${T.b1}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                <BrainCircuit size={28} color={T.b2} strokeWidth={1.5} />
              </div>
              <div style={{ textAlign: "center" }}>
                <p style={{ margin: 0, fontSize: 14, color: T.t3, fontWeight: 300 }}>Awaiting analysis</p>
                <p style={{ margin: "4px 0 0", fontSize: 12, color: T.t4, fontWeight: 300 }}>Upload an image to begin</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <style>{`
        @media (max-width: 767px) {
          .mh-uploader { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </>
  );
}