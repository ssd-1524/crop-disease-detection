import Link from "next/link";
import { Leaf, ArrowRight, Scan, Activity, ShieldCheck, Zap } from "lucide-react";

const T = {
  font: "'Outfit', system-ui, sans-serif",
  bg: "#09090b",
  card: "#18181b",
  cardAlt: "#111113",
  b1: "#27272a",
  b2: "#3f3f46",
  t1: "#f4f4f5",
  t2: "#a1a1aa",
  t3: "#71717a",
  t4: "#52525b",
  em: "#10b981",
  emL: "#34d399",
  emBg: "#052014",
  emBd: "#14532d",
};

const FEATURES = [
  { icon: Scan, title: "Disease Classification", desc: "Detects Blight, Common Rust, Gray Leaf Spot and more with high accuracy." },
  { icon: ShieldCheck, title: "SAM2 Segmentation", desc: "Pixel-level disease overlay using Meta's Segment Anything Model 2." },
  { icon: Activity, title: "Severity Scoring", desc: "Quantifies affected leaf area with spot and region breakdowns." },
  { icon: Zap, title: "Instant Results", desc: "Full analysis pipeline completes in seconds, not minutes." },
];

export default function HomePage() {
  return (
    <div style={{ minHeight: "100vh", background: T.bg, fontFamily: T.font, overflowX: "hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500;600;700;800&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; background: #09090b; }
        @keyframes mh-float {
          0%, 100% { transform: translateY(0px); }
          50%       { transform: translateY(-12px); }
        }
        @keyframes mh-pulse-ring {
          0%   { transform: scale(0.95); opacity: 0.6; }
          100% { transform: scale(1.4);  opacity: 0; }
        }
        @keyframes mh-fade-up {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0);    }
        }
        .mh-hero-btn:hover { background: #059669 !important; box-shadow: 0 0 40px rgba(16,185,129,0.6) !important; }
        .mh-ghost-btn:hover { background: #27272a !important; color: #f4f4f5 !important; }
        .mh-feature-card:hover { background: #1c1c20 !important; border-color: #3f3f46 !important; transform: translateY(-2px); }
        .mh-feature-card { transition: background 0.2s, border-color 0.2s, transform 0.2s; }
      `}</style>

      {/* ── Background effects ─────────────────────────────────────────── */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0 }}>
        {/* Grid */}
        <div style={{
          position: "absolute", inset: 0, opacity: 0.025,
          backgroundImage: "linear-gradient(rgba(255,255,255,.6) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.6) 1px,transparent 1px)",
          backgroundSize: "56px 56px",
        }} />
        {/* Emerald glow center */}
        <div style={{
          position: "absolute", top: "30%", left: "50%", transform: "translate(-50%,-50%)",
          width: 700, height: 700, borderRadius: "50%",
          background: "radial-gradient(circle, rgba(16,185,129,0.07) 0%, transparent 65%)",
        }} />
        {/* Bottom glow */}
        <div style={{
          position: "absolute", bottom: 0, left: "50%", transform: "translateX(-50%)",
          width: 500, height: 300, borderRadius: "50%",
          background: "radial-gradient(circle, rgba(16,185,129,0.04) 0%, transparent 70%)",
        }} />
      </div>

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section style={{
        position: "relative", zIndex: 1,
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", minHeight: "100vh",
        padding: "120px 24px 80px", textAlign: "center",
        animation: "mh-fade-up 0.6s ease-out both",
      }}>

        {/* Badge */}
        <div style={{
          display: "inline-flex", alignItems: "center", gap: 8,
          padding: "6px 14px 6px 10px", borderRadius: 99,
          background: T.emBg, border: `1px solid ${T.emBd}`,
          marginBottom: 32,
        }}>
          <div style={{
            width: 6, height: 6, borderRadius: "50%", background: T.emL,
            boxShadow: "0 0 8px rgba(52,211,153,0.9)",
          }} />
          <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: T.emL }}>
            AI-Powered Crop Health
          </span>
        </div>

        {/* Headline */}
        <h1 style={{
          margin: "0 0 20px", maxWidth: 700,
          fontSize: "clamp(36px, 6vw, 72px)",
          fontWeight: 200, lineHeight: 1.05,
          letterSpacing: "-0.03em", color: T.t1,
        }}>
          Detect maize disease{" "}
          <span style={{
            fontWeight: 800,
            background: "linear-gradient(135deg, #34d399 0%, #10b981 50%, #059669 100%)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            backgroundClip: "text",
          }}>
            instantly.
          </span>
        </h1>

        {/* Sub */}
        <p style={{
          margin: "0 0 48px", maxWidth: 480,
          fontSize: 17, fontWeight: 300, lineHeight: 1.7,
          color: T.t3, letterSpacing: "-0.01em",
        }}>
          Upload a maize leaf image and our AI detects Blight, Common Rust, Gray Leaf Spot — with SAM2 segmentation and severity scoring.
        </p>

        {/* CTA */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Link href="/login" style={{ textDecoration: "none" }}>
            <button
              className="mh-hero-btn"
              style={{
                position: "relative", height: 50, padding: "0 28px",
                borderRadius: 14, border: "none", cursor: "pointer",
                background: "#10b981", color: "#fff",
                fontSize: 15, fontWeight: 600, fontFamily: "'Outfit', system-ui, sans-serif",
                display: "flex", alignItems: "center", gap: 8,
                boxShadow: "0 0 32px rgba(16,185,129,0.35)",
                transition: "all 0.2s",
              }}
            >
              Get Started
              <ArrowRight size={16} />
            </button>
          </Link>
        </div>

        {/* Floating leaf icon */}
        <div style={{ marginTop: 72, position: "relative" }}>
          {/* Pulse ring */}
          <div style={{
            position: "absolute", inset: -16, borderRadius: "50%",
            border: `1px solid ${T.emBd}`,
            animation: "mh-pulse-ring 2.5s ease-out infinite",
          }} />
          <div style={{
            width: 72, height: 72, borderRadius: 22,
            background: T.emBg, border: `1px solid ${T.emBd}`,
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 0 40px rgba(16,185,129,0.2)",
            animation: "mh-float 4s ease-in-out infinite",
          }}>
            <Leaf size={32} color={T.emL} strokeWidth={1.5} />
          </div>
        </div>
      </section>

      {/* ── Features ──────────────────────────────────────────────────────── */}
      <section style={{
        position: "relative", zIndex: 1,
        padding: "80px 24px 120px",
      }}>
        <div style={{ maxWidth: 960, margin: "0 auto" }}>

          {/* Section label */}
          <div style={{ textAlign: "center", marginBottom: 56 }}>
            <p style={{ margin: "0 0 12px", fontSize: 11, fontWeight: 600, letterSpacing: "0.2em", textTransform: "uppercase", color: T.emL }}>
              What it does
            </p>
            <h2 style={{ margin: 0, fontSize: "clamp(24px, 4vw, 38px)", fontWeight: 200, color: T.t1, letterSpacing: "-0.02em" }}>
              Full analysis pipeline,{" "}
              <span style={{ fontWeight: 700 }}>one upload.</span>
            </h2>
          </div>

          {/* Cards grid */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
            {FEATURES.map(({ icon: Icon, title, desc }) => (
              <div
                key={title}
                className="mh-feature-card"
                style={{
                  background: T.card, border: `1px solid ${T.b1}`,
                  borderRadius: 16, padding: "24px 20px",
                  boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
                }}
              >
                <div style={{
                  width: 40, height: 40, borderRadius: 12, marginBottom: 16,
                  background: T.emBg, border: `1px solid ${T.emBd}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  <Icon size={18} color={T.emL} strokeWidth={1.5} />
                </div>
                <p style={{ margin: "0 0 8px", fontSize: 15, fontWeight: 600, color: T.t1 }}>{title}</p>
                <p style={{ margin: 0, fontSize: 13, fontWeight: 300, color: T.t3, lineHeight: 1.6 }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA banner ────────────────────────────────────────────────────── */}
      <section style={{ position: "relative", zIndex: 1, padding: "0 24px 100px" }}>
        <div style={{
          maxWidth: 720, margin: "0 auto",
          background: T.emBg, border: `1px solid ${T.emBd}`,
          borderRadius: 24, padding: "48px 40px",
          textAlign: "center",
          boxShadow: "0 0 60px rgba(16,185,129,0.08), 0 8px 40px rgba(0,0,0,0.4)",
        }}>
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
            <div style={{ width: 48, height: 48, borderRadius: 14, background: "#0a3320", border: `1px solid ${T.emBd}`, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 20px rgba(16,185,129,0.3)" }}>
              <Leaf size={22} color={T.emL} strokeWidth={1.5} />
            </div>
          </div>
          <h3 style={{ margin: "0 0 12px", fontSize: "clamp(22px, 3vw, 30px)", fontWeight: 200, color: T.t1, letterSpacing: "-0.02em" }}>
            Ready to protect your crop?
          </h3>
          <p style={{ margin: "0 0 28px", fontSize: 14, fontWeight: 300, color: T.t3, lineHeight: 1.6 }}>
            Free to use. No setup required. Just upload and go.
          </p>
          <Link href="/login" style={{ textDecoration: "none" }}>
            <button
              className="mh-hero-btn"
              style={{
                height: 48, padding: "0 28px", borderRadius: 12,
                border: "none", cursor: "pointer",
                background: T.em, color: "#fff",
                fontSize: 14, fontWeight: 600, fontFamily: T.font,
                display: "inline-flex", alignItems: "center", gap: 8,
                boxShadow: "0 0 28px rgba(16,185,129,0.35)",
                transition: "all 0.2s",
              }}
            >
              Get Started Free
              <ArrowRight size={15} />
            </button>
          </Link>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer style={{
        position: "relative", zIndex: 1,
        borderTop: `1px solid ${T.b1}`,
        padding: "20px 24px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        flexWrap: "wrap", gap: 12,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 22, height: 22, borderRadius: 7, background: T.em, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 10px rgba(16,185,129,0.4)" }}>
            <Leaf size={11} color="#fff" strokeWidth={2.5} />
          </div>
          <span style={{ fontSize: 13, fontWeight: 300, color: T.t3 }}>
            Maize<span style={{ fontWeight: 700, color: T.emL }}>Health</span>
          </span>
        </div>
        <p style={{ margin: 0, fontSize: 12, color: T.t4, fontWeight: 300 }}>
          © {new Date().getFullYear()} MaizeHealth — Precision Agriculture AI
        </p>
      </footer>
    </div>
  );
}