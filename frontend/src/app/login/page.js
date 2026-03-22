"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { toast } from "sonner";
import { Leaf, ArrowRight, Loader2, Scan, Activity, ShieldCheck } from "lucide-react";

const T = {
  font: "'Outfit', system-ui, sans-serif",
  bg: "#09090b", panelBg: "#0f1012",
  card: "#18181b", b1: "#27272a", b2: "#3f3f46",
  t1: "#f4f4f5", t2: "#a1a1aa", t3: "#71717a", t4: "#52525b",
  em: "#10b981", emL: "#34d399", emBg: "#052014", emBd: "#14532d",
};

const FEATURES = [
  { icon: Scan, text: "SAM2 segmentation overlay" },
  { icon: Activity, text: "Real-time severity scoring" },
  { icon: ShieldCheck, text: "Multi-disease classification" },
];

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState("login");
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const supabase = createClient();

  const handleSubmit = async () => {
    if (!email || !password) return;
    setLoading(true);
    if (mode === "login") {
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) toast.error(error.message);
      else { toast.success("Welcome back."); router.push("/dashboard"); }
    } else {
      const { error } = await supabase.auth.signUp({ email, password, options: { emailRedirectTo: `${location.origin}/auth/callback` } });
      if (error) toast.error(error.message);
      else toast.success("Check your email to confirm.");
    }
    setLoading(false);
  };

  const inputStyle = {
    display: "block", width: "100%", height: 44,
    padding: "0 16px", borderRadius: 12,
    background: T.card, border: `1px solid ${T.b2}`,
    color: T.t1, fontFamily: T.font, fontSize: 14, fontWeight: 300,
    outline: "none", boxSizing: "border-box", caretColor: T.em,
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500;600;700&display=swap');
        @keyframes mh-spin { to { transform: rotate(360deg); } }
        .mh-login-left { display: none; }
        @media (min-width: 1024px) { .mh-login-left { display: flex !important; } .mh-login-mbrand { display: none !important; } }
        .mh-input:focus { border-color: #10b981 !important; box-shadow: 0 0 0 3px rgba(16,185,129,0.1) !important; }
        .mh-input::placeholder { color: #52525b; }
      `}</style>

      <div style={{ minHeight: "100vh", background: T.bg, display: "flex", fontFamily: T.font }}>

        {/* Left panel */}
        <div className="mh-login-left" style={{
          flex: "0 0 52%", position: "relative",
          flexDirection: "column", justifyContent: "space-between", padding: 48,
          background: T.panelBg, overflow: "hidden",
        }}>
          {/* Grid bg */}
          <div style={{
            position: "absolute", inset: 0, opacity: 0.04, pointerEvents: "none",
            backgroundImage: "linear-gradient(rgba(255,255,255,.5) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.5) 1px,transparent 1px)",
            backgroundSize: "48px 48px"
          }} />
          {/* Glow */}
          <div style={{
            position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)", width: 480, height: 480, borderRadius: "50%", pointerEvents: "none",
            background: "radial-gradient(circle, rgba(16,185,129,0.07) 0%, transparent 70%)"
          }} />
          {/* Leaf art */}
          <div style={{ position: "absolute", right: 0, top: "50%", transform: "translateY(-50%)", opacity: 0.1, pointerEvents: "none" }}>
            <svg width="280" height="380" viewBox="0 0 280 380" fill="none">
              <path d="M140 18C210 55 268 130 250 225 232 320 160 372 140 372 120 372 48 320 30 225 12 130 70 55 140 18Z" stroke="#10B981" strokeWidth="1" fill="none" />
              <path d="M140 55C200 85 242 148 226 225 210 302 158 348 140 348 122 348 80 302 54 225 38 148 80 85 140 55Z" stroke="#10B981" strokeWidth="0.5" fill="none" strokeDasharray="4 6" />
              <line x1="140" y1="18" x2="140" y2="372" stroke="#10B981" strokeWidth="0.5" strokeDasharray="3 8" />
              <line x1="30" y1="210" x2="250" y2="210" stroke="#10B981" strokeWidth="0.5" strokeDasharray="3 8" />
              {[80, 130, 180, 230].map(y => <circle key={y} cx="140" cy={y} r="3" fill="#10B981" opacity="0.5" />)}
            </svg>
          </div>

          {/* Brand */}
          <div style={{ position: "relative", display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 32, height: 32, borderRadius: 10, background: T.em, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 20px rgba(16,185,129,0.5)" }}>
              <Leaf size={16} color="#fff" strokeWidth={2.5} />
            </div>
            <span style={{ fontSize: 15, fontWeight: 300, color: T.t1 }}>
              Maize<span style={{ fontWeight: 700, color: T.emL }}>Health</span>
            </span>
          </div>

          {/* Headline */}
          <div style={{ position: "relative" }}>
            <p style={{ margin: "0 0 12px", fontSize: 11, fontWeight: 600, letterSpacing: "0.2em", textTransform: "uppercase", color: T.emL }}>Precision Agriculture AI</p>
            <h1 style={{ margin: "0 0 16px", fontSize: 38, fontWeight: 200, color: T.t1, lineHeight: 1.1, letterSpacing: "-0.02em" }}>
              Diagnose crop disease<br />
              <span style={{ fontWeight: 700, color: T.emL }}>at the speed of sight.</span>
            </h1>
            <p style={{ margin: "0 0 28px", fontSize: 14, color: T.t3, fontWeight: 300, lineHeight: 1.6, maxWidth: 320 }}>
              Upload a leaf image and receive instant AI classification, SAM2 segmentation, and severity analysis.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {FEATURES.map(({ icon: Icon, text }) => (
                <div key={text} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{ width: 28, height: 28, borderRadius: 9, flexShrink: 0, background: T.emBg, border: `1px solid ${T.emBd}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <Icon size={13} color={T.emL} strokeWidth={1.5} />
                  </div>
                  <p style={{ margin: 0, fontSize: 13, color: T.t2, fontWeight: 300 }}>{text}</p>
                </div>
              ))}
            </div>
          </div>

          <p style={{ position: "relative", fontSize: 11, color: T.t4, fontWeight: 300, margin: 0 }}>
            © {new Date().getFullYear()} MaizeHealth
          </p>
        </div>

        {/* Right form */}
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 32, background: T.bg }}>
          <div style={{ width: "100%", maxWidth: 360 }}>

            {/* Mobile brand */}
            <div className="mh-login-mbrand" style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 40 }}>
              <div style={{ width: 28, height: 28, borderRadius: 9, background: T.em, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 14px rgba(16,185,129,0.4)" }}>
                <Leaf size={14} color="#fff" strokeWidth={2.5} />
              </div>
              <span style={{ fontSize: 15, fontWeight: 300, color: T.t1 }}>
                Maize<span style={{ fontWeight: 700, color: T.emL }}>Health</span>
              </span>
            </div>

            <div style={{ marginBottom: 32 }}>
              <h2 style={{ margin: "0 0 6px", fontSize: 26, fontWeight: 300, color: T.t1, letterSpacing: "-0.02em" }}>
                {mode === "login" ? "Welcome back" : "Get started"}
              </h2>
              <p style={{ margin: 0, fontSize: 14, color: T.t3, fontWeight: 300 }}>
                {mode === "login" ? "Sign in to your account." : "Create a free account in seconds."}
              </p>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <label style={{ display: "block", fontSize: 11, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: T.t4, marginBottom: 6 }}>
                  Email address
                </label>
                <input type="email" value={email} onChange={e => setEmail(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleSubmit()}
                  placeholder="you@example.com"
                  className="mh-input" style={inputStyle} />
              </div>
              <div>
                <label style={{ display: "block", fontSize: 11, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: T.t4, marginBottom: 6 }}>
                  Password
                </label>
                <input type="password" value={password} onChange={e => setPassword(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleSubmit()}
                  placeholder="••••••••"
                  className="mh-input" style={inputStyle} />
              </div>
              <button
                onClick={handleSubmit}
                disabled={loading || !email || !password}
                style={{
                  marginTop: 4, width: "100%", height: 44, borderRadius: 12, border: "none",
                  background: loading || !email || !password ? "#10b98150" : T.em,
                  color: "#fff", fontSize: 14, fontWeight: 600, cursor: loading || !email || !password ? "not-allowed" : "pointer",
                  fontFamily: T.font, display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                  boxShadow: "0 0 24px rgba(16,185,129,0.25)",
                }}
              >
                {loading
                  ? <Loader2 size={16} style={{ animation: "mh-spin 0.8s linear infinite" }} />
                  : <>{mode === "login" ? "Sign in" : "Create account"}<ArrowRight size={14} /></>
                }
              </button>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "24px 0" }}>
              <div style={{ flex: 1, height: 1, background: T.b1 }} />
              <span style={{ fontSize: 11, color: T.t4 }}>or</span>
              <div style={{ flex: 1, height: 1, background: T.b1 }} />
            </div>

            <p style={{ textAlign: "center", fontSize: 14, color: T.t3, fontWeight: 300, margin: 0 }}>
              {mode === "login" ? "No account yet? " : "Already have one? "}
              <button
                onClick={() => setMode(mode === "login" ? "signup" : "login")}
                style={{ background: "none", border: "none", cursor: "pointer", color: T.emL, fontWeight: 600, fontSize: 14, fontFamily: T.font, padding: 0 }}
              >
                {mode === "login" ? "Sign up free" : "Sign in"}
              </button>
            </p>
          </div>
        </div>
      </div>
    </>
  );
}