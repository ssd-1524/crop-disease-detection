import ImageUploader from "@/components/ImageUploader";
import HistorySidebar from "@/components/HistorySidebar";

const S = {
  font: "'Outfit', system-ui, sans-serif",
};

export default function DashboardPage() {
  return (
    <div style={{ minHeight: "100vh", background: "#09090b", fontFamily: S.font }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500;600;700&display=swap');
        body { background: #09090b !important; }
        @media (max-width: 1023px) {
          .mh-dash-grid { grid-template-columns: 1fr !important; }
          .mh-dash-main { grid-column: span 1 !important; }
        }
      `}</style>

      <div style={{ maxWidth: 1120, margin: "0 auto", padding: "112px 24px 64px" }}>

        {/* Header */}
        <div style={{ marginBottom: 36 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
            <div style={{
              width: 6, height: 6, borderRadius: "50%", background: "#34d399",
              boxShadow: "0 0 8px rgba(52,211,153,0.9)",
            }} />
            <span style={{
              fontSize: 11, fontWeight: 600, letterSpacing: "0.18em",
              textTransform: "uppercase", color: "#34d399", fontFamily: S.font,
            }}>
              Analysis Studio
            </span>
          </div>
          <h1 style={{
            margin: 0, fontSize: 32, fontWeight: 200, color: "#f4f4f5",
            letterSpacing: "-0.02em", lineHeight: 1.2, fontFamily: S.font,
          }}>
            Disease &amp; Severity{" "}
            <span style={{ fontWeight: 700 }}>Analysis</span>
          </h1>
          <p style={{ margin: "8px 0 0", fontSize: 14, color: "#71717a", fontWeight: 300, fontFamily: S.font }}>
            Upload a maize leaf image for instant AI-powered diagnosis.
          </p>
        </div>

        {/* Grid */}
        <div
          className="mh-dash-grid"
          style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, alignItems: "start" }}
        >
          <div className="mh-dash-main" style={{ gridColumn: "span 2" }}>
            <ImageUploader />
          </div>
          <div style={{ position: "sticky", top: 80 }}>
            <HistorySidebar />
          </div>
        </div>
      </div>
    </div>
  );
}