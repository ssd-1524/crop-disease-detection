"use client";

import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import LogoutButton from "./logoutButton";
import { Leaf } from "lucide-react";

export default function Navbar() {
  const { user, loading } = useAuth();

  return (
    <>
      {/* Inject Outfit font once */}
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500;600;700&display=swap');`}</style>

      <header style={{
        position: "fixed", top: 16, left: 0, right: 0,
        display: "flex", justifyContent: "center",
        padding: "0 16px", zIndex: 9999,
        pointerEvents: "none",
        fontFamily: "'Outfit', system-ui, sans-serif",
      }}>
        <nav style={{
          pointerEvents: "all",
          width: "100%", maxWidth: 880,
          display: "flex", alignItems: "center", justifyContent: "space-between",
          height: 52, padding: "0 20px",
          background: "rgba(15,15,18,0.92)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          border: "1px solid rgba(255,255,255,0.07)",
          borderRadius: 16,
          boxShadow: "0 4px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04)",
        }}>

          {/* Brand */}
          <Link href="/" style={{ display: "flex", alignItems: "center", gap: 10, textDecoration: "none" }}>
            <div style={{
              width: 28, height: 28, borderRadius: 8, background: "#10b981", flexShrink: 0,
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 0 16px rgba(16,185,129,0.5)",
            }}>
              <Leaf size={14} color="#fff" strokeWidth={2.5} />
            </div>
            <span style={{ fontSize: 15, fontWeight: 300, color: "#f4f4f5", letterSpacing: "-0.01em" }}>
              Maize<span style={{ fontWeight: 700, color: "#34d399" }}>Health</span>
            </span>
          </Link>

          {/* Right actions */}
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            {loading ? (
              <div style={{ height: 28, width: 120, background: "#27272a", borderRadius: 10, opacity: 0.5 }} />
            ) : user ? (
              <>
                <LogoutButton />
              </>
            ) : (
              <Link href="/login" style={{ textDecoration: "none" }}>
                <button style={{
                  height: 32, padding: "0 16px", borderRadius: 10, border: "none", cursor: "pointer",
                  background: "#10b981", color: "#fff", fontSize: 13, fontWeight: 600,
                  fontFamily: "inherit", boxShadow: "0 0 16px rgba(16,185,129,0.4)",
                  display: "flex", alignItems: "center",
                }}>
                  Sign in
                </button>
              </Link>
            )}
          </div>
        </nav>
      </header>
    </>
  );
}