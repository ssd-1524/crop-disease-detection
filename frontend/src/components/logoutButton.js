"use client";

import { useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { toast } from "sonner";
import { LogOut } from "lucide-react";

export default function LogoutButton() {
  const { signOut } = useAuth();
  const router = useRouter();

  const handleLogout = async () => {
    await signOut();
    toast.success("Signed out.");
    router.push("/");
  };

  return (
    <button
      onClick={handleLogout}
      style={{
        height: 32, padding: "0 12px", borderRadius: 10, border: "none", cursor: "pointer",
        background: "transparent", color: "#71717a",
        fontSize: 13, fontWeight: 500,
        fontFamily: "'Outfit', system-ui, sans-serif",
        display: "flex", alignItems: "center", gap: 6,
      }}
      onMouseEnter={e => { e.currentTarget.style.background = "#1c0a0a"; e.currentTarget.style.color = "#f87171"; }}
      onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#71717a"; }}
    >
      <LogOut size={13} />
      Sign out
    </button>
  );
}