"use client"; // <-- Now a client component

import Link from "next/link";
import { useAuth } from "@/context/AuthContext"; // <-- Use our new hook
import { Button } from "@/components/ui/button";
import LogoutButton from "./logoutButton";

export default function Navbar() {
  const { user, loading } = useAuth(); // <-- Get user state instantly

  return (
    <nav className="sticky top-4 inset-x-0 max-w-4xl mx-auto z-50 bg-white/60 backdrop-blur-lg rounded-full border border-gray-200/50 shadow-lg">
      <div className="flex items-center justify-between h-16 px-6">
        <Link href="/" className="text-2xl font-bold text-gray-800">
          Maize<span className="text-green-600">Health</span>
        </Link>
        <div className="flex items-center gap-4">
          {loading ? (
            <div className="h-8 w-24 bg-gray-200 rounded-full animate-pulse"></div>
          ) : user ? (
            <>
              <Link href="/dashboard">
                <Button variant="ghost">Upload Image</Button>
              </Link>
              <LogoutButton />
            </>
          ) : (
            <Link href="/login">
              <Button>Login / Sign Up</Button>
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
}
