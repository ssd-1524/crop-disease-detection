"use client";

import { createContext, useContext, useState, useEffect, useMemo } from "react";
import { createClient } from "@/lib/supabase/client";

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // createClient() returns the singleton, but wrapping in useMemo ensures the
  // reference is stable across re-renders so the useEffect below doesn't fire
  // repeatedly, which would hammer the IndexedDB auth lock.
  const supabase = useMemo(() => createClient(), []);

  useEffect(() => {
    // onAuthStateChange immediately fires an 'INITIAL_SESSION' event upon registration.
    // Calling getSession() manually at the same time creates a race condition for the
    // browser Web Lock, causing the "Lock broken by another request with 'steal' option".
    const { data: authListener } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setUser(session?.user ?? null);
        setLoading(false);
      }
    );

    return () => {
      authListener.subscription.unsubscribe();
    };
  }, [supabase]);

  const value = {
    user,
    signOut: () => supabase.auth.signOut(),
    loading,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
