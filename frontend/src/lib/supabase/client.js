// src/lib/supabase/client.js
import { createBrowserClient } from "@supabase/ssr";

// Singleton instance — ensures only ONE client exists in the browser,
// preventing IndexedDB auth-lock contention (AbortError: "Lock broken by
// another request with the 'steal' option").
let client;

export function createClient() {
  if (!client) {
    client = createBrowserClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
    );
  }
  return client;
}
