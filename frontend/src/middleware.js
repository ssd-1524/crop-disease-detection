import { createClient } from "@/lib/supabase/middleware";
import { NextResponse } from "next/server";

export async function middleware(request) {
  const { supabase, response } = createClient(request);

  // getUser() refreshes the session cookie AND validates the token server-side.
  // Calling getSession() + getUser() back-to-back acquires the lock twice,
  // which contributes to the "Lock broken by another request with the 'steal'
  // option" AbortError in the browser. One call is sufficient.
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user && request.nextUrl.pathname.startsWith("/dashboard")) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  if (user && request.nextUrl.pathname.startsWith("/login")) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
