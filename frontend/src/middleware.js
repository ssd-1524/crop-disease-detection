import { createClient } from "@/lib/supabase/middleware";
import { NextResponse } from "next/server";

export async function middleware(request) {
  const { supabase, response } = createClient(request);

  // This is the most important part: it refreshes the session cookie
  await supabase.auth.getSession();

  // Now, get the user and perform route protection
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
