"use client";

import { useRouter } from "next/navigation";
// 1. Import the correct client creator function
import { createClient } from "@/lib/supabase/client";
import { Button } from "./ui/button";

export default function LogoutButton() {
  const router = useRouter();
  // 2. Create an instance of the new Supabase client
  const supabase = createClient();

  const handleLogout = async () => {
    // This will correctly clear the session cookie
    await supabase.auth.signOut();

    // Redirect to the home page after logout
    router.push("/");

    // This is crucial: it forces the server to re-render the
    // Navbar, which will now see that the user is logged out.
    router.refresh();
  };

  return (
    <Button variant="destructive" onClick={handleLogout}>
      Logout
    </Button>
  );
}
