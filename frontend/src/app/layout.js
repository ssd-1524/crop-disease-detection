// src/app/layout.js

import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/navbar";
import { Toaster } from "@/components/ui/sonner";
import { AuthProvider } from "@/context/AuthContext";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Maize Disease Detection",
  description: "Detect diseases in maize leaves using AI.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      {/* Add suppressHydrationWarning to the body tag */}
      <body
        className={`${inter.className} animated-gradient`}
        suppressHydrationWarning={true}
      >
        <AuthProvider>
          <Navbar />
          <main>{children}</main>
          <Toaster richColors />
        </AuthProvider>
      </body>
    </html>
  );
}
