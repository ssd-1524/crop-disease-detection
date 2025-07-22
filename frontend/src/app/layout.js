import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar"; // <-- Import the Navbar

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Maize Disease Detection",
  description: "Detect diseases in maize leaves using AI.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Navbar /> {/* <-- Add Navbar here */}
        <main className="flex-grow">{children}</main>
      </body>
    </html>
  );
}
