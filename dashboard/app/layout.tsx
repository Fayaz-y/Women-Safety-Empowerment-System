import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import { Toaster } from "react-hot-toast";

export const metadata: Metadata = {
  title: "Women Safety AI Monitor",
  description: "Real-time AI monitoring dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div style={{ display: "flex", height: "100vh" }}>
          <Sidebar />
          <main
            style={{
              flex: 1,
              overflow: "hidden",
              display: "flex",
              flexDirection: "column",
            }}
          >
            {children}
          </main>
        </div>
        <Toaster position="bottom-right" />
      </body>
    </html>
  );
}
