import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LocalMind | The Reading Desk",
  description: "A private, document-first research workspace. Read, ask, and trace every answer to its source.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}