import { NextRequest, NextResponse } from "next/server";
import { assertSameOrigin, SESSION_COOKIE } from "@/lib/session";

export const runtime = "nodejs";
export async function POST(request: NextRequest) {
  if (!assertSameOrigin(request)) return NextResponse.json({ detail: "Invalid request origin." }, { status: 403 });
  const response = NextResponse.json({ authenticated: false });
  response.cookies.delete(SESSION_COOKIE);
  response.headers.set("Cache-Control", "no-store");
  return response;
}
