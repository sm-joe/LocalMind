import { createHash, timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";
import { assertSameOrigin, issueSession, SESSION_AGE_SECONDS, SESSION_COOKIE } from "@/lib/session";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  if (!assertSameOrigin(request)) {
    return NextResponse.json({ detail: "Invalid request origin." }, { status: 403 });
  }
  const expected = process.env.LOCALRAG_ADMIN_PASSWORD;
  if (!expected || expected.length < 12) {
    return NextResponse.json({ detail: "Web authentication is not configured." }, { status: 503 });
  }
  let password: string;
  try {
    const body = await request.json();
    password = typeof body.password === "string" ? body.password : "";
  } catch {
    return NextResponse.json({ detail: "Invalid request body." }, { status: 400 });
  }
  if (password.length > 1024) {
    return NextResponse.json({ detail: "Invalid credentials." }, { status: 401 });
  }
  const left = createHash("sha256").update(password).digest();
  const right = createHash("sha256").update(expected).digest();
  if (!timingSafeEqual(left, right)) {
    return NextResponse.json({ detail: "Invalid credentials." }, { status: 401 });
  }
  const response = NextResponse.json({ authenticated: true });
  response.cookies.set(SESSION_COOKIE, issueSession(), {
    httpOnly: true, sameSite: "strict", secure: process.env.NODE_ENV === "production" && request.nextUrl.protocol === "https:",
    path: "/", maxAge: SESSION_AGE_SECONDS,
  });
  response.headers.set("Cache-Control", "no-store");
  return response;
}
