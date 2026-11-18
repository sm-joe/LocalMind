import { NextRequest, NextResponse } from "next/server";
import { assertSameOrigin, hasSession } from "@/lib/session";

export const runtime = "nodejs";
const ALLOWED = new Set(["GET", "POST", "PATCH", "DELETE"]);

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  if (!(await hasSession())) return NextResponse.json({ detail: "Authentication required." }, { status: 401 });
  if (!ALLOWED.has(request.method)) return NextResponse.json({ detail: "Method not allowed." }, { status: 405 });
  if (request.method !== "GET" && !assertSameOrigin(request)) {
    return NextResponse.json({ detail: "Invalid request origin." }, { status: 403 });
  }
  const key = process.env.LOCALRAG_API_KEY;
  const backend = process.env.LOCALRAG_API_INTERNAL_URL;
  if (!key || !backend) return NextResponse.json({ detail: "API gateway not configured." }, { status: 503 });
  const { path } = await context.params;
  if (!path?.length || path.some((segment) => !segment || segment === "." || segment === "..")) {
    return NextResponse.json({ detail: "Invalid API path." }, { status: 400 });
  }
  const pathname = "/" + path.map(encodeURIComponent).join("/");
  const target = new URL(pathname, backend);
  target.search = request.nextUrl.search;
  const headers = new Headers();
  headers.set("X-API-Key", key);
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("Content-Type", contentType);
  const idempotencyKey = request.headers.get("idempotency-key");
  if (idempotencyKey) headers.set("Idempotency-Key", idempotencyKey);
  const contentLength = Number(request.headers.get("content-length") || 0);
  if (contentLength > 12 * 1024 * 1024) {
    return NextResponse.json({ detail: "Request body too large." }, { status: 413 });
  }
  try {
    const body = request.method === "GET" ? undefined : await request.arrayBuffer();
    if (body && body.byteLength > 12 * 1024 * 1024) {
      return NextResponse.json({ detail: "Request body too large." }, { status: 413 });
    }
    // A local CPU-run LLM may need several minutes for its first response.
    // Uploads and ordinary API calls retain the shorter network timeout.
    const timeoutMs = path.length === 1 && path[0] === "chat" && request.method === "POST"
      ? 300000
      : 120000;
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
    const result = new NextResponse(upstream.body, { status: upstream.status });
    const responseType = upstream.headers.get("content-type");
    if (responseType) result.headers.set("Content-Type", responseType);
    result.headers.set("Cache-Control", "no-store");
    return result;
  } catch (error) {
    // Do not tell the user that the API is down when a slow first model
    // inference exceeded the gateway timeout. No internal URLs or API keys
    // are included in the response.
    const timedOut = error instanceof Error && (
      error.name === "TimeoutError" || error.name === "AbortError"
    );
    console.error("LocalRAG API proxy request failed", {
      operation: path[0],
      errorType: error instanceof Error ? error.name : "unknown",
      timedOut,
    });
    return NextResponse.json({
      detail: timedOut
        ? "The local model took too long to respond. Check Ollama memory and CPU usage, then retry."
        : "Cannot reach the LocalRAG API. Check that the api container is running and healthy.",
    }, { status: timedOut ? 504 : 502 });
  }
}

export const GET = proxy;
export const POST = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
