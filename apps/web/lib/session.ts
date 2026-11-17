import "server-only";
import { createHmac, timingSafeEqual } from "node:crypto";
import { cookies } from "next/headers";

export const SESSION_COOKIE = "localrag_session";
export const SESSION_AGE_SECONDS = 12 * 60 * 60;

function sessionSecret(): string {
  const secret = process.env.LOCALRAG_SESSION_SECRET;
  if (!secret || secret.length < 32) {
    throw new Error("LOCALRAG_SESSION_SECRET must contain at least 32 characters");
  }
  return secret;
}

export function issueSession(): string {
  const expires = Math.floor(Date.now() / 1000) + SESSION_AGE_SECONDS;
  const payload = `v1.${expires}`;
  const signature = createHmac("sha256", sessionSecret()).update(payload).digest("hex");
  return `${payload}.${signature}`;
}

export function verifySession(token: string | undefined): boolean {
  if (!token || token.length > 256) return false;
  const parts = token.split(".");
  if (parts.length !== 3 || parts[0] !== "v1") return false;
  const expires = Number(parts[1]);
  if (!Number.isSafeInteger(expires) || expires <= Date.now() / 1000) return false;
  const actual = parts[2];
  if (!/^[a-f0-9]{64}$/.test(actual)) return false;
  const expected = createHmac("sha256", sessionSecret())
    .update(`v1.${parts[1]}`).digest("hex");
  return timingSafeEqual(Buffer.from(actual, "hex"), Buffer.from(expected, "hex"));
}

export async function hasSession(): Promise<boolean> {
  const cookie = (await cookies()).get(SESSION_COOKIE)?.value;
  return verifySession(cookie);
}

export function assertSameOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  // Browsers send Origin for mutations; reject cross-site and missing Origin.
  if (!origin) return false;
  try {
    return new URL(origin).origin === new URL(request.url).origin;
  } catch {
    return false;
  }
}
