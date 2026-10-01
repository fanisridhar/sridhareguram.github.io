// Cloudflare Worker: receives the portfolio contact form and emails it via Resend.
// Bindings (see wrangler.toml and README.md): RESEND_API_KEY (secret), TO_EMAIL, FROM_EMAIL, ALLOWED_ORIGIN.

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MAX_BODY_BYTES = 10_000;

function corsHeaders(origin, env) {
  const allowed = (env.ALLOWED_ORIGIN || "").split(",").map((s) => s.trim()).filter(Boolean);
  const ok = allowed.includes(origin);
  return {
    ok,
    headers: {
      "Access-Control-Allow-Origin": ok ? origin : allowed[0] || "null",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Max-Age": "86400",
      Vary: "Origin",
    },
  };
}

function json(body, status, headers) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...headers },
  });
}

const escapeHtml = (s) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const cors = corsHeaders(origin, env);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: cors.ok ? 204 : 403, headers: cors.headers });
    }
    if (request.method !== "POST") return json({ error: "Method not allowed" }, 405, cors.headers);
    if (!cors.ok) return json({ error: "Origin not allowed" }, 403, cors.headers);

    const raw = await request.text();
    if (raw.length > MAX_BODY_BYTES) return json({ error: "Message too large" }, 413, cors.headers);

    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      return json({ error: "Invalid JSON" }, 400, cors.headers);
    }

    // Honeypot: bots fill the hidden field. Pretend success so they do not retry.
    if (typeof data.website === "string" && data.website !== "") return json({ ok: true }, 200, cors.headers);

    const name = typeof data.name === "string" ? data.name.trim() : "";
    const email = typeof data.email === "string" ? data.email.trim() : "";
    const message = typeof data.message === "string" ? data.message.trim() : "";

    if (name.length < 2 || name.length > 120) return json({ error: "Invalid name" }, 400, cors.headers);
    if (!EMAIL_RE.test(email) || email.length > 200) return json({ error: "Invalid email" }, 400, cors.headers);
    if (message.length < 10 || message.length > 4000) return json({ error: "Invalid message" }, 400, cors.headers);

    if (!env.RESEND_API_KEY || !env.TO_EMAIL || !env.FROM_EMAIL) {
      return json({ error: "Server is not configured" }, 500, cors.headers);
    }

    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: { Authorization: `Bearer ${env.RESEND_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        from: env.FROM_EMAIL,
        to: [env.TO_EMAIL],
        reply_to: email,
        // Header values must not contain line breaks.
        subject: `Portfolio message from ${name.replace(/[\r\n]+/g, " ")}`,
        text: `${message}\n\nFrom: ${name} <${email}>`,
        html: `<p>${escapeHtml(message).replace(/\n/g, "<br>")}</p><hr><p>From: ${escapeHtml(name)} &lt;${escapeHtml(email)}&gt;</p>`,
      }),
    });

    if (!res.ok) return json({ error: "Could not send message" }, 502, cors.headers);
    return json({ ok: true }, 200, cors.headers);
  },
};
