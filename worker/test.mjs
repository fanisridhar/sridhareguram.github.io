// Offline test of the Worker's validation and mail call. Run: node worker/test.mjs
import assert from "node:assert/strict";
import worker from "./contact-worker.js";

const env = { RESEND_API_KEY: "k", TO_EMAIL: "me@example.com", FROM_EMAIL: "P <p@example.com>", ALLOWED_ORIGIN: "https://ok.example" };
const sent = [];
globalThis.fetch = async (url, init) => {
  sent.push({ url, body: JSON.parse(init.body) });
  return new Response("{}", { status: 200 });
};

const call = (body, { origin = "https://ok.example", method = "POST" } = {}) =>
  worker.fetch(new Request("https://w.example/", { method, headers: { Origin: origin, "Content-Type": "application/json" }, body: method === "POST" ? (typeof body === "string" ? body : JSON.stringify(body)) : undefined }), env);

const good = { name: "Ada", email: "ada@example.com", message: "Hello there, nice portfolio.", website: "" };

let r = await call(good);
assert.equal(r.status, 200);
assert.equal(sent.length, 1);
assert.equal(sent[0].body.reply_to, "ada@example.com");

r = await call(good, { origin: "https://evil.example" });
assert.equal(r.status, 403);

r = await call(good, { method: "OPTIONS" });
assert.equal(r.status, 204);
assert.equal(r.headers.get("Access-Control-Allow-Origin"), "https://ok.example");

r = await call({ ...good, website: "spam" });
assert.equal(r.status, 200);
assert.equal(sent.length, 1, "honeypot must not send mail");

for (const bad of [{ ...good, name: "" }, { ...good, email: "nope" }, { ...good, message: "short" }, { ...good, message: "x".repeat(4001) }]) {
  r = await call(bad);
  assert.equal(r.status, 400);
}

r = await call("{not json");
assert.equal(r.status, 400);

r = await call({ ...good, name: "Eve\r\nBcc: x@y.z" });
assert.equal(sent.at(-1).body.subject.includes("\n"), false, "no header injection");

r = await call({ ...good, message: "<script>alert(1)</script> hello" });
assert.ok(!sent.at(-1).body.html.includes("<script>"), "html is escaped");

console.log("worker tests passed");
