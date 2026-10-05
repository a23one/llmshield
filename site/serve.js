// Local static server for the landing page. Run with `bun run dev`.
import { resolve, sep } from "node:path";

const ROOT = resolve(import.meta.dir, "public");

export async function handle(request) {
  const { pathname } = new URL(request.url);
  const relative = decodeURIComponent(pathname).replace(/\/$/, "/index.html");
  const path = resolve(ROOT, "." + relative);
  if (path !== ROOT && !path.startsWith(ROOT + sep)) {
    return new Response("Not found", { status: 404 });
  }
  const file = Bun.file(path);
  if (!(await file.exists())) {
    return new Response("Not found", { status: 404 });
  }
  return new Response(file, { headers: { "Cache-Control": "no-store" } });
}

if (import.meta.main) {
  const server = Bun.serve({
    hostname: process.env.HOST ?? "localhost",
    port: Number(process.env.PORT ?? 3000),
    fetch: handle,
  });
  console.log(`LLMShield site running at ${server.url}`);
}
