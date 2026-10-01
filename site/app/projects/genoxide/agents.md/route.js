import { getRepoFile } from "@/lib/projects/genoxide/github";

/**
 * /projects/genoxide/agents.md: AGENTS.md, the complete guide in one page, as
 * plain text at the pinned commit, so an assistant can read it in one request.
 */
export async function GET() {
  const text = await getRepoFile("AGENTS.md");
  if (text == null) return new Response("AGENTS.md is not available\n", { status: 502 });
  return new Response(text, {
    headers: { "Content-Type": "text/markdown; charset=utf-8", "Cache-Control": "public, max-age=3600" },
  });
}
