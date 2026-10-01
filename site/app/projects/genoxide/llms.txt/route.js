import { LLMS_TXT } from "@/lib/projects/genoxide/llms";

/**
 * /projects/genoxide/llms.txt: a map of genoxide for AI assistants, in the
 * llms.txt format (https://llmstxt.org): a summary, then links to the guide,
 * the API references, the examples and the benchmarks.
 */
export function GET() {
  return new Response(LLMS_TXT, {
    headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "public, max-age=3600" },
  });
}
