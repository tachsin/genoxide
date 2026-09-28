import { getBenchmarkRun } from "@/lib/projects/genoxide/benchmarks";

/**
 * One library's method in one benchmark scenario, as JSON, for the benchmark
 * page's details panel: `?scenario=rastrigin-30-matched&library=genoxide&solver=de`.
 * Its runs, their output and its adapter's code, highlighted on the server,
 * from the run details at the pinned commit (getBenchmarkRun). They change
 * only with a new pin, so browsers may keep an answer for an hour.
 */
export async function GET(request) {
  const params = new URL(request.url).searchParams;
  const result = await getBenchmarkRun(params.get("scenario"), params.get("library"), params.get("solver"));
  if (result.error) return Response.json({ error: result.error }, { status: result.status });
  return Response.json(result, { headers: { "Cache-Control": "public, max-age=3600" } });
}
