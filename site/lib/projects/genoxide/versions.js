import { cache } from "react";
import { fetchCached } from "@/lib/projects/fetch-cached";

/**
 * The latest published versions, from crates.io and PyPI (cached daily).
 * Either is null when its registry can't be reached; the pages then leave
 * the version out rather than show a stale hard-coded one.
 * @returns {Promise<{ rust: string | null, python: string | null }>}
 */
export const getGenoxideVersions = cache(async () => {
  const [crate, pypi] = await Promise.all([
    // crates.io asks API clients for a User-Agent that identifies them (set in fetchCached).
    fetchCached("https://crates.io/api/v1/crates/genoxide", {
      as: "json",
      headers: { Accept: "application/json" },
      tags: ["genoxide"],
    }),
    fetchCached("https://pypi.org/pypi/genoxide/json", { as: "json", tags: ["genoxide"] }),
  ]);
  return {
    rust: crate?.crate?.max_stable_version || crate?.crate?.newest_version || null,
    python: pypi?.info?.version || null,
  };
});

