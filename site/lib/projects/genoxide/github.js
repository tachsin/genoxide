import { cache } from "react";
import { fetchCached } from "@/lib/projects/fetch-cached";
import { GENOXIDE_BRANCH, GENOXIDE_REPO } from "./meta";

/**
 * Reads the public genoxide repository on GitHub, server-side.
 *
 * One GitHub API call lists every file (the git tree of the branch); the
 * files themselves come from raw.githubusercontent.com, which has no API
 * rate limit. Both go through Next's data cache (daily), so GitHub's 60
 * unauthenticated API calls an hour are never close to being used.
 */

const TREE_URL = `https://api.github.com/repos/${GENOXIDE_REPO}/git/trees/${GENOXIDE_BRANCH}?recursive=1`;
export const RAW_BASE = `https://raw.githubusercontent.com/${GENOXIDE_REPO}/${GENOXIDE_BRANCH}/`;
export const BLOB_BASE = `https://github.com/${GENOXIDE_REPO}/blob/${GENOXIDE_BRANCH}/`;
export const TREE_BASE = `https://github.com/${GENOXIDE_REPO}/tree/${GENOXIDE_BRANCH}/`;

/**
 * Every file path of the branch, or null when GitHub can't be reached.
 * Deduplicated per request with React's cache().
 * @returns {Promise<Set<string> | null>}
 */
export const getRepoFiles = cache(async () => {
  const tree = await fetchCached(TREE_URL, {
    as: "json",
    headers: { Accept: "application/vnd.github+json" },
    tags: ["genoxide"],
  });
  if (!tree || !Array.isArray(tree.tree)) return null;
  return new Set(tree.tree.filter((e) => e?.type === "blob").map((e) => e.path));
});

/**
 * A file of the branch as text, or null.
 * @param {string} path  repository-relative, e.g. "examples/knapsack/README.md"
 */
export const getRepoFile = cache(async (path) =>
  fetchCached(`${RAW_BASE}${path.split("/").map(encodeURIComponent).join("/")}`, {
    tags: ["genoxide"],
  }),
);
