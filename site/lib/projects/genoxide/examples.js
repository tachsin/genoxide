import { cache } from "react";
import { parse as parseYaml } from "yaml";
import { BLOB_BASE, RAW_BASE, TREE_BASE, getRepoFile, getRepoFiles } from "./github";
import { GENOXIDE_PATH } from "./meta";

/**
 * genoxide's examples, read from the repository at request time (cached
 * daily, see lib/projects/fetch-cached.js).
 *
 * Layout in the repository, one folder per example:
 *
 *   examples/<name>/README.md   YAML front matter + the example's text
 *   examples/<name>/main.rs     the Rust example (cargo run --example <name>)
 *   examples/<name>/main.py     the Python example, when there is one
 *   examples/<name>/output.txt  what the example prints (a seeded run), optional
 *   examples/<name>/trace.json  its run recorded for the page's player, optional
 *
 * A folder with its own Cargo.toml is a crate of its own (examples/gpu):
 * its Rust code is src/main.rs and it runs with --manifest-path.
 *
 * Front matter fields: title, category, summary, reference, reference_url,
 * optimum, languages, order. All optional: a missing one falls back to
 * something derived from the folder, so a half-written README still shows.
 */

const EXAMPLES_DIR = "examples";
const DEFAULT_ORDER = 1000;
const FRONT_MATTER = /^---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)/;

export const EXAMPLES_PATH = `${GENOXIDE_PATH}/examples`;

/** @typedef {"rust" | "python"} CodeLanguage */

/**
 * @typedef {object} ExampleSummary
 * @property {string} slug          URL segment, e.g. "n-queens"
 * @property {string} dir           folder name, e.g. "n_queens"
 * @property {string} title
 * @property {string} category
 * @property {string} summary
 * @property {string | null} reference
 * @property {string | null} referenceUrl
 * @property {string | null} optimum
 * @property {CodeLanguage[]} languages  those with a file in the folder, Rust first
 * @property {number} order
 * @property {boolean} isCrate      has its own Cargo.toml
 * @property {{ rust: string | null, python: string | null, output: string | null, trace: string | null }} files
 *   repository paths; `output` is output.txt, `trace` is trace.json
 */

/**
 * Split YAML front matter from a Markdown file.
 * @param {string} source
 * @returns {{ data: Record<string, any>, body: string }}
 */
export function parseFrontMatter(source) {
  const text = String(source ?? "").replace(/^﻿/, "");
  const match = FRONT_MATTER.exec(text);
  if (!match) return { data: {}, body: text };
  let data = {};
  try {
    const parsed = parseYaml(match[1]);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) data = parsed;
  } catch {
    // Malformed YAML: keep the body, drop the fields.
  }
  return { data, body: text.slice(match[0].length) };
}

export function slugForDir(dir) {
  return dir
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function humanize(dir) {
  const words = dir.replace(/[_-]+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function text(value) {
  if (value === null || value === undefined) return null;
  const s = String(value).trim();
  return s ? s : null;
}

function parseLanguages(value) {
  const list = Array.isArray(value) ? value : typeof value === "string" ? value.split(/[\s,]+/) : [];
  return list.map((l) => String(l).trim().toLowerCase()).filter(Boolean);
}

/**
 * @param {string} dir
 * @param {Record<string, any>} data  front matter
 * @param {Set<string>} files         every file path of the repository
 * @returns {ExampleSummary}
 */
function toSummary(dir, data, files) {
  const base = `${EXAMPLES_DIR}/${dir}`;
  const isCrate = files.has(`${base}/Cargo.toml`);
  const rust = [`${base}/main.rs`, `${base}/src/main.rs`].find((p) => files.has(p)) ?? null;
  const python = files.has(`${base}/main.py`) ? `${base}/main.py` : null;
  const output = files.has(`${base}/output.txt`) ? `${base}/output.txt` : null;
  const trace = files.has(`${base}/trace.json`) ? `${base}/trace.json` : null;

  // A language shows only when its file exists; the front matter can narrow
  // the list, never add a language without code.
  const available = /** @type {CodeLanguage[]} */ (
    [rust && "rust", python && "python"].filter(Boolean)
  );
  const declared = parseLanguages(data.languages);
  const languages = declared.length ? available.filter((l) => declared.includes(l)) : available;

  const order = Number(data.order);
  return {
    slug: slugForDir(dir),
    dir,
    title: text(data.title) ?? humanize(dir),
    category: text(data.category) ?? "Other",
    summary: text(data.summary) ?? "",
    reference: text(data.reference),
    referenceUrl: text(data.reference_url),
    optimum: text(data.optimum),
    languages: languages.length ? languages : available,
    order: Number.isFinite(order) ? order : DEFAULT_ORDER,
    isCrate,
    files: { rust, python, output, trace },
  };
}

function exampleDirs(files) {
  const dirs = new Set();
  for (const path of files) {
    const m = /^examples\/([^/]+)\/README\.md$/.exec(path);
    if (m) dirs.add(m[1]);
  }
  return [...dirs];
}

/**
 * Every example, sorted by `order` then title.
 * `ok` is false when GitHub couldn't be reached: pages then show a link to
 * the repository instead of the list.
 * @returns {Promise<{ ok: boolean, examples: ExampleSummary[] }>}
 */
export const getExamples = cache(async () => {
  const files = await getRepoFiles();
  if (!files) return { ok: false, examples: [] };

  const dirs = exampleDirs(files);
  const examples = await Promise.all(
    dirs.map(async (dir) => {
      const readme = await getRepoFile(`${EXAMPLES_DIR}/${dir}/README.md`);
      const { data } = parseFrontMatter(readme ?? "");
      return toSummary(dir, data, files);
    }),
  );

  examples.sort((a, b) => a.order - b.order || a.title.localeCompare(b.title));
  return { ok: true, examples };
});

/** Categories in first-appearance order (the examples are already sorted). */
export function exampleCategories(examples) {
  return [...new Set(examples.map((e) => e.category))];
}

/**
 * Rewrites the README's relative links: another example's folder or README
 * becomes that example's page here, other repository paths go to GitHub
 * (images to raw.githubusercontent.com so they load).
 */
function makeUrlResolver(dir, slugByDir) {
  const folder = `https://example.invalid/${EXAMPLES_DIR}/${dir}/`;
  return (href, kind) => {
    if (!href || href.startsWith("#") || /^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith("//")) {
      return href;
    }
    let url;
    try {
      url = new URL(href, folder);
    } catch {
      return href;
    }
    const path = url.pathname.replace(/^\//, "");
    const other = /^examples\/([^/]+)\/?(README\.md)?$/i.exec(path);
    if (kind === "link" && other && slugByDir.has(other[1])) {
      return `${EXAMPLES_PATH}/${slugByDir.get(other[1])}${url.hash}`;
    }
    if (kind === "image") return `${RAW_BASE}${path}`;
    const isDir = path === "" || path.endsWith("/");
    return `${isDir ? TREE_BASE : BLOB_BASE}${path}${url.hash}`;
  };
}

/** Drops a leading "# Title" line: the page renders the title itself. */
function stripLeadingTitle(body) {
  return body.replace(/^\s*#\s+[^\n]*\n+/, "");
}

/**
 * One example with its README body and code, or null when there's no such
 * example (or GitHub is unreachable: then `getExamples().ok` is false too).
 * @param {string} slug
 */
export const getExample = cache(async (slug) => {
  const { ok, examples } = await getExamples();
  if (!ok) return null;
  const index = examples.findIndex((e) => e.slug === slug);
  if (index === -1) return null;
  const example = examples[index];

  const [readme, rust, python, output] = await Promise.all([
    getRepoFile(`${EXAMPLES_DIR}/${example.dir}/README.md`),
    example.files.rust && example.languages.includes("rust") ? getRepoFile(example.files.rust) : null,
    example.files.python && example.languages.includes("python") ? getRepoFile(example.files.python) : null,
    example.files.output ? getRepoFile(example.files.output) : null,
  ]);

  const slugByDir = new Map(examples.map((e) => [e.dir, e.slug]));
  return {
    ...example,
    body: stripLeadingTitle(parseFrontMatter(readme ?? "").body),
    resolveUrl: makeUrlResolver(example.dir, slugByDir),
    code: { rust, python },
    output: output?.trim() ? output.replace(/\s+$/, "") : null,
    // The browser fetches the trace itself, when the player scrolls into
    // view (raw.githubusercontent.com allows it: Access-Control-Allow-Origin *).
    traceUrl: example.files.trace ? rawUrl(example.files.trace) : null,
    folderUrl: `${TREE_BASE}${EXAMPLES_DIR}/${example.dir}`,
    previous: examples[index - 1] ?? null,
    next: examples[index + 1] ?? null,
  };
});

/** Repository URL of a file, for "view on GitHub" links. */
export function blobUrl(path) {
  return `${BLOB_BASE}${path}`;
}

/** Raw URL of a file of the branch the examples are read from, for the browser to fetch. */
export function rawUrl(path) {
  return `${RAW_BASE}${path.split("/").map(encodeURIComponent).join("/")}`;
}

/**
 * How to run an example from a clone of the repository.
 * @param {ExampleSummary} example
 * @param {CodeLanguage} language
 */
export function runCommand(example, language) {
  if (language === "python") return `pip install genoxide\npython ${example.files.python}`;
  if (example.isCrate) {
    return `cargo run --release --manifest-path ${EXAMPLES_DIR}/${example.dir}/Cargo.toml`;
  }
  return `cargo run --release --example ${example.dir}`;
}

/**
 * The command whose output is the example's output.txt: the run command
 * without the install step.
 * @param {ExampleSummary} example
 * @param {CodeLanguage} language
 */
export function outputCommand(example, language) {
  return runCommand(example, language).split(/\n/).at(-1);
}

/** Paths of every example page, for the sitemap. Empty when GitHub is unreachable. */
export async function genoxideExamplePaths() {
  const { examples } = await getExamples();
  return examples.map((e) => `${EXAMPLES_PATH}/${e.slug}`);
}
