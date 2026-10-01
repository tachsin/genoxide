import { cache } from "react";
import { parse as parseYaml } from "yaml";
import { BLOB_BASE, RAW_BASE, TREE_BASE, getRepoFile, getRepoFiles } from "./github";
import { GENOXIDE_PATH } from "./meta";

/**
 * genoxide's examples, read from the repository at request time, at the
 * commit the pages were synced from (cached daily, see github.js).
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
 * optimum, languages, order, and for the problems of a paper that defines
 * several (ZDT, DTLZ, WFG, CEC 2006...) family and tab: `family` names the
 * group, whose pages show its members as tabs, which the sidebar lists
 * together and the index shows as one card, and `tab` is the example's short
 * label there (its title when
 * absent). All optional: a missing one falls back to something derived from
 * the folder, so a half-written README still shows.
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
 * @property {string | null} family  the group of problems from one paper, e.g. "WFG"
 * @property {string | null} tab     the short label among the family, e.g. "g07"
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
    family: text(data.family),
    tab: text(data.tab),
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
 * A README that can't be fetched makes the whole list unavailable (`ok` false),
 * as a failed file listing does: a page built without it would have a made-up
 * title, category and order, and be indexed with them.
 *
 * @returns {Promise<{ ok: boolean, examples: ExampleSummary[], readmes: Map<string, string> }>}
 */
export const getExamples = cache(async () => {
  const files = await getRepoFiles();
  if (!files) return { ok: false, examples: [], readmes: new Map() };

  const dirs = exampleDirs(files);
  const readmes = await Promise.all(dirs.map((dir) => getRepoFile(`${EXAMPLES_DIR}/${dir}/README.md`)));
  if (readmes.some((readme) => readme === null || readme === undefined)) {
    return { ok: false, examples: [], readmes: new Map() };
  }
  const examples = dirs.map((dir, i) => toSummary(dir, parseFrontMatter(readmes[i]).data, files));

  examples.sort((a, b) => a.order - b.order || a.title.localeCompare(b.title));
  return { ok: true, examples, readmes: new Map(dirs.map((dir, i) => [dir, readmes[i]])) };
});

/** Categories in first-appearance order (the examples are already sorted). */
export function exampleCategories(examples) {
  return [...new Set(examples.map((e) => e.category))];
}

/**
 * @typedef {object} ExampleFamily
 * @property {string} name
 * @property {{ reference: string, referenceUrl: string | null }[]} sources
 *   the distinct references of its members, usually one
 * @property {{ slug: string, title: string, label: string }[]} members  in their order
 */

/**
 * The family `name` of the (sorted) examples, or null when it has none.
 * @param {ExampleSummary[]} examples
 * @param {string | null} name
 * @returns {ExampleFamily | null}
 */
export function exampleFamily(examples, name) {
  if (!name) return null;
  const members = examples.filter((e) => e.family === name);
  if (!members.length) return null;
  const sources = [];
  for (const e of members) {
    if (e.reference && !sources.some((s) => s.reference === e.reference)) {
      sources.push({ reference: e.reference, referenceUrl: e.referenceUrl });
    }
  }
  return {
    name,
    sources,
    members: members.map((e) => ({ slug: e.slug, title: e.title, label: e.tab ?? e.title })),
  };
}

/**
 * @typedef {{ kind: "example", slug: string, title: string }
 *   | { kind: "family", name: string, members: { slug: string, title: string, label: string }[] }} ExampleTreeEntry
 */

/**
 * Every example for the sidebar: by category, in the order of exampleCategories,
 * and a family as one entry where its first member is (in that member's
 * category), with its members inside.
 * @param {ExampleSummary[]} examples  sorted
 * @returns {{ category: string, entries: ExampleTreeEntry[] }[]}
 */
export function exampleTree(examples) {
  const groups = new Map(exampleCategories(examples).map((c) => [c, []]));
  const families = new Map();
  for (const e of examples) {
    if (!e.family) {
      groups.get(e.category).push({ kind: "example", slug: e.slug, title: e.title });
      continue;
    }
    let family = families.get(e.family);
    if (!family) {
      family = { kind: "family", name: e.family, members: [] };
      families.set(e.family, family);
      groups.get(e.category).push(family);
    }
    family.members.push({ slug: e.slug, title: e.title, label: e.tab ?? e.title });
  }
  return [...groups]
    .map(([category, entries]) => ({ category, entries }))
    .filter((group) => group.entries.length);
}

/**
 * What a family's problems test, for its card in the examples grid: one sentence, general enough
 * to stay true as problems are added. A family missing here shows its paper's title instead.
 * @type {Record<string, string>}
 */
const FAMILY_SUMMARIES = {
  "CEC 2006":
    "The constrained problems of the CEC 2006 competition: linear and nonlinear inequalities and equalities, some with a feasible region that is tiny or in pieces.",
  CTP: "Two objectives, two variables and one or two constraints from a tunable generator: fronts cut into pieces or points, behind infeasible bands or at the ends of narrow tunnels.",
  "C-DTLZ": "DTLZ problems with constraints, here with three objectives: an infeasible barrier before the front, holes in it, or a new front on the constraints' boundaries.",
  "DAS-CMOP": "Two or three objectives with constraints whose difficulty a triplet sets: fronts in pieces, a narrow band of feasible distances, and infeasible regions in the way.",
  "DC-DTLZ": "DTLZ problems with constraints on the variables, here with three objectives: the front in cones or patches, and local optima of the constraint violation on the way.",
  DTLZ: "Scalable to any number of objectives, here three: fronts that are a plane, part of a sphere, a curve or disconnected regions, some behind many local fronts, and convex, scaled and inverted versions.",
  MW: "Two or three objectives over 15 variables with one to four constraints: fronts whole, in pieces, in points or on the constraints' boundaries, some reached only through narrow feasible regions.",
  WFG: "Two objectives from the WFG toolkit: convex, concave, linear, mixed and disconnected fronts behind biased, deceptive, multimodal and non-separable parameters.",
  ZDT: "Two conflicting objectives, with convex, concave, disconnected, multimodal, deceptive and unevenly crowded Pareto fronts.",
  Hartmann: "Hartmann's function, four Gaussian wells in the unit cube, in 3 and 6 dimensions.",
  Shekel: "Shekel's function in 4 dimensions, with 5, 7 or 10 narrow wells.",
  Schwefel: "Two of the problems of Schwefel's book: a rotated ellipsoid and a deceptive function.",
  Schaffer: "One variable and two objectives, from the paper of the first multi-objective genetic algorithm, VEGA.",
  Viennet: "Three objectives of two variables, with curved and split Pareto fronts.",
  "N-Queens":
    "N queens on an N×N chessboard with no two in a row, a column or a diagonal: the usual 8×8 board and larger ones, solved by the same search.",
};

/**
 * The authors' surnames, the year and the title of a reference: from "Huband, S., Hingston, P.,
 * Barone, L. and While, L. (2006). A review of multiobjective test problems...", "Huband et al.",
 * "2006" and "A review of multiobjective test problems...". Null for a reference in another form.
 * @param {string} reference
 * @returns {{ authors: string, year: string, title: string | null } | null}
 */
function citation(reference) {
  const match = /^(.*?)\s*\((\d{4})[a-z]?\)\.?\s*/.exec(reference);
  if (!match) return null;
  // "Surname, I. J., Other, K. and Last, L.": the surnames are the parts that aren't initials
  const surnames = match[1]
    .replace(/\s+and\s+/g, ", ")
    .split(/,\s*/)
    .map((part) => part.trim())
    .filter((part) => part && !part.endsWith("."));
  if (!surnames.length) return null;
  const authors =
    surnames.length === 1 ? surnames[0] : surnames.length === 2 ? surnames.join(" and ") : `${surnames[0]} et al.`;
  const title = /^(.+?[.?!])(?:\s|$)/.exec(reference.slice(match[0].length))?.[1].replace(/\.$/, "") ?? null;
  return { authors, year: match[2], title };
}

/**
 * @typedef {object} ExampleCardData  one example in the grid
 * @property {string} slug
 * @property {string} title
 * @property {string} category
 * @property {string} summary
 * @property {CodeLanguage[]} languages
 */

/**
 * @typedef {object} ExampleFamilyCardData  a family in the grid
 * @property {"family"} kind
 * @property {string} name
 * @property {string} category  its first member's
 * @property {string} summary   from FAMILY_SUMMARIES, or its paper's title
 * @property {string | null} cite  its papers' authors and years, e.g. "Deb et al. (2001, 2002)"
 * @property {CodeLanguage[]} languages  those every member has
 * @property {(ExampleCardData & { label: string })[]} members  in their order
 */

/** @typedef {({ kind: "example" } & ExampleCardData) | ExampleFamilyCardData} ExampleGridEntry */

/**
 * Every example for the index's grid, in order, with only what the grid shows: a family as one
 * entry where its first member is, with its members inside, as in the sidebar's exampleTree.
 * @param {ExampleSummary[]} examples  sorted
 * @returns {ExampleGridEntry[]}
 */
export function exampleGrid(examples) {
  const card = ({ slug, title, category, summary, languages }) => ({ slug, title, category, summary, languages });
  const entries = [];
  const families = new Map();
  for (const e of examples) {
    if (!e.family) {
      entries.push({ kind: "example", ...card(e) });
      continue;
    }
    let family = families.get(e.family);
    if (!family) {
      family = { members: [], references: [] };
      families.set(e.family, family);
      entries.push({ kind: "family", name: e.family, category: e.category, family });
    }
    family.members.push({ ...card(e), label: e.tab ?? e.title });
    if (e.reference && !family.references.includes(e.reference)) family.references.push(e.reference);
  }
  return entries.map((entry) => {
    if (entry.kind === "example") return entry;
    const { members, references } = entry.family;
    const cites = references.map(citation).filter(Boolean);
    // one per list of authors, with its years: "Deb et al. (2001, 2002)"
    const years = new Map();
    for (const { authors, year } of cites) years.set(authors, [...new Set([...(years.get(authors) ?? []), year])]);
    return {
      kind: "family",
      name: entry.name,
      category: entry.category,
      summary: FAMILY_SUMMARIES[entry.name] ?? cites.find((c) => c.title)?.title ?? "",
      cite: years.size ? [...years].map(([authors, list]) => `${authors} (${list.sort().join(", ")})`).join("; ") : null,
      languages: ["rust", "python"].filter((l) => members.every((m) => m.languages.includes(l))),
      members,
    };
  });
}

/**
 * Rewrites the README's relative links: another example's folder or README
 * becomes that example's page here, other repository paths go to GitHub
 * (images to raw.githubusercontent.com so they load).
 */
// The READMEs link to their own pages here ("[The project page](https://tachsin.gr/...)"), for
// readers on GitHub: here, a link to this page goes to its player, and to another example's page
// stays on the site.
const SITE_EXAMPLES = `https://tachsin.gr${EXAMPLES_PATH}/`;

function makeUrlResolver(dir, slugByDir) {
  const folder = `https://example.invalid/${EXAMPLES_DIR}/${dir}/`;
  return (href, kind) => {
    if (kind === "link" && href?.startsWith(SITE_EXAMPLES)) {
      const slug = href.slice(SITE_EXAMPLES.length).replace(/[/#?].*$/, "");
      return slug === slugForDir(dir) ? "#example-run-heading" : `${EXAMPLES_PATH}/${slug}`;
    }
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
  const { ok, examples, readmes } = await getExamples();
  if (!ok) return null;
  const index = examples.findIndex((e) => e.slug === slug);
  if (index === -1) return null;
  const example = examples[index];

  // the README as getExamples read it: fetched once, so it can't fail here after succeeding there
  const readme = readmes.get(example.dir);
  const [rust, python, output] = await Promise.all([
    example.files.rust && example.languages.includes("rust") ? getRepoFile(example.files.rust) : null,
    example.files.python && example.languages.includes("python") ? getRepoFile(example.files.python) : null,
    example.files.output ? getRepoFile(example.files.output) : null,
  ]);

  const slugByDir = new Map(examples.map((e) => [e.dir, e.slug]));
  const front = parseFrontMatter(readme ?? "");
  return {
    ...example,
    body: stripLeadingTitle(front.body),
    // what the player plays, when it isn't the run of the code below (front matter trace_note)
    traceNote: typeof front.data.trace_note === "string" ? front.data.trace_note : null,
    resolveUrl: makeUrlResolver(example.dir, slugByDir),
    code: { rust, python },
    output: output?.trim() ? output.replace(/\s+$/, "") : null,
    // The browser fetches the trace itself, when the player scrolls into
    // view (raw.githubusercontent.com allows it: Access-Control-Allow-Origin *).
    traceUrl: example.files.trace ? rawUrl(example.files.trace) : null,
    folderUrl: `${TREE_BASE}${EXAMPLES_DIR}/${example.dir}`,
    // its paper's other problems, for the tabs at the top of its page
    familyGroup: exampleFamily(examples, example.family),
    previous: examples[index - 1] ?? null,
    next: examples[index + 1] ?? null,
  };
});

/** Repository URL of a file, for "view on GitHub" links. */
export function blobUrl(path) {
  return `${BLOB_BASE}${path}`;
}

/** Raw URL of a file of the commit the examples are read from, for the browser to fetch. */
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
