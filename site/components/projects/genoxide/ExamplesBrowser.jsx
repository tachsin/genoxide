"use client";

import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import ExampleCard from "./ExampleCard";
import ExampleFamilyCard from "./ExampleFamilyCard";

const ALL = "All";

/** @typedef {import("@/lib/projects/genoxide/examples").ExampleGridEntry} ExampleGridEntry */

/** What a search looks in for an example: its title, short label, summary and category. */
function searchText(example) {
  return `${example.title} ${example.label ?? ""} ${example.summary} ${example.category}`.toLowerCase();
}

/**
 * Whether a text has the query at the start of a word: "g1" finds g10 to g19, not WFG1; a query of
 * several words is matched as written. Null for an empty query.
 * @param {string} query
 * @returns {((text: string) => boolean) | null}
 */
function matcher(query) {
  const q = query.trim().toLowerCase();
  if (!q) return null;
  const escaped = q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`(?:^|[^\\p{L}\\p{N}])${escaped}`, "u");
  return (text) => pattern.test(text);
}

/**
 * Filterable grid of examples: a category row and a text search. A family (the problems of one
 * paper) is one card with a chip per problem; a search that matches some of its problems shows the
 * card with those highlighted, one that matches its name all of it. The full list is
 * server-rendered (the initial state is "All", no query), so every example's link is in the HTML
 * for search engines; filtering is client-only.
 *
 * The category counts are examples, a family's problems each, as in the sidebar; a family's card
 * says how many problems it has.
 *
 * @param {object} props
 * @param {ExampleGridEntry[]} props.entries
 * @param {string[]} props.categories
 * @param {string} props.basePath
 */
export default function ExamplesBrowser({ entries, categories, basePath }) {
  const [category, setCategory] = useState(ALL);
  const [query, setQuery] = useState("");

  const { counts, total, texts } = useMemo(() => {
    const counts = new Map();
    const texts = new Map();
    let total = 0;
    for (const entry of entries) {
      for (const e of entry.kind === "family" ? entry.members : [entry]) {
        counts.set(e.category, (counts.get(e.category) ?? 0) + 1);
        texts.set(e.slug, searchText(e));
        total += 1;
      }
    }
    return { counts, total, texts };
  }, [entries]);

  // each shown card, with the members it shows highlighted when not all of them match
  const { visible, shown } = useMemo(() => {
    const found = matcher(query);
    const inCategory = (e) => category === ALL || e.category === category;
    const visible = [];
    let shown = 0;
    for (const entry of entries) {
      if (entry.kind === "example") {
        if (inCategory(entry) && (!found || found(texts.get(entry.slug)))) {
          visible.push({ entry, matches: null });
          shown += 1;
        }
        continue;
      }
      const pool = entry.members.filter(inCategory);
      let matched = pool;
      // the family's name matches all of it; else its members that match; else its own text
      if (found && !found(entry.name.toLowerCase())) {
        matched = pool.filter((m) => found(texts.get(m.slug)));
        if (!matched.length && found(`${entry.summary} ${entry.cite ?? ""}`.toLowerCase())) matched = pool;
      }
      if (!matched.length) continue;
      const matches = matched.length === entry.members.length ? null : new Set(matched.map((m) => m.slug));
      visible.push({ entry, matches });
      shown += matched.length;
    }
    return { visible, shown };
  }, [entries, texts, category, query]);

  return (
    <div>
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div role="group" aria-label="Category" className="flex flex-wrap gap-2">
          {[ALL, ...categories].map((c) => (
            <button
              key={c}
              type="button"
              className="proj-chip"
              aria-pressed={category === c}
              onClick={() => setCategory(c)}
            >
              {c}
              <span className="proj-chip-count">{c === ALL ? total : counts.get(c)}</span>
            </button>
          ))}
        </div>
        <label className="relative block w-full lg:max-w-xs">
          <span className="sr-only">Search the examples</span>
          <Search
            size={16}
            aria-hidden
            className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-base-content/45"
          />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search examples"
            className="proj-search"
          />
        </label>
      </div>

      <p className="sr-only" aria-live="polite">
        {shown} {shown === 1 ? "example" : "examples"} shown
      </p>

      {visible.length ? (
        <ul className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {visible.map(({ entry, matches }) =>
            entry.kind === "family" ? (
              <li key={`family:${entry.name}`}>
                <ExampleFamilyCard family={entry} matches={matches} basePath={basePath} />
              </li>
            ) : (
              <li key={entry.slug}>
                <ExampleCard example={entry} basePath={basePath} />
              </li>
            ),
          )}
        </ul>
      ) : (
        <div className="proj-card mt-8 p-8 text-center">
          <p className="font-medium">No example matches.</p>
          <button
            type="button"
            className="btn btn-sm btn-ghost mt-3"
            onClick={() => {
              setCategory(ALL);
              setQuery("");
            }}
          >
            Show all examples
          </button>
        </div>
      )}
    </div>
  );
}
