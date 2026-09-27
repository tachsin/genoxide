"use client";

import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import ExampleCard from "./ExampleCard";

const ALL = "All";

/**
 * Filterable grid of examples: a category row and a text search. The full
 * list is server-rendered (the initial state is "All", no query), so every
 * example is in the HTML for search engines; filtering is client-only.
 *
 * @param {object} props
 * @param {{ slug: string, title: string, category: string, summary: string, languages: string[] }[]} props.examples
 * @param {string[]} props.categories
 * @param {string} props.basePath
 */
export default function ExamplesBrowser({ examples, categories, basePath }) {
  const [category, setCategory] = useState(ALL);
  const [query, setQuery] = useState("");

  const counts = useMemo(() => {
    const map = new Map();
    for (const e of examples) map.set(e.category, (map.get(e.category) ?? 0) + 1);
    return map;
  }, [examples]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return examples.filter(
      (e) =>
        (category === ALL || e.category === category) &&
        (!q || `${e.title} ${e.summary} ${e.category}`.toLowerCase().includes(q)),
    );
  }, [examples, category, query]);

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
              <span className="proj-chip-count">{c === ALL ? examples.length : counts.get(c)}</span>
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
        {visible.length} {visible.length === 1 ? "example" : "examples"} shown
      </p>

      {visible.length ? (
        <ul className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {visible.map((e) => (
            <li key={e.slug}>
              <ExampleCard example={e} basePath={basePath} />
            </li>
          ))}
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
