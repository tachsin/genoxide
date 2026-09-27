import { ArrowUpRight } from "lucide-react";
import Link from "next/link";
import { CODE_LANG_LABELS } from "@/components/projects/code-lang";

/**
 * One example in a grid. No hooks, so it renders from the server page
 * (landing) and inside the client ExamplesBrowser alike.
 *
 * @param {{ example: { slug: string, title: string, category: string, summary: string, languages: string[] }, basePath: string }} props
 */
export default function ExampleCard({ example, basePath }) {
  const rustOnly = example.languages.length === 1 && example.languages[0] === "rust";
  return (
    <Link href={`${basePath}/${example.slug}`} className="proj-card group flex h-full flex-col p-5">
      <div className="flex items-center justify-between gap-3">
        <span className="proj-tag" data-tone="accent">
          {example.category}
        </span>
        <ArrowUpRight
          size={16}
          aria-hidden
          className="text-base-content/35 transition-colors group-hover:text-primary"
        />
      </div>
      <h3 className="mt-3 font-semibold text-lg tracking-tight">{example.title}</h3>
      {example.summary ? <p className="proj-lead mt-1.5 line-clamp-3 text-sm">{example.summary}</p> : null}
      <div className="mt-auto flex flex-wrap gap-1.5 pt-4">
        {rustOnly ? (
          <span className="proj-tag">Rust only</span>
        ) : (
          example.languages.map((l) => (
            <span key={l} className="proj-tag">
              {CODE_LANG_LABELS[l] ?? l}
            </span>
          ))
        )}
      </div>
    </Link>
  );
}
