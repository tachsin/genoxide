import { ArrowUpRight } from "lucide-react";
import Link from "next/link";
import { CODE_LANG_LABELS } from "@/components/projects/code-lang";

// The look of ExampleCard's link (a.proj-card) on a card that isn't one link: it lifts when the
// pointer is over any part of it, and its outline shows when its title's link has the focus.
// Important (!), as .proj-card's own border and transition are outside Tailwind's layers.
const CARD =
  "proj-card group relative flex h-full flex-col p-5 hover:border-primary/45! motion-safe:hover:[transform:translateY(-2px)] hover:shadow-[0_18px_40px_-26px_color-mix(in_oklab,var(--color-primary)_70%,transparent)]";

const CHIP =
  "relative z-10 block rounded-md border px-2 py-1 font-medium text-xs leading-4 transition-[color,background-color,border-color,opacity] focus-visible:outline-2 focus-visible:outline-primary focus-visible:outline-offset-2";
const CHIP_IDLE = "border-base-content/10 text-base-content/70 hover:border-primary/50 hover:text-primary";
const CHIP_MATCH = "border-primary/40 bg-primary/10 text-primary hover:border-primary";
const CHIP_DIM = "border-base-content/10 text-base-content/70 opacity-55 hover:opacity-100 hover:text-primary";

/**
 * The problems of one paper (a family, e.g. WFG1 to WFG9) as one card of the examples grid: its
 * name, what the problems test, the paper, and a chip for each problem, a link to its page. The
 * card itself links to the first problem (the first that matches a search, when some don't): its
 * title's link covers the card, under the chips. No hooks, like ExampleCard.
 *
 * @param {object} props
 * @param {import("@/lib/projects/genoxide/examples").ExampleFamilyCardData} props.family
 * @param {Set<string> | null} [props.matches]  the slugs of the members a search or filter matches
 *   when not all do: those are highlighted, the others dimmed
 * @param {string} props.basePath
 */
export default function ExampleFamilyCard({ family, matches = null, basePath }) {
  const { name, category, summary, cite, languages, members } = family;
  const first = members.find((m) => !matches || matches.has(m.slug)) ?? members[0];
  const headingId = `family-${members[0].slug}`;
  const rustOnly = languages.length === 1 && languages[0] === "rust";
  return (
    <article className={CARD} aria-labelledby={headingId}>
      <div className="flex items-center justify-between gap-3">
        <span className="proj-tag" data-tone="accent">
          {category}
        </span>
        <ArrowUpRight
          size={16}
          aria-hidden
          className="text-base-content/35 transition-colors group-hover:text-primary"
        />
      </div>
      <h3 id={headingId} className="mt-3 flex flex-wrap items-baseline gap-x-2 font-semibold text-lg tracking-tight">
        <Link
          href={`${basePath}/${first.slug}`}
          className="rounded-2xl outline-none after:absolute after:inset-0 after:rounded-2xl focus-visible:after:outline-2 focus-visible:after:outline-primary focus-visible:after:outline-offset-2"
        >
          {name}
        </Link>
        <span className={`font-normal text-sm ${matches ? "text-primary" : "text-base-content/50"}`}>
          {matches ? `${matches.size} of ${members.length} problems` : `${members.length} problems`}
        </span>
      </h3>
      {summary ? <p className="proj-lead mt-1.5 line-clamp-3 text-sm">{summary}</p> : null}
      {cite ? <p className="mt-1.5 truncate text-base-content/50 text-xs">{cite}</p> : null}
      <ul aria-label={`The ${name} problems`} className="flex flex-wrap gap-1.5 pt-4">
        {members.map((member) => {
          const tone = !matches ? CHIP_IDLE : matches.has(member.slug) ? CHIP_MATCH : CHIP_DIM;
          const full = member.label === member.title ? undefined : member.title;
          return (
            <li key={member.slug}>
              <Link href={`${basePath}/${member.slug}`} aria-label={full} title={full} className={`${CHIP} ${tone}`}>
                {member.label}
              </Link>
            </li>
          );
        })}
      </ul>
      {languages.length ? (
        <div className="mt-auto flex flex-wrap gap-1.5 pt-4">
          {rustOnly ? (
            <span className="proj-tag">Rust only</span>
          ) : (
            languages.map((l) => (
              <span key={l} className="proj-tag">
                {CODE_LANG_LABELS[l] ?? l}
              </span>
            ))
          )}
        </div>
      ) : null}
    </article>
  );
}
