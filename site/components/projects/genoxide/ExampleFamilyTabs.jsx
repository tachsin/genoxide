"use client";

import { ExternalLink } from "lucide-react";
import Link from "next/link";
import { useLayoutEffect, useRef } from "react";

// Where the tab bar was on screen when one of its tabs was clicked: the next page's tab bar is put
// back there, so switching problems swaps what's below it without moving it, like a tab. Module
// scope, as the page (and this component with it) is replaced by the navigation.
/** @type {{ slug: string, top: number } | null} */
let pending = null;

/**
 * The problems of the example's paper (its family) as tabs, at the top of its page. Each tab is a
 * link to that problem's own page, so each keeps its URL, title and metadata; the links don't
 * scroll to the top, and the tab bar stays where it was on screen.
 *
 * Prefetching: the pages are rendered per request (the site's nonce CSP), so a full prefetch is a
 * render on the server. The tabs next to the current one are prefetched in full as they show, the
 * others (up to 17 for CEC 2006) when the pointer is over them or they're touched.
 *
 * @param {object} props
 * @param {import("@/lib/projects/genoxide/examples").ExampleFamily} props.family
 * @param {string} props.current  the slug of this page's example
 * @param {string} props.basePath
 */
export default function ExampleFamilyTabs({ family, current, basePath }) {
  const listRef = useRef(/** @type {HTMLUListElement | null} */ (null));

  useLayoutEffect(() => {
    const list = listRef.current;
    if (!list) return;
    if (pending?.slug === current) {
      const moved = list.getBoundingClientRect().top - pending.top;
      if (Math.abs(moved) >= 1) window.scrollBy({ top: moved, behavior: "instant" });
    }
    pending = null;
    // On a narrow screen the tabs scroll sideways: bring the current one into view (sideways
    // only, not the page).
    const tab = list.querySelector('[aria-current="page"]');
    if (tab instanceof HTMLElement && list.scrollWidth > list.clientWidth) {
      list.scrollLeft = tab.offsetLeft - (list.clientWidth - tab.offsetWidth) / 2;
    }
  }, [current]);

  const { name, sources, members } = family;
  const currentIndex = members.findIndex((member) => member.slug === current);
  return (
    <nav aria-label={`The ${name} problems`} className="mb-8">
      <p className="proj-eyebrow">
        {name} · {members.length} problems
      </p>
      {sources.length ? (
        <ul className="mt-1.5 space-y-0.5 text-base-content/60 text-sm">
          {sources.map((source) => (
            <li key={source.reference}>
              {source.referenceUrl ? (
                <a
                  href={source.referenceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline decoration-base-content/20 underline-offset-2 transition-colors hover:text-base-content hover:decoration-primary"
                >
                  {source.reference}
                  <ExternalLink size={11} aria-hidden className="ml-1 inline align-baseline opacity-60" />
                </a>
              ) : (
                source.reference
              )}
            </li>
          ))}
        </ul>
      ) : null}
      <ul
        ref={listRef}
        className="relative mt-4 flex gap-0.5 overflow-x-auto overscroll-x-contain border-base-content/10 border-b [scrollbar-width:thin]"
      >
        {members.map((member, i) => {
          const isCurrent = member.slug === current;
          const isNeighbor = Math.abs(i - currentIndex) === 1;
          return (
            <li key={member.slug} className="shrink-0">
              <Link
                href={`${basePath}/${member.slug}`}
                prefetch={isNeighbor ? true : null}
                unstable_dynamicOnHover
                scroll={false}
                aria-current={isCurrent ? "page" : undefined}
                title={member.label === member.title ? undefined : member.title}
                onClick={(event) => {
                  if (isCurrent || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
                    return;
                  }
                  const list = listRef.current;
                  if (list) pending = { slug: member.slug, top: list.getBoundingClientRect().top };
                }}
                className={`block whitespace-nowrap rounded-t-md px-3 py-2 font-medium text-sm transition-colors focus-visible:outline-2 focus-visible:outline-primary focus-visible:-outline-offset-2 ${
                  isCurrent
                    ? "text-base-content shadow-[inset_0_-2px_0_var(--color-primary)]"
                    : "text-base-content/60 hover:bg-base-content/5 hover:text-base-content"
                }`}
              >
                {member.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
