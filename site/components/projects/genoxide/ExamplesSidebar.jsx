"use client";

import { ChevronRight, PanelLeftOpen, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useId, useMemo, useRef, useState } from "react";

/** @typedef {import("@/lib/projects/genoxide/examples").ExampleTreeEntry} ExampleTreeEntry */
/** @typedef {{ category: string, entries: ExampleTreeEntry[] }[]} ExampleTree */

/** The slug of the example the path shows, "" for the index, null for another page. */
function currentSlug(pathname, basePath) {
  if (pathname === basePath || pathname === `${basePath}/`) return "";
  if (!pathname?.startsWith(`${basePath}/`)) return null;
  return decodeURIComponent(pathname.slice(basePath.length + 1).split("/")[0]);
}

/** Scrolls `container` (only) so that its current link shows, when it doesn't. */
function revealCurrent(container) {
  const link = container?.querySelector('[aria-current="page"]');
  if (!(link instanceof HTMLElement)) return;
  const top = link.getBoundingClientRect().top - container.getBoundingClientRect().top;
  if (top < 0 || top + link.offsetHeight > container.clientHeight) {
    container.scrollTop += top - (container.clientHeight - link.offsetHeight) / 2;
  }
}

const LINK =
  "block truncate rounded-md px-2 py-1.5 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-primary focus-visible:-outline-offset-2";
const LINK_IDLE = "text-base-content/70 hover:bg-base-content/5 hover:text-base-content";
const LINK_CURRENT = "bg-primary/10 font-medium text-primary";

/**
 * Every example: by category, a family's problems under its name (collapsed, but for the current
 * example's family), the current example highlighted.
 *
 * @param {{ tree: ExampleTree, basePath: string, total: number, current: string | null, onNavigate?: () => void }} props
 */
function ExamplesNav({ tree, basePath, total, current, onNavigate }) {
  const id = useId();
  const currentFamily = useMemo(() => {
    for (const { entries } of tree) {
      for (const entry of entries) {
        if (entry.kind === "family" && entry.members.some((m) => m.slug === current)) return entry.name;
      }
    }
    return null;
  }, [tree, current]);

  // The families opened by hand stay open; the current example's opens as it becomes current.
  const [expanded, setExpanded] = useState(() => new Set(currentFamily ? [currentFamily] : []));
  const [shownFamily, setShownFamily] = useState(currentFamily);
  if (shownFamily !== currentFamily) {
    setShownFamily(currentFamily);
    if (currentFamily && !expanded.has(currentFamily)) setExpanded(new Set(expanded).add(currentFamily));
  }

  const toggle = (name) => {
    const next = new Set(expanded);
    if (!next.delete(name)) next.add(name);
    setExpanded(next);
  };

  const link = (slug, label, title) => (
    <Link
      href={`${basePath}/${slug}`}
      aria-current={slug === current ? "page" : undefined}
      title={title && title !== label ? title : undefined}
      onClick={onNavigate}
      className={`${LINK} ${slug === current ? LINK_CURRENT : LINK_IDLE}`}
    >
      {label}
    </Link>
  );

  return (
    <nav aria-label="All examples">
      <Link
        href={basePath}
        aria-current={current === "" ? "page" : undefined}
        onClick={onNavigate}
        className={`${LINK} flex items-center justify-between gap-2 font-semibold ${
          current === "" ? LINK_CURRENT : "text-base-content hover:bg-base-content/5"
        }`}
      >
        All examples
        <span className="font-normal text-base-content/45 text-xs">{total}</span>
      </Link>
      {tree.map(({ category, entries }, g) => (
        <div key={category} className="mt-5">
          <p id={`${id}-c${g}`} className="px-2 font-semibold text-[0.7rem] text-base-content/50 uppercase tracking-wider">
            {category}
          </p>
          <ul aria-labelledby={`${id}-c${g}`} className="mt-1 space-y-px">
            {entries.map((entry) => {
              if (entry.kind === "example") return <li key={entry.slug}>{link(entry.slug, entry.title)}</li>;
              const open = expanded.has(entry.name);
              const listId = `${id}-f${g}-${entry.members[0].slug}`;
              const holdsCurrent = entry.name === currentFamily;
              return (
                <li key={`family:${entry.name}`}>
                  <button
                    type="button"
                    aria-expanded={open}
                    aria-controls={listId}
                    onClick={() => toggle(entry.name)}
                    className={`flex w-full items-center gap-1 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-base-content/5 focus-visible:outline-2 focus-visible:outline-primary focus-visible:-outline-offset-2 ${
                      holdsCurrent && !open ? "font-medium text-primary" : "text-base-content/80 hover:text-base-content"
                    }`}
                  >
                    <ChevronRight
                      size={14}
                      aria-hidden
                      className={`-ml-0.5 shrink-0 text-base-content/45 transition-transform ${open ? "rotate-90" : ""}`}
                    />
                    <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                    <span className="text-base-content/45 text-xs">{entry.members.length}</span>
                  </button>
                  <ul id={listId} hidden={!open} className="mt-px ml-3.5 space-y-px border-base-content/10 border-l pl-1.5">
                    {entry.members.map((member) => (
                      <li key={member.slug}>{link(member.slug, member.label, member.title)}</li>
                    ))}
                  </ul>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
  );
}

/**
 * The sidebar of the examples: a sticky column beside the content on wide screens (xl and up), and
 * below that a button that opens it as a drawer (a modal dialog: focus moves into it and back to
 * the button, Escape or a click outside closes it, as does following a link).
 *
 * @param {{ tree: ExampleTree, basePath: string, total: number }} props
 */
export default function ExamplesSidebar({ tree, basePath, total }) {
  const pathname = usePathname();
  const current = currentSlug(pathname, basePath);
  const columnRef = useRef(/** @type {HTMLDivElement | null} */ (null));
  const buttonRef = useRef(/** @type {HTMLButtonElement | null} */ (null));
  const dialogRef = useRef(/** @type {HTMLDialogElement | null} */ (null));
  const drawerListRef = useRef(/** @type {HTMLDivElement | null} */ (null));
  const [open, setOpen] = useState(false);

  // the column keeps its scroll across pages (it's in the layout): show the new current example
  useEffect(() => {
    if (current !== null) revealCurrent(columnRef.current);
  }, [current]);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!open || !dialog) return;
    if (!dialog.open) dialog.showModal();
    revealCurrent(drawerListRef.current);
    const link = drawerListRef.current?.querySelector('[aria-current="page"]');
    if (link instanceof HTMLElement) link.focus({ preventScroll: true });
    // the page behind doesn't scroll while the drawer is open
    const root = document.documentElement;
    const overflow = root.style.overflow;
    root.style.overflow = "hidden";
    return () => {
      root.style.overflow = overflow;
    };
  }, [open]);

  // a navigation (a link, or back and forward) closes the drawer
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs on a new path only
  useEffect(() => {
    dialogRef.current?.close();
  }, [pathname]);

  const close = () => dialogRef.current?.close();

  return (
    <>
      <div className="hidden pt-14 xl:block">
        <div
          ref={columnRef}
          className="sticky top-28 max-h-[calc(100dvh-7.5rem)] overflow-y-auto overscroll-contain pr-2 pb-8 [scrollbar-width:thin]"
        >
          <ExamplesNav tree={tree} basePath={basePath} total={total} current={current} />
        </div>
      </div>

      <div className="proj-container -mb-4 pt-6 xl:hidden">
        <button
          ref={buttonRef}
          type="button"
          aria-haspopup="dialog"
          aria-expanded={open}
          onClick={() => setOpen(true)}
          className="proj-pill cursor-pointer focus-visible:outline-2 focus-visible:outline-primary focus-visible:outline-offset-2"
        >
          <PanelLeftOpen size={15} aria-hidden />
          All examples
        </button>
      </div>

      {open ? (
        // a click on the backdrop closes it too (its target is the dialog itself); Escape natively
        <dialog
          ref={dialogRef}
          aria-label="All examples"
          onClose={() => {
            setOpen(false);
            buttonRef.current?.focus();
          }}
          onClick={(event) => {
            if (event.target === dialogRef.current) close();
          }}
          className="m-0 h-dvh max-h-dvh w-[min(21rem,88vw)] max-w-none overflow-hidden border-base-content/10 border-r bg-base-100 p-0 text-base-content shadow-2xl backdrop:bg-black/50"
        >
          <div className="flex h-full flex-col">
            <div className="flex items-center justify-between gap-2 border-base-content/10 border-b px-4 py-3">
              <span className="font-semibold">Examples</span>
              <button
                type="button"
                onClick={close}
                aria-label="Close"
                className="btn btn-ghost btn-sm btn-square"
              >
                <X size={18} aria-hidden />
              </button>
            </div>
            <div ref={drawerListRef} className="relative min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 pt-3 pb-8">
              <ExamplesNav tree={tree} basePath={basePath} total={total} current={current} onNavigate={close} />
            </div>
          </div>
        </dialog>
      ) : null}
    </>
  );
}
