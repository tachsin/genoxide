"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

/**
 * The library highlighted in every benchmark chart of the page: pinned by a
 * click on its legend entry, previewed while the pointer (or the keyboard's
 * focus) is on one.
 *
 * And the run selected on the page: a bar's library and method in its
 * scenario, whose runs, output and code the details panel shows (RunDetails).
 * It lives in the URL's hash, `#run=<scenario>/<library>/<solver>`, so a link
 * opens the page with it, and each selection is a history entry the back
 * button returns from. Selecting a run pins its library.
 */
const HighlightContext = createContext({
  focus: null,
  pinned: null,
  pin: () => {},
  preview: () => {},
  details: false,
  run: null,
  select: () => {},
  scrollRequest: 0,
});

const RUN_HASH = /^#run=([a-z0-9]+-\d+-[a-z]+)\/([a-z0-9_]+)\/([a-z0-9_]+)$/;

/** The run a URL hash selects, or null. */
export function parseRunHash(hash) {
  const match = RUN_HASH.exec(hash ?? "");
  return match ? { scenario: match[1], library: match[2], solver: match[3] } : null;
}

/** The hash of a selected run. */
export function runHash(run) {
  return `#run=${run.scenario}/${run.library}/${run.solver}`;
}

/** Whether two selections are the same run. */
export function sameRun(a, b) {
  return Boolean(a && b && a.scenario === b.scenario && a.library === b.library && a.solver === b.solver);
}

export function HighlightProvider({ children, details = false }) {
  const [pinned, setPinned] = useState(null);
  const [previewed, setPreviewed] = useState(null);
  const [run, setRun] = useState(null);
  // incremented when the details panel should scroll into view: a selection, or a link's hash
  const [scrollRequest, setScrollRequest] = useState(0);

  // the hash's run when the page opens, and when back and forward (or a link) change it; the
  // browser restores the scroll position of a history entry itself
  useEffect(() => {
    if (!details) return undefined;
    const opened = parseRunHash(window.location.hash);
    if (opened) {
      setRun(opened);
      setPinned(opened.library);
      setScrollRequest((n) => n + 1);
    }
    const follow = () => {
      const next = parseRunHash(window.location.hash);
      setRun((current) => (sameRun(current, next) ? current : next));
      setPinned(next ? next.library : null);
    };
    window.addEventListener("popstate", follow);
    window.addEventListener("hashchange", follow);
    return () => {
      window.removeEventListener("popstate", follow);
      window.removeEventListener("hashchange", follow);
    };
  }, [details]);

  const select = useCallback((next) => {
    const { pathname, search, hash } = window.location;
    const target = next ? runHash(next) : "";
    if (hash !== target) window.history.pushState(null, "", `${pathname}${search}${target}`);
    setRun(next);
    setPinned(next ? next.library : null);
    if (next) setScrollRequest((n) => n + 1);
  }, []);

  const value = useMemo(
    () => ({
      focus: previewed ?? pinned,
      pinned,
      pin: (library) => setPinned((current) => (current === library ? null : library)),
      preview: setPreviewed,
      // whether the page has the run details: without them (a pin from before them), no bar selects
      details,
      run: details ? run : null,
      select,
      scrollRequest,
    }),
    [pinned, previewed, details, run, select, scrollRequest],
  );
  return <HighlightContext.Provider value={value}>{children}</HighlightContext.Provider>;
}

export function useHighlight() {
  return useContext(HighlightContext);
}
