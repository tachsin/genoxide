"use client";

import { createContext, useContext, useMemo, useState } from "react";

/**
 * The library highlighted in every benchmark chart of the page: pinned by a
 * click on its legend entry, previewed while the pointer (or the keyboard's
 * focus) is on one.
 */
const HighlightContext = createContext({ focus: null, pinned: null, pin: () => {}, preview: () => {} });

export function HighlightProvider({ children }) {
  const [pinned, setPinned] = useState(null);
  const [previewed, setPreviewed] = useState(null);
  const value = useMemo(
    () => ({
      focus: previewed ?? pinned,
      pinned,
      pin: (library) => setPinned((current) => (current === library ? null : library)),
      preview: setPreviewed,
    }),
    [pinned, previewed],
  );
  return <HighlightContext.Provider value={value}>{children}</HighlightContext.Provider>;
}

export function useHighlight() {
  return useContext(HighlightContext);
}
