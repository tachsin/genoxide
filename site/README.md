# site

genoxide's pages on tachsin.gr, [tachsin.gr/projects/genoxide](https://tachsin.gr/projects/genoxide):
the Next.js routes, data loaders and components of `/projects/genoxide`.

## How it reaches tachsin.gr

tachsin.gr is a Next.js app (Next 16, React, Tailwind 4, daisyUI) in a private repository. Its
`pnpm sync:genoxide` downloads this folder at the commit pinned in its `genoxide-site.json` and
copies each folder below to the same path in the app, replacing it whole. The app's build and dev
scripts run the sync first, so a deploy builds the pinned commit. A change here goes live when the
app pins a commit that has it (`pnpm sync:genoxide --update` pins main's latest).

| Folder | In the app |
| --- | --- |
| `app/projects/genoxide/` | the routes: overview, `/examples`, `/examples/[slug]`, `/benchmarks`, the sub-navigation layout and the Open Graph image |
| `lib/projects/genoxide/` | the data: static facts (`meta.js`), and the examples, benchmarks and versions read from GitHub, crates.io and PyPI |

The examples (and the benchmark pages' file list) are read from GitHub at the pinned commit, not
at `main`: the pages always get the files they were written for, and pinning a new commit is new
URLs, so nothing the app cached from an older commit is served for it. An example added to `main`
shows once the app pins a commit that has it. Links for readers ("view on GitHub") go to `main`.
| `components/projects/genoxide/` | the components only these pages use; `player/` plays an example's recorded run (its `trace.json`) |

Nothing else in `site/` is copied: `scripts/berlin_districts.py` regenerates the Berlin map of the berlin52 example's tour plot (`player/plots/berlin.js`). Imports use the app's `@/` alias, which is the app's root.

To see edits on the app's dev server: `pnpm dev` there, then
`pnpm sync:genoxide --from <this folder> --watch`.

## What the pages use from the app

The contract between the two repositories: these have to exist in the app, with these exports.

| Module | Exports |
| --- | --- |
| `@/genoxide-site.json` | `commit`: the pinned commit, the one these pages were synced from |
| `@/components/projects/Breadcrumbs` | default |
| `@/components/projects/JsonLd` | default |
| `@/components/projects/LangCodeGroup` | default (the Rust/Python code tabs) |
| `@/components/projects/ProjectSubNav` | default |
| `@/components/projects/SourceUnavailable` | default |
| `@/components/projects/code-lang` | `CODE_LANG_LABELS` |
| `@/lib/projects/fetch-cached` | `fetchCached` |
| `@/lib/projects/highlight` | `renderMarkdown` |
| `@/lib/projects/json-ld` | `WEBSITE_ID`, `breadcrumbList`, `website` |
| `@/lib/projects/metadata` | `projectsMetadata` |
| `@/lib/projects/og-image` | `OG_CONTENT_TYPE`, `OG_SIZE`, `projectsOgImage` |
| `@/lib/projects/registry` | `getProject` (and its `genoxide` entry: name, href, sections) |
| `@/lib/tools-metadata` | `absoluteUrl` |

Also from the app: the `/projects` layout around these pages, the `proj-*` classes of
`app/projects/projects.css`, Tailwind and daisyUI utilities (the app's `app/globals.css` lists the
three folders with `@source`), and the npm packages `next`, `react`, `lucide-react`, `react-icons`
and `yaml`. The player fetches `trace.json` in the browser from raw.githubusercontent.com, so the
app's Content-Security-Policy has to allow it in `connect-src`.

The other way, the app uses `genoxideExamplePaths` from `lib/projects/genoxide/examples.js` for its
sitemap, and links to `/projects/genoxide/examples` from its `/projects` 404 page.
