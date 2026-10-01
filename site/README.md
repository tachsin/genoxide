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
| `app/projects/genoxide/` | the routes: overview, `/examples`, `/examples/[slug]` and their layout (the sidebar of every example), `/benchmarks` and `/benchmarks/run` (a route handler: one method's runs in one scenario, as JSON, for the benchmark page's details panel), the sub-navigation layout and the Open Graph image |
| `lib/projects/genoxide/` | the data: static facts (`meta.js`), and the examples, benchmarks and versions read from GitHub, crates.io and PyPI |

The examples (and the benchmark page's file list, its chart data, `docs/benchmarks/charts.json`, and its run details, `docs/benchmarks/runs/<scenario>.json` with the adapters' files they point to) are read from GitHub at the pinned commit, not
at `main`: the pages always get the files they were written for, and pinning a new commit is new
URLs, so nothing the app cached from an older commit is served for it. An example added to `main`
shows once the app pins a commit that has it. Links for readers ("view on GitHub") go to `main`.
| `components/projects/genoxide/` | the components only these pages use; `ExamplesBrowser` is the index's grid (a category filter and a search) of `ExampleCard`s, a family's problems in one `ExampleFamilyCard` with a link to each, `ExamplesSidebar` lists every example (a column on wide screens, a drawer below), `ExampleFamilyTabs` shows the problems of an example's paper (its `family` in the front matter) as tabs, `player/` plays an example's recorded run (its `trace.json`), `benchmarks/` draws the benchmark page's interactive charts from `charts.json`, and `RunDetails` shows the runs, output and code of the bar selected in them (`#run=<scenario>/<library>/<solver>` in the URL), from `/benchmarks/run`; `game/` is the overview's hero: `EvolvingTitle` evolves the title from random letters (Dawkins' weasel), `HeroSwarm` draws double helices bred by a small GA behind it (a canvas, decorative, still under reduced motion), and its button opens `camouflage/CamouflageGame`: the visitor is a bird hunting moths on bark, the survivors breed (one-point crossover and Gaussian mutation, shown on each moth's five-gene DNA strip), and the moths evolve camouflage; five levels add natural selection, a changing environment (re-evaluation), two habitats with migration (islands), mates against predators (a Pareto front) and genoxide hunting by tournament selection |

Nothing else in `site/` is copied: `scripts/berlin_districts.py` regenerates the Berlin map of the berlin52 example's tour plot (`player/plots/berlin.js`). Imports use the app's `@/` alias, which is the app's root.

To see edits on the app's dev server: `pnpm dev` there, then
`pnpm sync:genoxide --from <this folder> --watch`.

## What the pages use from the app

The contract between the two repositories: these have to exist in the app, with these exports.

| Module | Exports |
| --- | --- |
| `@/genoxide-site.json` | `commit`: the pinned commit, the one these pages were synced from |
| `@/components/projects/Breadcrumbs` | default |
| `@/components/projects/CopyButton` | default (copies the code of its `[data-copy-root]`) |
| `@/components/projects/JsonLd` | default |
| `@/components/projects/LangCodeGroup` | default (the Rust/Python code tabs) |
| `@/components/projects/ProjectSubNav` | default |
| `@/components/projects/SourceUnavailable` | default |
| `@/components/projects/code-lang` | `CODE_LANG_LABELS` |
| `@/lib/projects/fetch-cached` | `fetchCached` |
| `@/lib/projects/highlight` | `renderMarkdown`, `highlightCode` (the benchmark run details: the adapters' code, in Rust, Python, Java, Julia or C++, plain where the app's highlighter has no grammar for it, and the runs' output) |
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
