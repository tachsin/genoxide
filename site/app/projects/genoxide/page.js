import {
  ArrowRight,
  BookOpen,
  ChartBar,
  Cpu,
  Dna,
  Mountain,
  Orbit,
  Package,
  Scale,
  Shuffle,
  Sparkles,
  Terminal,
} from "lucide-react";
import Link from "next/link";
import { SiDocsdotrs, SiGithub, SiPypi, SiPython } from "react-icons/si";
import ExampleCard from "@/components/projects/genoxide/ExampleCard";
import EvolvingTitle from "@/components/projects/genoxide/game/EvolvingTitle";
import HeroSwarm from "@/components/projects/genoxide/game/HeroSwarm";
import JsonLd from "@/components/projects/JsonLd";
import LangCodeGroup from "@/components/projects/LangCodeGroup";
import SourceUnavailable from "@/components/projects/SourceUnavailable";
import { EXAMPLES_PATH, getExamples } from "@/lib/projects/genoxide/examples";
import {
  GENOXIDE_DESCRIPTION,
  GENOXIDE_FIRST_EXAMPLE,
  GENOXIDE_HIGHLIGHTS,
  GENOXIDE_KEYWORDS,
  GENOXIDE_LICENSE,
  GENOXIDE_LINKS,
  GENOXIDE_OG_IMAGE,
  GENOXIDE_PATH,
  GENOXIDE_PROPERTIES,
  GENOXIDE_TAGLINE,
} from "@/lib/projects/genoxide/meta";
import { getGenoxideVersions } from "@/lib/projects/genoxide/versions";
import { WEBSITE_ID, breadcrumbList, website } from "@/lib/projects/json-ld";
import { projectsMetadata } from "@/lib/projects/metadata";
import { absoluteUrl } from "@/lib/tools-metadata";

export const metadata = projectsMetadata({
  title: "genoxide",
  fullTitle: `genoxide — ${GENOXIDE_TAGLINE}`,
  description: GENOXIDE_DESCRIPTION,
  path: GENOXIDE_PATH,
  image: GENOXIDE_OG_IMAGE,
  imageAlt: `genoxide — ${GENOXIDE_TAGLINE}`,
  keywords: GENOXIDE_KEYWORDS,
});

const HIGHLIGHT_ICONS = {
  genomes: Dna,
  ga: Shuffle,
  es: Orbit,
  de: Sparkles,
  local: Mountain,
  multi: Scale,
  engine: Cpu,
  python: Terminal,
};

const HERO_LINKS = [
  { label: "GitHub", href: GENOXIDE_LINKS.github, Icon: SiGithub },
  { label: "crates.io", href: GENOXIDE_LINKS.crates, Icon: Package },
  { label: "PyPI", href: GENOXIDE_LINKS.pypi, Icon: SiPypi },
  { label: "docs.rs", href: GENOXIDE_LINKS.docsRs, Icon: SiDocsdotrs },
  { label: "Python API", href: GENOXIDE_LINKS.pythonApi, Icon: SiPython },
];

const RESOURCES = [
  { label: "Documentation and examples", href: GENOXIDE_LINKS.docsSite, note: "tachsin.github.io/genoxide" },
  { label: "Rust API reference", href: GENOXIDE_LINKS.docsRs, note: "docs.rs" },
  { label: "Python API reference", href: GENOXIDE_LINKS.pythonApi, note: "every class and parameter" },
  { label: "Feature list", href: GENOXIDE_LINKS.features, note: "docs/features.md" },
  { label: "Command-line program", href: GENOXIDE_LINKS.cli, note: "fitness functions in any language" },
  { label: "Roadmap", href: GENOXIDE_LINKS.roadmap, note: "what's planned" },
  { label: "Changelog", href: GENOXIDE_LINKS.changelog, note: "every release" },
  { label: "Source code", href: GENOXIDE_LINKS.github, note: "github.com/tachsin/genoxide" },
];

function structuredData(versions) {
  const url = absoluteUrl(GENOXIDE_PATH);
  const version = versions.rust ?? versions.python ?? undefined;
  return {
    "@graph": [
      website(),
      {
        "@type": "SoftwareSourceCode",
        "@id": `${url}#source`,
        name: "genoxide",
        description: GENOXIDE_DESCRIPTION,
        url,
        codeRepository: GENOXIDE_LINKS.github,
        programmingLanguage: [
          { "@type": "ComputerLanguage", name: "Rust" },
          { "@type": "ComputerLanguage", name: "Python" },
        ],
        runtimePlatform: ["Rust", "CPython 3.10+"],
        license: ["https://opensource.org/licenses/MIT", "https://www.apache.org/licenses/LICENSE-2.0"],
        keywords: GENOXIDE_KEYWORDS.join(", "),
        author: { "@type": "Person", name: "tachsin", url: "https://github.com/tachsin" },
        ...(version ? { version } : {}),
        targetProduct: { "@id": `${url}#software` },
        isPartOf: { "@id": WEBSITE_ID },
      },
      {
        "@type": "SoftwareApplication",
        "@id": `${url}#software`,
        name: "genoxide",
        description: GENOXIDE_DESCRIPTION,
        url,
        applicationCategory: "DeveloperApplication",
        applicationSubCategory: "Optimization library",
        operatingSystem: "Linux, macOS, Windows",
        ...(version ? { softwareVersion: version } : {}),
        downloadUrl: GENOXIDE_LINKS.crates,
        installUrl: GENOXIDE_LINKS.pypi,
        softwareHelp: { "@type": "CreativeWork", url: GENOXIDE_LINKS.docsSite },
        license: ["https://opensource.org/licenses/MIT", "https://www.apache.org/licenses/LICENSE-2.0"],
        offers: { "@type": "Offer", price: "0", priceCurrency: "EUR" },
        sameAs: [GENOXIDE_LINKS.github, GENOXIDE_LINKS.crates, GENOXIDE_LINKS.pypi, GENOXIDE_LINKS.docsRs],
      },
      breadcrumbList([
        { name: "Projects", path: "/projects" },
        { name: "genoxide", path: GENOXIDE_PATH },
      ]),
    ],
  };
}

function describeVersions({ rust, python }) {
  if (rust && python && rust === python) return `v${rust} on crates.io and PyPI`;
  if (rust && python) return `v${rust} on crates.io · v${python} on PyPI`;
  if (rust) return `v${rust} on crates.io`;
  if (python) return `v${python} on PyPI`;
  return null;
}

function SectionHeading({ eyebrow, title, children, id }) {
  return (
    <div className="max-w-2xl">
      {eyebrow ? <p className="proj-eyebrow">{eyebrow}</p> : null}
      <h2 id={id} className="mt-2 scroll-mt-32 font-semibold text-3xl tracking-tight">
        {title}
      </h2>
      {children ? <p className="proj-lead mt-3">{children}</p> : null}
    </div>
  );
}

export default async function GenoxidePage() {
  const [versions, examplesResult] = await Promise.all([getGenoxideVersions(), getExamples()]);
  const versionLabel = describeVersions(versions);
  const featured = examplesResult.examples.slice(0, 6);

  return (
    <main>
      <JsonLd data={structuredData(versions)} />

      {/* ---------- Hero ---------- */}
      <section className="proj-container relative isolate pt-16 pb-20 text-center sm:pt-24">
        <div className="proj-rise flex justify-center">
          {versionLabel ? (
            <a
              href={GENOXIDE_LINKS.changelog}
              target="_blank"
              rel="noopener noreferrer"
              className="proj-pill"
            >
              <span className="size-1.5 rounded-full bg-success" aria-hidden />
              <span>{versionLabel}</span>
              <ArrowRight size={13} aria-hidden />
            </a>
          ) : (
            <span className="proj-pill">Open source · {GENOXIDE_LICENSE}</span>
          )}
        </div>

        <EvolvingTitle
          text="genoxide"
          className="proj-rise proj-gradient-text mt-6 font-mono font-semibold text-6xl tracking-tighter sm:text-8xl"
        />
        <p className="proj-rise-1 mx-auto mt-5 max-w-2xl text-balance text-base-content/80 text-lg sm:text-xl">
          {GENOXIDE_TAGLINE}: genetic algorithms, evolution strategies, CMA-ES, differential evolution, particle
          swarms, local search and multi-objective optimization in one library.
        </p>

        <div className="proj-rise-2 mx-auto mt-9 max-w-md">
          <LangCodeGroup
            id="install"
            size="compact"
            label="Install for"
            panels={[
              { lang: "rust", syntax: "shellscript", code: "cargo add genoxide" },
              { lang: "python", syntax: "shellscript", code: "pip install genoxide" },
            ]}
          />
          <p className="mt-2 text-base-content/50 text-xs">
            Python wheels for Linux, macOS and Windows, CPython 3.10 or later
          </p>
        </div>

        <div className="proj-rise-2 mt-8 flex flex-wrap justify-center gap-3">
          <a href="#first-example" className="btn btn-primary">
            Get started
            <ArrowRight size={16} aria-hidden />
          </a>
          <Link href={EXAMPLES_PATH} className="btn btn-ghost border-base-content/15">
            Browse examples
          </Link>
        </div>

        <ul className="proj-rise-2 mt-8 flex flex-wrap justify-center gap-2">
          {HERO_LINKS.map(({ label, href, Icon }) => (
            <li key={label}>
              <a href={href} target="_blank" rel="noopener noreferrer" className="proj-pill">
                <Icon size={14} aria-hidden />
                {label}
              </a>
            </li>
          ))}
        </ul>

        <HeroSwarm />
      </section>

      {/* ---------- Highlights ---------- */}
      <section className="proj-container py-16" aria-labelledby="highlights">
        <SectionHeading id="highlights" eyebrow="What's in it" title="One library, the whole toolbox">
          Genomes, algorithms and an engine that runs them, from Rust or from Python.
        </SectionHeading>
        <ul className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {GENOXIDE_HIGHLIGHTS.map((h) => {
            const Icon = HIGHLIGHT_ICONS[h.icon] ?? Sparkles;
            return (
              <li key={h.title} className="proj-card p-5">
                <span className="proj-icon-tile">
                  <Icon size={19} aria-hidden />
                </span>
                <h3 className="mt-4 font-semibold tracking-tight">{h.title}</h3>
                <p className="proj-lead mt-1.5 text-sm">{h.body}</p>
              </li>
            );
          })}
        </ul>

        <dl className="mt-6 grid gap-px overflow-hidden rounded-2xl border border-base-content/10 bg-base-content/10 sm:grid-cols-2 lg:grid-cols-4">
          {GENOXIDE_PROPERTIES.map((p) => (
            <div key={p.title} className="bg-base-100 p-5">
              <dt className="font-semibold text-sm">{p.title}</dt>
              <dd className="proj-lead mt-1 text-sm">{p.body}</dd>
            </div>
          ))}
        </dl>
      </section>

      {/* ---------- First example ---------- */}
      <section className="proj-container py-16" aria-labelledby="first-example">
        <div className="grid items-start gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          <div className="lg:sticky lg:top-32">
            <SectionHeading id="first-example" eyebrow="First example" title="A first run">
              Find the 100-bit string with the most ones: a binary genome, tournament selection, uniform
              crossover and bit-flip mutation, stopped at the optimum or after 1,000 generations. The same run in
              Rust and in Python.
            </SectionHeading>
            <ul className="mt-6 space-y-3 text-sm">
              <li className="flex gap-3">
                <span className="proj-tag" data-tone="accent">
                  1
                </span>
                <span className="proj-lead">
                  Configure the algorithm: a genome, selection, crossover, mutation and a seed. Invalid settings
                  are errors before anything runs.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="proj-tag" data-tone="accent">
                  2
                </span>
                <span className="proj-lead">Give it a fitness function of the genome.</span>
              </li>
              <li className="flex gap-3">
                <span className="proj-tag" data-tone="accent">
                  3
                </span>
                <span className="proj-lead">
                  Run it until a stop condition, here the optimum or 1,000 generations.
                </span>
              </li>
            </ul>
            <div className="mt-7 flex flex-wrap gap-2">
              <a href={GENOXIDE_LINKS.docsRs} target="_blank" rel="noopener noreferrer" className="proj-pill">
                <BookOpen size={14} aria-hidden />
                Rust API
              </a>
              <a href={GENOXIDE_LINKS.pythonApi} target="_blank" rel="noopener noreferrer" className="proj-pill">
                <SiPython size={14} aria-hidden />
                Python API
              </a>
            </div>
          </div>

          <LangCodeGroup
            id="first-run"
            label="Language of the example"
            panels={[
              { lang: "rust", code: GENOXIDE_FIRST_EXAMPLE.rust, filename: "src/main.rs" },
              { lang: "python", code: GENOXIDE_FIRST_EXAMPLE.python, filename: "onemax.py" },
            ]}
          />
        </div>
      </section>

      {/* ---------- Examples ---------- */}
      <section className="proj-container py-16" aria-labelledby="examples">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <SectionHeading id="examples" eyebrow="Examples" title="Runnable examples">
            Each example is a folder in the repository with its explanation and its code, in Rust and, for most,
            in Python.
          </SectionHeading>
          <Link href={EXAMPLES_PATH} className="btn btn-sm btn-ghost border-base-content/15">
            All examples
            <ArrowRight size={15} aria-hidden />
          </Link>
        </div>
        <div className="mt-8">
          {examplesResult.ok && featured.length ? (
            <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {featured.map((e) => (
                <li key={e.slug}>
                  <ExampleCard example={e} basePath={EXAMPLES_PATH} />
                </li>
              ))}
            </ul>
          ) : (
            <SourceUnavailable href={GENOXIDE_LINKS.examplesDir} linkLabel="Examples on GitHub">
              The examples are all in the repository's examples folder.
            </SourceUnavailable>
          )}
        </div>
      </section>

      {/* ---------- Benchmarks + resources ---------- */}
      <section className="proj-container grid gap-6 py-16 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Link href={`${GENOXIDE_PATH}/benchmarks`} className="proj-card group flex flex-col p-7">
          <span className="proj-icon-tile">
            <ChartBar size={19} aria-hidden />
          </span>
          <h2 className="mt-5 font-semibold text-2xl tracking-tight">Benchmarks</h2>
          <p className="proj-lead mt-2">
            Three problems, one method each, the same in every library, under public rules.
          </p>
          <span className="mt-auto inline-flex items-center gap-1 pt-6 font-medium text-primary text-sm">
            How it's measured
            <ArrowRight size={15} aria-hidden className="transition-transform group-hover:translate-x-0.5" />
          </span>
        </Link>

        <div className="proj-card p-7">
          <h2 className="font-semibold text-2xl tracking-tight">Resources</h2>
          <ul className="mt-5 grid gap-x-8 gap-y-1 sm:grid-cols-2">
            {RESOURCES.map((r) => (
              <li key={r.label}>
                <a
                  href={r.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="-mx-2 flex flex-col rounded-lg px-2 py-2 transition-colors hover:bg-base-content/5"
                >
                  <span className="font-medium text-sm">{r.label}</span>
                  <span className="text-base-content/55 text-xs">{r.note}</span>
                </a>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </main>
  );
}
