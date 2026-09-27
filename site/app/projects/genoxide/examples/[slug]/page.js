import { ArrowLeft, ArrowRight, BookOpen, ExternalLink, Target } from "lucide-react";
import Link from "next/link";
import { notFound } from "next/navigation";
import { SiGithub } from "react-icons/si";
import Breadcrumbs from "@/components/projects/Breadcrumbs";
import JsonLd from "@/components/projects/JsonLd";
import LangCodeGroup from "@/components/projects/LangCodeGroup";
import SourceUnavailable from "@/components/projects/SourceUnavailable";
import ExampleOutput from "@/components/projects/genoxide/ExampleOutput";
import ExamplePlayer from "@/components/projects/genoxide/player/ExamplePlayer";
import { EXAMPLES_PATH, blobUrl, getExample, getExamples, runCommand } from "@/lib/projects/genoxide/examples";
import { GENOXIDE_LINKS, GENOXIDE_OG_IMAGE, GENOXIDE_PATH } from "@/lib/projects/genoxide/meta";
import { renderMarkdown } from "@/lib/projects/highlight";
import { breadcrumbList } from "@/lib/projects/json-ld";
import { projectsMetadata } from "@/lib/projects/metadata";
import { absoluteUrl } from "@/lib/tools-metadata";

const LANGUAGE_NAMES = { rust: "Rust", python: "Python" };

function describe(example) {
  if (example.summary) return example.summary;
  const langs = example.languages.map((l) => LANGUAGE_NAMES[l]).join(" and ");
  return `The genoxide ${example.title} example${langs ? `, in ${langs}` : ""}.`;
}

export async function generateMetadata({ params }) {
  const { slug } = await params;
  const example = await getExample(slug);
  if (!example) {
    // Unknown slug, or GitHub unreachable: keep it out of the index either way.
    return { title: { absolute: "Example — genoxide" }, robots: { index: false, follow: true } };
  }
  return projectsMetadata({
    title: example.title,
    fullTitle: `${example.title} — genoxide example`,
    description: describe(example),
    path: `${EXAMPLES_PATH}/${example.slug}`,
    image: GENOXIDE_OG_IMAGE,
    type: "article",
  });
}

function fileName(path) {
  if (!path) return null;
  const parts = path.split("/");
  // examples/<name>/main.rs -> <name>/main.rs; examples/gpu/src/main.rs -> gpu/src/main.rs
  return parts.slice(1).join("/");
}

export default async function ExamplePage({ params }) {
  const { slug } = await params;
  const example = await getExample(slug);

  if (!example) {
    const { ok } = await getExamples();
    if (ok) notFound();
    // GitHub is unreachable: the example may well exist, so point to it
    // rather than answer 404.
    return (
      <main className="proj-container pt-10 pb-8 sm:pt-14">
        <SourceUnavailable href={GENOXIDE_LINKS.examplesDir} linkLabel="Examples on GitHub">
          This example is in the repository's examples folder.
        </SourceUnavailable>
      </main>
    );
  }

  const path = `${EXAMPLES_PATH}/${example.slug}`;
  const crumbs = [
    { name: "Projects", path: "/projects" },
    { name: "genoxide", path: GENOXIDE_PATH },
    { name: "Examples", path: EXAMPLES_PATH },
    { name: example.title, path },
  ];

  const bodyHtml = example.body.trim() ? await renderMarkdown(example.body, { resolveUrl: example.resolveUrl }) : "";

  const codePanels = example.languages
    .filter((lang) => example.code[lang])
    .map((lang) => ({
      lang,
      code: example.code[lang],
      filename: fileName(example.files[lang]),
      href: blobUrl(example.files[lang]),
    }));
  const rustOnly = example.languages.length === 1 && example.languages[0] === "rust";
  const note = rustOnly ? "Rust only" : undefined;

  const structuredData = {
    "@graph": [
      {
        "@type": "SoftwareSourceCode",
        name: `${example.title} — genoxide example`,
        description: describe(example),
        url: absoluteUrl(path),
        codeRepository: GENOXIDE_LINKS.github,
        codeSampleType: "full solution",
        programmingLanguage: example.languages.map((l) => ({ "@type": "ComputerLanguage", name: LANGUAGE_NAMES[l] })),
        inLanguage: "en",
        isPartOf: { "@id": `${absoluteUrl(GENOXIDE_PATH)}#source` },
        ...(example.category ? { genre: example.category } : {}),
        ...(example.referenceUrl
          ? { citation: { "@type": "CreativeWork", name: example.reference ?? example.referenceUrl, url: example.referenceUrl } }
          : {}),
      },
      breadcrumbList(crumbs),
    ],
  };

  return (
    <main className="proj-container pt-10 pb-8 sm:pt-14">
      <JsonLd data={structuredData} />
      <Breadcrumbs items={crumbs} />

      <header className="proj-rise max-w-3xl">
        <div className="flex flex-wrap items-center gap-2">
          <Link href={EXAMPLES_PATH} className="proj-tag" data-tone="accent">
            {example.category}
          </Link>
          {rustOnly ? (
            <span className="proj-tag">Rust only</span>
          ) : (
            example.languages.map((l) => (
              <span key={l} className="proj-tag">
                {LANGUAGE_NAMES[l]}
              </span>
            ))
          )}
        </div>
        <h1 className="mt-4 font-semibold text-4xl tracking-tight sm:text-5xl">{example.title}</h1>
        {example.summary ? <p className="proj-lead mt-4 text-lg">{example.summary}</p> : null}

        {example.optimum || example.reference ? (
          <dl className="proj-card proj-facts mt-6">
            {example.reference ? (
              <div className="proj-fact">
                <dt>
                  <BookOpen size={15} aria-hidden />
                  Reference
                </dt>
                <dd>
                  {example.referenceUrl ? (
                    <a
                      href={example.referenceUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline decoration-base-content/25 underline-offset-2 hover:decoration-primary"
                    >
                      {example.reference}
                      <ExternalLink size={12} aria-hidden className="ml-1 inline align-baseline opacity-60" />
                    </a>
                  ) : (
                    example.reference
                  )}
                </dd>
              </div>
            ) : null}
            {example.optimum ? (
              <div className="proj-fact">
                <dt>
                  <Target size={15} aria-hidden />
                  Optimum
                </dt>
                <dd className="font-medium">{example.optimum}</dd>
              </div>
            ) : null}
          </dl>
        ) : null}
      </header>

      {example.traceUrl ? (
        <section aria-labelledby="example-run-heading" className="proj-rise-1 mt-12">
          <div className="mb-4 max-w-3xl">
            <h2 id="example-run-heading" className="font-semibold text-2xl tracking-tight">
              The run
            </h2>
            <p className="proj-lead mt-1 text-sm">
              {example.traceNote ?? "Recorded from the seeded run below."} Play it, pause it, or scrub through
              the run.
            </p>
          </div>
          <ExamplePlayer traceUrl={example.traceUrl} sourceUrl={blobUrl(example.files.trace)} />
        </section>
      ) : null}

      {bodyHtml ? (
        <article className="proj-prose proj-rise-1 mt-10 max-w-3xl" dangerouslySetInnerHTML={{ __html: bodyHtml }} />
      ) : null}

      <section aria-labelledby="example-code-heading" className="proj-rise-2 mt-12">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <h2 id="example-code-heading" className="font-semibold text-2xl tracking-tight">
            The code
          </h2>
          <a
            href={example.folderUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-base-content/65 text-sm transition-colors hover:text-base-content"
          >
            <SiGithub size={14} aria-hidden />
            examples/{example.dir}
            <ExternalLink size={12} aria-hidden />
          </a>
        </div>
        {codePanels.length ? (
          <LangCodeGroup id="example-code" label="Language of the example" panels={codePanels} note={note} />
        ) : (
          <SourceUnavailable href={example.folderUrl} title="The code couldn't be loaded from GitHub right now" />
        )}

        {example.languages.length ? (
          <>
            <h3 className="mt-10 mb-3 font-semibold text-lg tracking-tight">Run it</h3>
            <p className="proj-lead mb-4 text-sm">
              From a clone of{" "}
              <a href={GENOXIDE_LINKS.github} className="underline decoration-base-content/25 underline-offset-2 hover:decoration-primary">
                the repository
              </a>
              :
            </p>
            <LangCodeGroup
              id="example-run"
              size="compact"
              label="Language to run"
              note={note}
              panels={example.languages.map((lang) => ({
                lang,
                syntax: "shellscript",
                code: runCommand(example, lang),
              }))}
            />
          </>
        ) : null}

        {example.output && example.languages.length ? (
          <>
            <h3 className="mt-10 mb-3 font-semibold text-lg tracking-tight">Output</h3>
            <ExampleOutput example={example} note={note} />
          </>
        ) : null}
      </section>

      {example.previous || example.next ? (
        <nav aria-label="More examples" className="mt-16 grid gap-4 sm:grid-cols-2">
          {example.previous ? (
            <Link href={`${EXAMPLES_PATH}/${example.previous.slug}`} className="proj-card group p-5">
              <span className="inline-flex items-center gap-1 text-base-content/55 text-xs">
                <ArrowLeft size={13} aria-hidden />
                Previous
              </span>
              <span className="mt-1 block font-semibold">{example.previous.title}</span>
            </Link>
          ) : (
            <span />
          )}
          {example.next ? (
            <Link href={`${EXAMPLES_PATH}/${example.next.slug}`} className="proj-card group p-5 text-right">
              <span className="inline-flex items-center gap-1 text-base-content/55 text-xs">
                Next
                <ArrowRight size={13} aria-hidden />
              </span>
              <span className="mt-1 block font-semibold">{example.next.title}</span>
            </Link>
          ) : null}
        </nav>
      ) : null}
    </main>
  );
}
