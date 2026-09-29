import Breadcrumbs from "@/components/projects/Breadcrumbs";
import ExamplesBrowser from "@/components/projects/genoxide/ExamplesBrowser";
import JsonLd from "@/components/projects/JsonLd";
import SourceUnavailable from "@/components/projects/SourceUnavailable";
import { EXAMPLES_PATH, exampleCategories, exampleGrid, getExamples } from "@/lib/projects/genoxide/examples";
import { GENOXIDE_LINKS, GENOXIDE_OG_IMAGE, GENOXIDE_PATH } from "@/lib/projects/genoxide/meta";
import { breadcrumbList } from "@/lib/projects/json-ld";
import { projectsMetadata } from "@/lib/projects/metadata";
import { absoluteUrl } from "@/lib/tools-metadata";

const DESCRIPTION =
  "Runnable genoxide examples in Rust and Python: genetic algorithms, permutations, real-valued functions, constraints, multi-objective optimization and more, each with its explanation and code.";

export const metadata = projectsMetadata({
  title: "Examples",
  fullTitle: "Examples — genoxide",
  description: DESCRIPTION,
  path: EXAMPLES_PATH,
  image: GENOXIDE_OG_IMAGE,
});

const CRUMBS = [
  { name: "Projects", path: "/projects" },
  { name: "genoxide", path: GENOXIDE_PATH },
  { name: "Examples", path: EXAMPLES_PATH },
];

export default async function ExamplesPage() {
  const { ok, examples } = await getExamples();
  // Only what the grid shows goes to the client component: a card per example, a family's problems
  // (one paper's, e.g. WFG1 to WFG9) in one card.
  const entries = exampleGrid(examples);

  const structuredData = {
    "@graph": [
      {
        "@type": "CollectionPage",
        url: absoluteUrl(EXAMPLES_PATH),
        name: "genoxide examples",
        description: DESCRIPTION,
        inLanguage: "en",
        mainEntity: {
          "@type": "ItemList",
          itemListElement: examples.map((e, i) => ({
            "@type": "ListItem",
            position: i + 1,
            url: absoluteUrl(`${EXAMPLES_PATH}/${e.slug}`),
            name: e.title,
          })),
        },
      },
      breadcrumbList(CRUMBS),
    ],
  };

  return (
    <main className="proj-container pt-10 pb-8 sm:pt-14">
      <JsonLd data={structuredData} />
      <Breadcrumbs items={CRUMBS} />

      <header className="proj-rise max-w-2xl">
        <p className="proj-eyebrow">genoxide</p>
        <h1 className="mt-2 font-semibold text-4xl tracking-tight sm:text-5xl">Examples</h1>
        <p className="proj-lead mt-4 text-lg">
          Each example runs as it is: the code here is the code in the repository. Choosing Rust or Python on one
          example selects it on every page.
        </p>
      </header>

      <div className="proj-rise-1 mt-10">
        {ok && entries.length ? (
          <ExamplesBrowser entries={entries} categories={exampleCategories(examples)} basePath={EXAMPLES_PATH} />
        ) : (
          <SourceUnavailable href={GENOXIDE_LINKS.examplesDir} linkLabel="Examples on GitHub">
            The examples are all in the repository's examples folder, each with its README and code.
          </SourceUnavailable>
        )}
      </div>
    </main>
  );
}
