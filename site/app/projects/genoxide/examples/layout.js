import ExamplesSidebar from "@/components/projects/genoxide/ExamplesSidebar";
import { EXAMPLES_PATH, exampleTree, getExamples } from "@/lib/projects/genoxide/examples";

/**
 * The examples' index and pages, with the sidebar of every example: a column on the left on wide
 * screens, a drawer below. The layout stays across the pages, so the sidebar keeps its scroll and
 * open families. From xl up it widens the page by the sidebar's width, so the content keeps the
 * width of the other genoxide pages where the screen allows (from about 1500px).
 *
 * Without the list (GitHub unreachable) the pages show a link to the repository instead, alone.
 */
export default async function ExamplesLayout({ children }) {
  const { ok, examples } = await getExamples();
  if (!ok || !examples.length) return children;
  return (
    <div className="xl:mx-auto xl:grid xl:max-w-[94rem] xl:grid-cols-[14.5rem_minmax(0,1fr)] xl:gap-x-2 xl:pl-8">
      <ExamplesSidebar tree={exampleTree(examples)} basePath={EXAMPLES_PATH} total={examples.length} />
      <div className="min-w-0">{children}</div>
    </div>
  );
}
