import ProjectSubNav from "@/components/projects/ProjectSubNav";
import { GENOXIDE_LINKS } from "@/lib/projects/genoxide/meta";
import { getProject } from "@/lib/projects/registry";

const project = getProject("genoxide");

const LINKS = [
  { label: "GitHub", href: GENOXIDE_LINKS.github },
  { label: "docs.rs", href: GENOXIDE_LINKS.docsRs },
  { label: "Python API", href: GENOXIDE_LINKS.pythonApi },
];

export default function GenoxideLayout({ children }) {
  return (
    <>
      <ProjectSubNav name={project.name} href={project.href} sections={project.sections} links={LINKS} />
      {children}
    </>
  );
}
