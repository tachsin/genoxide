import { GENOXIDE_TAGLINE } from "@/lib/projects/genoxide/meta";
import { OG_CONTENT_TYPE, OG_SIZE, projectsOgImage } from "@/lib/projects/og-image";

export const alt = `genoxide — ${GENOXIDE_TAGLINE}`;
export const size = OG_SIZE;
export const contentType = OG_CONTENT_TYPE;

export default function Image() {
  return projectsOgImage({
    title: "genoxide",
    description:
      "Genetic algorithms, evolution strategies, CMA-ES, differential evolution, particle swarms, local search and multi-objective optimization. Rust library, Python package.",
    chips: ["Rust", "Python", "MIT OR Apache-2.0"],
    footer: "tachsin.gr/projects/genoxide",
  });
}
