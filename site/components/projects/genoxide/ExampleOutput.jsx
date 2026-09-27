import LangCodeGroup from "@/components/projects/LangCodeGroup";
import { blobUrl, outputCommand } from "@/lib/projects/genoxide/examples";

/**
 * What an example prints (its output.txt), under the command that prints
 * it. The output is the same in Rust and Python; the command follows the
 * page-wide language choice, like every code group on the page.
 *
 * @param {object} props
 * @param {object} props.example  from getExample(), with `output`
 * @param {string} [props.note]  e.g. "Rust only"
 */
export default function ExampleOutput({ example, note }) {
  if (!example.output) return null;
  return (
    <LangCodeGroup
      id="example-output"
      label="Language of the command"
      note={note}
      className="[&_pre]:max-h-[32rem]"
      panels={example.languages.map((lang) => ({
        lang,
        syntax: "text",
        code: example.output,
        filename: `$ ${outputCommand(example, lang)}`,
        href: example.files.output ? blobUrl(example.files.output) : undefined,
      }))}
    />
  );
}
