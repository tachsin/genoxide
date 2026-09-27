// The PNG exports of the brand SVGs, rendered by Chrome at their exact sizes.
//
// Run from the repository's root, after `python assets/brand/build.py`:
//   npm install --no-save playwright-core
//   node assets/brand/export.mjs
// It uses the installed Chrome (playwright-core's "chrome" channel), and writes next to the SVGs.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const DIR = dirname(fileURLToPath(import.meta.url));

// [svg, png, width, height]: the social preview for GitHub (1280×640, under 1 MB), and the
// favicons and app icons; everything else uses the SVGs
const EXPORTS = [
  ["social-preview.svg", "social-preview.png", 1280, 640],
  ...[16, 32, 180, 192, 512].map((size) => ["logo.svg", `logo-${size}.png`, size, size]),
];

const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ deviceScaleFactor: 1 });
for (const [source, target, width, height] of EXPORTS) {
  const svg = readFileSync(join(DIR, source), "utf8").replace(/<svg([^>]*?) width="\d+" height="\d+"/, `<svg$1 width="${width}" height="${height}"`);
  await page.setViewportSize({ width, height });
  await page.setContent(`<!doctype html><html><body style="margin:0;background:transparent">${svg}</body></html>`);
  await page.screenshot({ path: join(DIR, target), omitBackground: true, clip: { x: 0, y: 0, width, height } });
  console.log(`${target}: ${width}×${height}`);
}
await browser.close();
