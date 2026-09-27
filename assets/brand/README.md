# genoxide's brand files

The design is genes + oxide: a double helix in iron oxide's colors, from deep red through rust and copper to amber, in a hexagon, a bolt head and a chemical ring. The wordmark sets "oxide" in the same gradient. It's rust-themed without the Rust logo, a trademark of the Rust Foundation.

| File | What | Where it goes |
|---|---|---|
| `logo.svg` | The mark, 128 × 128 | docs.rs (`html_logo_url` in `src/lib.rs`), the docs site's `theme.logo` |
| `logo-16.png`, `logo-32.png` | Favicons | docs.rs (`html_favicon_url`), the docs site's `theme.favicon` |
| `logo-180.png`, `logo-192.png`, `logo-512.png` | App and touch icons | Pages that want them (Apple touch icon 180, Android 192 and 512) |
| `wordmark-light.svg`, `wordmark-dark.svg` | "genoxide", for light and dark backgrounds | Wherever the name stands alone |
| `lockup-light.svg`, `lockup-dark.svg` | The mark beside the wordmark, for light and dark backgrounds | Headers, slides |
| `banner.svg` | The banner, 1280 × 320: a dark card that suits both themes | The top of README.md and python/README.md, the docs site's home page |
| `social-preview.svg`, `social-preview.png` | 1280 × 640 | GitHub's social preview (Settings → General → Social preview: upload the PNG), link previews |

The README files, docs.rs and the docs site load them from `main` on raw.githubusercontent.com (the docs site copies its three through `docs/hooks/examples.py`), so they don't go into the crate or the wheels.

## Making them

Every SVG comes from `build.py`, the PNGs from `export.mjs`; don't edit the files by hand.

```sh
pip install fonttools
python assets/brand/build.py               # the SVGs
npm install --no-save playwright-core
node assets/brand/export.mjs               # the PNGs, with the installed Chrome
```

## Colors

| Name | Hex | Use |
|---|---|---|
| Deep oxide | `#8a2a10` | The hexagon's rim, the darkest dots |
| Rust | `#b7410e` | A strand, the start of "oxide" |
| Orange | `#d9611c` | A strand |
| Copper | `#ee8a32` | The rim, "oxide", a strand |
| Amber | `#f7b955` | A strand, the end of "oxide" |
| Cream | `#fbe7c6` | The helix's rungs |
| Card | `#2b1d17` → `#161010` | The hexagon and the banners' background |
| Ink / paper | `#231a16` / `#f6ede5` | "gen" on light / dark backgrounds |

## The font

The wordmark and the banners' text are Space Grotesk (Florian Karsten), under the SIL Open Font License 1.1, converted to outlines by `build.py`: the SVGs need no font, and the font isn't redistributed here. `build.py` downloads it from Google Fonts' repository.
