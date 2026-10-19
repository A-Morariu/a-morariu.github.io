# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal website/blog (https://a-morariu.github.io) built with [Quarto](https://quarto.org) and published via GitHub Pages. Content is `.qmd` files; there is no test suite or linter.

## Commands

- `quarto preview` — live-reload dev server
- `quarto render` — build the whole site into `docs/`
- `quarto render posts/<post-dir>/index.qmd` — render a single post

Python env is managed with `uv` (`pyproject.toml`, `uv.lock`, Python 3.12, constrained to `<3.13`): `uv sync`, then run Quarto with the `.venv` active so code cells use it (JAX, distrax, tensorflow-probability, gemlib, etc.). `numpy<2` and `jax<0.7` are pinned deliberately.

## Architecture

- **`docs/` is the committed build output** and is what GitHub Pages serves (`output-dir: docs` in [_quarto.yml](_quarto.yml)). After changing content, re-render and commit `docs/` along with the sources. Don't hand-edit files in `docs/`.
- **`_freeze/` is also committed.** [posts/_metadata.yml](posts/_metadata.yml) sets `freeze: auto`, so posts with executed code cells are only re-executed when their source changes; otherwise cached results are reused. Commit updated `_freeze/` entries with new/edited posts.
- Top-level pages: [index.qmd](index.qmd), [about.qmd](about.qmd), [posts.qmd](posts.qmd) (a grid listing over `posts/`, with RSS feed), [talks.qmd](talks.qmd). Navbar and footer are defined in [_quarto.yml](_quarto.yml).
- **Posts** live in `posts/YYYY-MM-DD-slug/index.qmd` with images alongside. Folder-wide defaults (author, toc, banner title) come from `posts/_metadata.yml`. Front matter used by the listing: `title`, `description`, `date` (MM-DD-YYYY), `categories`, `image`.
- **Talks** live in `talks/<date>-<name>/`; start from `talks/index_talk_template.qmd`.
- Theming: light = `lux`, dark = `slate`, both with [styles.scss](styles.scss) (`styles.css` is also present). Font Awesome icons come from `_extensions/quarto-ext/fontawesome` (`{{< fa ... >}}` shortcode).
- `Resume_AMorariu.pdf` / `CV_AMorariu.pdf` in the root are linked from the navbar; Quarto copies them into `docs/`.
- `.backup` files (`poetry.lock.backup`, `pyroject.toml.backup`) are legacy; ignore.
