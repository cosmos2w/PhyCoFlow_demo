# Art-only implementation and acceptance

## Reusable Codex prompt

> Work on the existing DMF-Gen figure sources. First read `art_style/00_GENERAL_ART_STYLE.md`, `art_style/style_contract.json`, `editorial/02_AUTHOR_CHECKS.md` and the requested figure's `FIGURE_0N_*.md` file. Refine typography, semantic colour, stroke weights, spacing, legends, colourbars and export quality only. Preserve the complete panel inventory, model/case/recipe order, plotted numerical values, masks, sensor locations, zooms, contour levels, interpolation, geometry, density estimates, statistics, axis transforms/limits and colour normalization. Locate the actual source code and data; do not reconstruct scientific content from PDF pixels. Record hashes and a baseline render before editing. Keep scientific/label corrections requiring author approval in a separate change set; report unresolved items rather than inferring them. Use the existing asset basenames for final exports only after approval. Provide editable sources, PDF and PNG previews, a style-only diff, a before/after scientific-state comparison and a changelog. Test the figures at both 180-mm design width and the current manuscript's 162-mm insertion width. Do not claim completion if sources are missing, text remains below the readability floor or scientific state changed.

For one figure, append its exact asset and instruction file, for example:

> Apply the above to `Figure_MixedResolution.pdf` using `FIGURE_03_MIXED_RESOLUTION.md` and the approved V3-7 producer binding. Preserve a–e, all five panel-c columns, both zero-H sensor sweeps, all three multiscale rows and both 4×9 heatmaps. Do not resolve the selected physical variable, full-field versus zoom/local error or variance-bias units without the author's approved mapping.

## Work order

Start with source discovery and inventory rather than drawing. Then apply the global typography and model palette to a representative quantitative section and check print-scale legibility. Refine Fig. 3 and Fig. 4 first because their information density makes them the strongest tests of the system, then apply the same system to Fig. 5. Recheck these three figures together after the last edit. Do not modify Figs. 1–2 in this pass.

A style commit must be independently reviewable. If an author subsequently authorizes a scientific label correction, make that correction in a separate commit with a reference to the resolved author-check item. Do not conceal a numerical update inside a large style diff.

## Source inventory to record

Record the repository commit, generating script/native vector file, input data files and hashes, export command, figure basename, figure panel IDs, software environment and resolved font family. Record which figure components are editable vector paths and which are existing raster layers. A filename in the manuscript is not proof that the corresponding source has been found.

For numerical panels, capture arrays immediately before plotting and before any style function: field arrays; coordinates; masks; sensor locations and channel marks; zoom index bounds; contours; numerical annotations; method/recipe labels and their order. Record axis scales/limits/ticks and normalization class/parameters. For KDE/violin/box outputs, capture the already-computed density paths or summary coordinates rather than recomputing them under another plotting library. For spectra and wavelets, record the already-computed plotted components and their shell/band assignments.

Use `source_lock.template.json` as a schema prompt, not as proof that a lock exists. Replace every unrecorded value from the actual source. The template intentionally starts with `status: NOT_RECORDED`.

## Required scientific-state checks

Compare before and after hashes for unchanged numeric arrays. Also compare shapes, dtypes, NaN masks and categorical labels so a serialization difference cannot hide a changed quantity. If exact hashes cannot be used because of an existing serialization conversion, document the reason and compare elementwise with zero tolerance for scientific values unless the author explicitly approves another tolerance. Colour transforms and text layout are not numerical-data transformations.

Check contour paths/levels, mesh connectivity, polygon masks, sensor coordinates and zoom bounds. Check numerical annotation strings and their attached panel IDs. Check error-bar/ribbon endpoints, violin/KDE paths, box statistics, reference lines and frequency-band boundaries. Check that the same method or recipe remains attached to each trace or cell. Check that no artist is dropped merely because it is small or lies near a legend.

Do not treat image-difference percentages as a content-integrity test: a colormap change can affect almost every pixel. Numerical/semantic comparison must pass independently, followed by visual inspection of the redesigned rendering.

## Required visual checks

Inspect both at design width and at the current 162-mm insertion width. Ordinary body/tick/legend/numeric base text should remain at least 7 pt after scaling; panel labels must be clear and consistent. Superscripts can be smaller naturally, but scientific-notation exponents must still be visible. Font embedding and math-glyph rendering must be checked in the exported PDF, not just the plotting window.

Apply the mandatory layout-integrity gates in `00_GENERAL_ART_STYLE.md` after the final draw. Renderer-based checks must report zero unexplained text/text, text/panel and text/evidence intersections, zero clipped artists and the required horizontal and vertical clearances at both widths. Inspect full-page and high-zoom previews as a separate check. Save the measured results, intentional in-data annotation exceptions and resolved cross-figure semantic style table in `LAYOUT_QA.json`; neither visual inspection nor geometry checks may substitute for the other.

Check that all panel letters are outside data regions and every label belongs unmistakably to its panel. Check colourbar endpoints, signs and exponents for cropping. Check long model/recipe names, rotated labels and multiline legends. Inspect thin airfoil boundaries, sensor circles, residual hotspots, fine-scale wavelet components and high-frequency spectra. Check that baseline curves remain visible when printed in grayscale and under a colour-vision-deficiency simulation.

Inspect masks and empty PDF bins: no-data areas must not be confused with zero residual or low probability. Inspect colour mapping across truth/prediction groups: a common physical value should have the same colour where the original group shared a normalization. Verify that no new interpolation made low-resolution input images look smoother.

Check manuscript fit with the existing caption. A standalone figure can be well designed yet become unreadable when inserted too small. Do not solve overflow by silently shrinking the entire page to fit.

## Per-figure content count

| Figure | Non-negotiable inventory check |
|---|---|
| 1 | a–c; four channel surfaces; both cross-attention stages; local and global routes; three generation states; grid/mesh/point-cloud icons |
| 2 | a–h; both elasticity and airfoil example pairs; eight methods in applicable quantitative panels; both geometry outlines, boxplot groups and ridgeline groups |
| 3 | a–e; three resolutions; five recipes; four methods in b; two zero-H sweeps; five columns in c; three scale rows × three columns in d; two 4×9 heatmaps in e |
| 4 | a–d; eight montage columns × three variables; three 8×6 heatmaps; three LSD/spectrum pairs; three PDF/JSD pairs |
| 5 | a–f; five methods in a–c; two six-category violin axes; truth plus six spectra; eight scorecard rows × five columns |

## Deliverable structure in the figure repository

Keep the original assets untouched until review. Write proposed exports to an `art_style_review/` directory with matching basenames. Include the generating sources or source diff, `STYLE_CHANGELOG.md`, `SOURCE_LOCK.json`, `SCIENTIFIC_STATE_COMPARISON.json`, `LAYOUT_QA.json` and before/after previews. Include a concise `AUTHOR_ACTIONS.md` listing only the relevant unresolved ledger items, not a newly invented technical critique.

The completion statement should identify exactly what was checked, any blockers and the available output paths. A successful style pass can coexist with unresolved scientific labels; in that case, mark the graphics as **art reviewed, scientific release pending**, not publication-ready.
