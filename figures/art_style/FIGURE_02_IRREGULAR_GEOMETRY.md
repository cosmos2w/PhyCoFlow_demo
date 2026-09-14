# Figure 2 — sparse reconstruction on irregular domains

**Source:** `Figure_IrregularGeometry.pdf`; D p. 7; LaTeX label `fig:irr_geometry`. Read the general contract first. Preserve a–h and their present reading order.

## Inventory and message

Keep a: two elasticity cases, each with truth/sensor layout and three reconstruction/residual comparisons; b: signed normalized stress-concentration-proxy error distributions; c: two airfoil cases with the existing selected comparisons; d: signed normalized pressure-drag error distributions; e: two sensor-count curves; f: two geometry-outline comparisons; g: two GMSD boxplot groups; h: two geometry-error ridgeline groups. All eight method identities in quantitative panels must remain, with each panel's existing order, and the generative prediction aggregation must not change.

The visual priority is field/boundary structure first, physical and population summaries second. The current heavy map frames and undersized method/metric labels weaken that hierarchy. The refinement should make the comparison matrix easier to read without reducing its density by removing examples.

## a and c — physical fields, sensor layouts and signed residuals

Retain the existing truth/prediction columns and the existing bottom row distinction: the truth-side lower cell shows sensor layout, whereas reconstruction-side lower cells show residuals. Do not replace the sensor layout by a zero residual image to make the matrix look uniform. Use aligned row spacing and a very light separating rule so the difference is legible.

Use a common positive-magnitude colormap such as viridis for the elasticity stress group and airfoil pressure group, applied identically to truth and all predictions. Use a light-zero diverging map for signed residuals. The displayed elasticity range is approximately 10–446, residual range ±197, pressure range 0.594–1.382 and pressure-residual range ±0.092 in the manuscript; these are cross-checks, not permission to overwrite the plot configuration. Preserve the authoritative source limits and norms exactly.

Keep physical mask interiors neutral and retain every boundary path. Reduce outer tile frames from their present heavy black treatment to approximately 0.4–0.5 pt. Use a slightly stronger boundary stroke, around 0.7–0.9 pt, applied identically to truth and predictions. The pressure jump, stress band and narrow airfoil must remain visible; never thicken an airfoil outline until it swallows the shape or neighbouring residuals.

Put all method headings on one baseline and all relative-L2 labels in an identical position. Use the full brand spelling DMF-Gen. Full FFM-Perceiver spelling is preferred where it fits; retaining FFM-Perc. is better than squeezing the type. Preserve the exact numerical values and precision. Do not use bold only on the best numerical error or introduce an improvement percentage.

The two cases should read as two repeated comparison units. Separate them with whitespace slightly larger than the within-case column gap. Use one existing shared colourbar per field/residual group, keeping its original range and tick values. Enlarge its label and move the exponent/unit text away from the endpoint tick. Do not share a colourbar across groups that had different normalizations.

## b and d — signed normalized physical-quantity errors

Keep horizontal violins, zero-reference lines, row order and every existing MAE annotation. Do not sort by MAE, clip long tails or change bandwidth. Use the global method palette with soft fills, thin outlines and legible internal summaries. The original orientation places DMF-Gen near the lower end; do not reverse it solely to place the method first.

Move the MAE annotations into a consistent right-side text gutter, aligned with each violin centre and inside the reserved panel area. Retain all digits. Let the plotted distributions and zero line remain unobstructed. Avoid enlarging DMF-Gen's violin more than other methods or suppressing baseline tails with transparency.

Use a neutral dashed zero line at modest contrast. Preserve x-axis scale, limits and tick positions; do not make the signed errors positive or turn normalized errors into percentages. Keep K_t as the named proxy and C_D as the pressure-drag coefficient. A small font or minus sign is a typography issue; renaming the physical quantity is not.

## e — sensor-count sweeps

Keep all eight curves, all original sensor counts, their x positions, and the original mean/IQR arrays. The caption specifies interquartile ranges, not confidence intervals; the shading must not be recomputed or relabelled as 95% uncertainty. Use identical ribbon alpha across methods and slightly stronger lines above the ribbons.

Place the compact shared legend above the two axes, arranged so method labels do not collide with the “Elasticity” and “Airfoil” titles. Keep the existing axis scale and ticks; do not linearize a logarithmic or nonuniform placement to create visual regularity. Retain the convergence between DMF-Gen and SiT at the largest airfoil sensor budget.

DMF-Gen may have a red line around 1.6 pt, alternatives around 1.1 pt with readable markers. Do not remove curves with near-overlap. Preserve marker coordinates and avoid new interpolation between the sensor budgets.

## f — inferred geometry outlines

Keep truth and every predicted boundary path. Give truth a dark neutral line and the predictions their model colours. Preserve each existing line-style distinction; use approximately the same readable baseline line weight rather than making less accurate geometries pale ghosts. Geometry is the result in this panel, so vector paths are strongly preferred.

Maintain physical aspect ratio. The narrow airfoil should not be vertically inflated to match the elasticity contour's visual bulk. Use the available whitespace around it, not a nonphysical aspect change. Move the legend outside both geometry windows, above them, with no line segment covering a legend label.

## g — GMSD boxplots

Keep box/whisker definitions, outlier treatment, category order and y-axis transforms. Use consistent box widths, medium-soft model fills and dark readable medians. Avoid thick black borders that dominate narrow boxes.

Increase method-label size. Wrapping or rotating the text is permitted, but keep the names attached to the same boxes and preserve all categories. A common angle is preferable to separate ad hoc rotations; include sufficient bottom gutter rather than shrinking long labels. Do not change the plot orientation or replace the boxes by points.

## h — geometry-error ridgelines

The two panels have different existing category orders. Preserve them; do not impose the global legend order. Keep the original density curves, baselines, scaling and median positions. Use muted fills and a thin visible outline, with modest red emphasis for DMF-Gen.

Keep the dashed median marks visible against the fills and aligned to their exact numerical positions. Do not normalize ridge heights differently by method or change their separation to imply a tighter distribution. Align method labels with ridge baselines, leaving enough left margin for full names.

## Cross-figure checks and gates

A02 must resolve Geo-FNO/FNO/FNO-R scientifically before the final name is propagated. Keep present plotted identities until then. Ensure the same model colour appears in e, f, g and h and in b/d even when their orders differ. The bar/line palette is categorical; it must not replace the physical colormap in a/c.

At final insertion size, the two-case matrices, MAE labels and eight-method legends should be legible without magnification. Confirm that all original sensor markers and boundary paths survive, every field/residual scale is unchanged, and no IQR band or KDE was recalculated. Deliver the comparison maps at sufficient raster resolution with boundaries and labels vector where supported.
