# Figure 1 — measurement-to-field architecture

**Source:** `Figure_1_Model.pdf`; D p. 4; LaTeX label `fig:Overview`. Read `00_GENERAL_ART_STYLE.md` first. This is a schematic/vector refinement, not permission to reinterpret the architecture.

## Preserve the figure's scientific inventory

Keep a as the marked measurement-set and query-coordinate construction; b as sensor/global/sensor interaction, top-K local gathering and velocity fusion; c as the reference-to-field transport and grid/mesh/point-cloud evaluation. Retain all four example channel surfaces, measured/unmeasured markers, coordinates, tokenizer objects, global/sensor token arrays, cross-attention blocks, local graph, equations, source/intermediate/final surfaces and the three output-discretization icons. Preserve the number of illustrated tokens and all connection endpoints. Do not add a new field autoencoder, dense query-to-query attention or a different process diagram.

The figure should communicate three scales of reading: the measurement/field/query distinction at first glance; the sensor–global–sensor cycle and local–global conditioning on a second pass; the mathematical annotations on close reading. This is a hierarchy adjustment within the existing panels, not a new panel design.

## Current art issues

The current drawing combines thick gray arrows, dashed outer frames, gray module boxes, differently styled token containers and very small equations. The local-gathering module is visually strong, but the main measurement-reasoning loop is comparatively quiet. Long elbow paths cross large empty areas. The bottom banner competes with the generation path. These are visible in the supplied PDF and should be corrected without altering connectivity.

## a — observations and query locations

Use the same four existing surfaces and exact sensor positions. Keep each channel's existing physical-value meaning; use muted, coherent scalar maps rather than decorative unrelated saturated palettes. A channel identity label, not arbitrary recolouring alone, identifies the channel. Do not impose new numerical ranges on illustrative surfaces.

Give the channel labels, measured/unmeasured legend and measurement-set formula at least the general text floor. Align the four surfaces and their labels to a common left edge. Keep the channel legend in whitespace above/right of the surfaces rather than across them. Use small crisp circles with readable edges; retain the distinction between measured, unmeasured and sensor/query markers. Increase marker edge contrast, not sensor count or position.

Reduce the heavy gray connecting arrows to a uniform stroke with consistent triangular heads. Align the tokenizer box with the incoming measurement-set arrow. The query plane and query tokenizer should align horizontally with the existing route into the velocity evaluation. Preserve the appearance and perspective of the existing plane; do not convert it into a new geometry example.

Replace the visually dominant dashed outside frame with a very light border or whitespace separation. Preserve the grouping and panel letter. Do not remove informative border segments if they are part of a geometric object rather than a decorative container.

## b — reasoning and conditioning

Use global blue for the global latent arrays, red-tinted sensor anchoring, a restrained warm accent for local gathering and teal for the fusion/output module. Keep these as light fills and medium-dark outlines. Avoid saturated fills behind formulas. All token boxes should share corner radius, outline width and internal cell spacing; array shapes and counts remain unchanged.

Use one visual language for both cross-attention blocks and both tokenizers: flat vector rectangles, no shadows, approximately 0.6-pt outlines and very pale fill. Leave Q and K/V tags readable and adjacent to the corresponding edge. The approved architecture determines their meaning; see A01 below.

Give the sensor/global/sensor route a stronger but restrained stroke than ancillary readouts. Retain every arrow's source, destination and direction. Existing elbow paths can be rerouted to clear labels, but no endpoint or branch may change. Use orthogonal segments and predictable bend radii; leave an arrowhead only at the destination unless the original scientific graphic explicitly indicates another direction.

Inside top-K gathering, enlarge the formula text and align the local graph, neighbourhood expression and weighted-sum equation. Retain all nodes, edge endpoints and sensor coordinates. Use thin neutral dashed neighbour edges, a small contrast outline for the query and warm edges for the selected sensor anchors. Do not add/drop neighbours, change K or illustrate a smoother neighbourhood.

The global-readout strip must remain attached to the same latent source and fusion destination. Align local and global inputs to the fusion box. The fusion output and velocity label should occupy one clear downward route into c. Do not make an optional path visually compulsory or vice versa by changing dashed/solid status without A01 approval.

## c — physical-field transport

Preserve the exact source, intermediate and endpoint surface geometry and the existing time positions. Recolour surfaces only; do not generate a new smooth reference realization or denoise an intermediate one. Use equal visual treatment of the three transport states and the same arrowhead scale as b.

Place flow-time labels consistently above the three states and align the integration formula beneath the transition. Keep the generation path visually continuous from left to right. The three discretization icons should share line weight and a baseline, with their labels below in a consistent font. Preserve their grid, mesh and point-cloud structures.

Reduce the bottom text band's background contrast so it does not look like a separate headline panel. Keep its text and mathematical content, wrapping only as necessary. Keep any cross-panel connector clear of this statement. Do not draw arrows that imply the mesh icons are separate independently sampled functions.

## Scientific release gates, outside the style pass

A01 records apparent Q/K/V inconsistencies with the text, the “Measured field value” annotation for the evolving state, the simplified RBF formula, mixed notation and “Optional global readout.” Preserve these in the style-only commit and request an approved text/connectivity patch. The literal “on a either” typo may be corrected separately as unambiguous copy editing. Aesthetic work on the reference surface is not permission to change the sampled process being represented.

## Acceptance

At 162-mm insertion width, all ordinary text and equation bases should meet the 7-pt minimum, the sensor/global/sensor route should be traceable without crossing labels, every original arrow should have the same semantic endpoints, and all four channel surfaces plus all three discretization icons should remain present. Supply an SVG or native editable vector source in addition to PDF when supported. Record semantic issues separately; a cleaner diagram is not an architecture verification.
