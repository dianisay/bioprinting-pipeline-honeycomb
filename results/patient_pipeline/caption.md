# Patient-level full-pipeline figure: caption draft

**Full pipeline on a rendered patient leg.** (1) One of 40 rendered photographs of a
leg from the public Syn3DWound dataset (Syn3DWound/subject_1/Syn3DWound_0415_leg_strong_lighting_phi). (2, left) The dataset's wound mask for the
same view, outlined in yellow. (2, right) The 40 2D masks are transferred to the 3D leg
mesh by projecting every vertex into every photograph and taking a majority vote; 153
vertices are labelled as wound (red). (3-4) The mesh is resampled as a height map in a local skin frame, and
a quadratic surface fitted to the surrounding healthy ring serves as an illustrative operator-accepted
target (ring RMS residual 0.16 mDU). The colour shows how far the wound bed lies below that
target (median 0.46 mDU, maximum 0.58 mDU). (5) Coverage map. Of the
54.14 mDU² wound, 24.91 mDU² is an excluded edge margin (pink) and
29.17 mDU² is safe to print on (eligible). The selected planner
(CLIPPED_BOUNDARY) uses 29 cells (20 clipped, 9 whole) and covers 94.2% of the safe area
(teal) and 50.8% of the whole wound (at most 53.9% is
possible because of the margin); red marks safe area left uncovered. The previous planner used one fixed
lattice of whole hexagons and kept only 8 cells in a central strip
(42.4% of the safe area). (6) Each stage prints one whole layer
(0.20 mDU) in every cell that still has at least one layer of room below the lower
uncertainty envelope, then re-measures (stage 1: 29 cells, stage 2: 20 cells). The run stops with NO_SAFE_SUPPORTED_REGION:
no accepted cell still has one full layer of room in every one of its pixels. 7.70 mDU² of safe area still lies at least
one layer below the envelope, so the cavity is not filled to the target everywhere. Violations: 0
(outside wound, edge margin, unsupported, above envelope, path self-intersections). (7) The resulting scaffold
on the 3D leg and (OUTPUT) projected back onto the photograph with the wound boundary (yellow), the safe-region
boundary (cyan, dashed), the accepted honeycomb and the uncovered safe area (red). Status: COMPLETE
(coverage gate 90% of the safe area, zero violations). The target is not anatomical truth, the
geometry is mesh-rendered (aligned RGB-D route), units are dataset units with no certified physical scale,
and printing is disabled. Data: Syn3DWound, CC BY-NC-SA 4.0, doi:10.25919/5rwz-ts17.
