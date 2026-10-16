# Notes

## Pure Rust, no OpenCV

The eventual home of this is the Raspberry Pi that runs the spinny
machine's backend, and OpenCV on a Pi is a build and a dependency to
carry. Everything here is plain crates: `tiny-skia` draws the board,
`image` only reads and writes files, `nalgebra` solves the small
eigenproblem of the homography fit, `v4l` and `zune-jpeg` capture.
The blob detector, the constellation search and the fits are written
here, which also keeps them tunable.

The V4L2 bindings run bindgen at build time. This shell exports the ESP
toolchain's `LIBCLANG_PATH`, whose clang has no x86 backend, so bindgen
dies on a pointer width assertion. `.cargo/config.toml` forces the
system llvm-18 path for cargo; on another machine change or drop it.

## KiCad conventions, checked against pcbnew

`tools/dump_board.py` loads a board through KiCad's own `pcbnew`
module and writes pad positions, rotations, drills, tracks, vias, zone
fills and the edge extent. The parser tests compare against it, which
settled the questions a file format leaves open:

- Coordinates are millimeters, y down. A positive rotation turns
  counterclockwise as seen on screen: `x' = x cos a + y sin a`,
  `y' = -x sin a + y cos a`.
- A pad's `(at x y rot)` is in the footprint's frame before rotation,
  and its `rot` is already the absolute angle (footprint plus pad).
- A footprint flipped to the back stores its pads' local coordinates
  already mirrored; the same transform applies.
- `*.Cu` expands to the board's copper layers. pcbnew reports all 32
  possible ones for such a pad, so the test intersects with the layers
  the board has.
- An edge mount connector has front and back pads with the same number
  at the same position; pads are told apart by layers as well.
- KiCad 5 arcs are `(start center) (end first point) (angle sweep)` with
  the sweep clockwise on screen for a positive angle; the current form
  is `start`, `mid`, `end`.

tiny-skia's `Transform::from_rotate` turns clockwise on screen in a y
down frame, so KiCad angles go in negated.

## What a camera sees of copper

A finished board shows copper only where the solder mask opens: the
pads, and anything drawn or poured on the mask layer. The renderer
composites copper coverage with mask opening coverage per pixel (both
drawn white on black with anti-aliasing, their minimum is the copper's
alpha), so one model serves both a bare board and a masked one.

## Features: island and hole centroids

The model's features are the centroids of the connected regions of
visible copper in its own raster, plus every drilled hole. A photo is
segmented the same way and its blob centroids are compared, so the
appearance of copper, gold, mask or substrate never has to match a
rendering; only the geometry does. The model raster is sampled at 20 px
per mm, fine enough that the centroid of a 0402 pad is right to a few
hundredths of a millimeter.

Island centroids are stable under perspective for small pads but not
for a large pour, which is also cut by the frame edge; such blobs are
left in and simply fail to pair.

## Two ways to segment

A pad surrounded by a copper pour has no local contrast at all: the
mean of its neighborhood is copper. The local contrast look (bright
spots against the local mean, robust to light that changes across the
frame) finds gold pads on a dark mask but nothing inside a pour. The
chroma look divides red minus blue by the sum, so copper and gold keep
one value under bright or dim light, and one Otsu threshold over the
frame splits copper from everything else. Otsu is only used when the
values above a floor really fall in two groups; a frame that is gold
and nothing else above the floor would otherwise be split down the
middle of the gold. White silkscreen has no chroma and drops out of
this look, which is why it pairs fewer clutter spots than luma does.

A side that is one big pour with a few tracks has no compact islands;
its holes and vias are its constellation, found as dark spots.

## The search

Two spots and two features fix a similarity. Spot pairs come from each
spot's eight nearest neighbors, model pairs from each feature's eight
nearest, sorted by length so a spot pair's compatible model pairs are a
slice. Each hypothesis passes three gates: the model's neighbors of its
first feature must land on spots (clutter adds spots but does not take
pads away, so this gate is driven from the model side), a sample of
spots across the frame must land on features, and finally every spot is
scored. Hits are weighted by closeness (one on the feature, zero at the
tolerance), because chance hits spread over the tolerance disc and
average half, which separates a real match from clutter better than a
count. The tolerance grows with distance from the hypothesis pair to
allow for perspective.

Early exit on a raw count let chance hypotheses end the search on dense
boards. Now a hypothesis that beats every earlier one and reaches a
modest score is refined and verified at once; the first one that passes
ends the search, and otherwise the best is refined at the end.

Verification needs two ratios. The fraction of the features the pose
puts in the frame that were found catches poses in the wrong place; but
a pose zoomed in far more than the truth expects few features in the
frame and finds them among hundreds of spots by chance, so the fraction
of the spots on the projected board that the pose explains is required
too.

## Refinement

A similarity from two neighboring spots is exact there and drifts with
distance under perspective. Pairing starts with the dozen spots nearest
the anchor and the circle grows by 1.7 each step; each step fits the
richest map its pairs determine (similarity, affine from five spread
pairs, homography from eight spread pairs) and the next step pairs with
that map. The homography is a normalized direct linear transform
reweighted twice by the projective denominator, which keeps the free
projective terms from amplifying noise at the far corners. Preferring
the simpler map unless the richer one was clearly better was tried and
dropped: a missed tilt costs millimeters at the far side of the frame,
a spurious projective term a tenth of that.

A map carried over from the last frame can be off by more than the
pitch of a pad array, which pairs every pad with its neighbor and gives a
confident, wrong fit. The spot nearest the frame center is therefore
tried against each feature within 2 mm of where the old map puts it,
and the shift that lands the most of its neighbors on features is
applied before growing.

## Synthetic pictures

The test pictures are the model's own raster seen through a camera map
with a lighting gradient, vignetting, noise and blur; outside the board
outline the table shows. The bare copper style has almost no luma
contrast against substrate, which is faithful to real boards and is why
the localizer cycles looks rather than assuming one. A single symmetric
footprint, pads only, is ambiguous by construction and makes no test
case.

## Numbers

On 16 phone photos of a masked board (4000 by 3000 and 1600 by 1200,
shrunk to 1024 wide for detection), every one locks, finding 87 to 89
percent of the expected pads on whole board shots with 0.03 to 0.08 mm
rms, in 65 to 470 ms for the search and 15 to 45 ms for a frame
afterward.
