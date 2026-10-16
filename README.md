# pcb-vision

Where on a board is the camera looking? pcb-vision reads a board's KiCad
design, watches a camera, and reports the pose of the picture on the
board as the frames come: the board point under the center of the
picture, how far the picture is turned, its scale, and the full map
from picture pixels to board millimeters. Pure Rust, no OpenCV.

| Directory | What |
| --- | --- |
| `crates/board` | `pcbv-board`: KiCad 5 to 10 `.kicad_pcb` reader, geometry, a raster of either side with a bare or a solder masked finish |
| `crates/vision` | `pcbv-vision`: spots in a picture, the search over the board's constellation, refinement to a homography, tracking, synthetic pictures |
| `crates/camera` | `pcbv-camera`: V4L2 capture, MJPEG and YUYV |
| `crates/app` | `pcbv`: the command line and the live viewer |
| `fixtures/boards` | boards with their ground truth dumped through KiCad's own `pcbnew`, for the parser tests |
| `tools` | `dump_board.py`: that ground truth |
| `docs` | [ARCHITECTURE.md](docs/ARCHITECTURE.md) |

## Quick start

Tools: a stable Rust toolchain and [mise](https://mise.jdx.dev). A V4L2
camera for the live view.

| Task | Does |
| --- | --- |
| `mise run live -- board.kicad_pcb --finish masked` | the live viewer on `/dev/video0`; `--camera`, `--size`, `--json`, `--no-window` |
| `mise run locate -- board.kicad_pcb photo.jpg --finish masked -o overlay.png --map map.png` | one picture: the pose, a marked up picture, the field of view on the board |
| `mise run render -- board.kicad_pcb -o board.png --style masked --finish masked` | draw a side |
| `mise run synth -- board.kicad_pcb -o shot.png --angle 30 --px-per-mm 25` | a camera picture made from the design, with its true pose |
| `mise run cameras` | the capture devices and their modes |
| `mise run test` | every suite |
| `mise run check` | clippy and the format check |

`--finish bare` is a board straight off a mill or laser, where all the
copper shows; `--finish masked` is a finished board, where only the
pads and whatever else the mask layer opens show. `--side back` looks at
the other side. Poses are reported in KiCad's board frame, millimeters
with x to the right and y down, for either side.

## How it works

The design is drawn as the camera would see the side, and the centroids
of its visible copper islands and of its drilled holes form a
constellation. A frame is looked at through one of three looks until one
locks: bright spots against a dark surround, copper by its chroma
against anything else, or dark holes. The spots are matched to the
constellation by trying similarities fixed by two spots and two
features, then the match is grown outward into a homography and tracked
from frame to frame. A pose counts only when it finds enough of the
features it says should be in view and explains enough of the spots on
the board.

`pcbv live --json` prints one line per frame for another program:
`state`, `center`, `angle_deg`, `mm_per_px`, `inliers`, `expected`,
`rms_mm`, `image_to_board` (a row major 3 by 3 map from full size pixels
to board millimeters) and `corners` (the board points under the four
picture corners).

## Limits

No lens model yet: a wide angle camera's distortion shows up as
residual at the picture edges. Silkscreen text is not drawn. A board
whose visible pattern is symmetric locks on either of its poses. One
board per frame.
