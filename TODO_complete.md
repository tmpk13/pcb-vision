# Completed

- Cargo workspace of four crates, dev profile tuned for realtime,
  LIBCLANG_PATH pinned for bindgen under the ESP shell.
- KiCad 5 to 10 board reader: footprints and pads with absolute
  positions and rotations, drills and slots, tracks, arcs, vias, zone
  fills, board and footprint drawings, nets, layer wildcards.
- pcbnew ground truth dumper and parser tests against three boards plus
  the Forest Sensor Platform when present.
- Rasterizer: copper, holes, outline, silkscreen, front and back, bare
  and masked finish through coverage compositing, four styles.
- Vision foundation: image containers, channels (luma, warmth, chroma),
  integral images, blob detection with sub-pixel centroids and shape
  moments, local and global segmentation with a floored Otsu threshold.
- Transforms: similarity, affine, weighted DLT homography, Jacobians.
- Board model features (islands, holes) in a view frame; constellation
  search with model-driven gating, weighted scoring, in-search
  verification; region-growing refinement with re-anchoring; coverage
  and explained acceptance; tracking state machine.
- Synthetic camera pictures from the model and end to end tests on the
  fixture boards; all 16 reference photos locate.
- V4L2 capture crate; `pcbv` with board, render, synth, locate, cameras
  and live (egui window or JSON lines, a camera or replayed pictures,
  window screenshots); overlays and board map pictures; coasting state.
- Docs, mise tasks, notes.
