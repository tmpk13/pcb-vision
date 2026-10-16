# TODO

- Lens calibration: a chessboard or dot grid routine and undistortion
  before detection, so a wide angle camera's edges fit as well as its
  center.
- Integration for spinny-laser-pcb: a small HTTP or socket service
  around the localizer, or Python bindings, so the backend can ask for
  the pose; and a machine frame on top of the board frame.
- Silkscreen text in the model (a stroke font), so text can be matched
  or at least not counted as clutter.
- Report ambiguity: a board whose visible pattern is symmetric should say
  so instead of picking one pose.
- Camera controls in the viewer: exposure, focus.
- One look that uses islands and holes together.
- Check frame rate on a Raspberry Pi and trim the detector if needed.
- KiCad 5 arc direction and trapezoid pads are not covered by a pcbnew
  dump; custom pad primitives only lightly.
- Reading gerbers through kicad-cli as an alternative board source, the
  route spinny-laser-pcb takes.
