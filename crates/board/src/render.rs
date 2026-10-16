//! Drawing one side of a board into a pixel map.
//!
//! The raster maps board millimeters to pixels at a fixed scale. The front
//! view keeps the board frame's orientation; the back view mirrors x, so the
//! picture is what a camera looking at that side sees.

use serde::{Deserialize, Serialize};
use tiny_skia::{
    FillRule, LineCap, LineJoin, Paint, Path, PathBuilder, Pixmap, Stroke, Transform,
};

use crate::geom::{arc_points, Point, Rect};
use crate::model::{Board, Graphic, Pad, PadShape, Primitive, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Front,
    Back,
}

impl Side {
    pub fn copper_layer(self) -> &'static str {
        match self {
            Side::Front => "F.Cu",
            Side::Back => "B.Cu",
        }
    }

    pub fn mask_layer(self) -> &'static str {
        match self {
            Side::Front => "F.Mask",
            Side::Back => "B.Mask",
        }
    }

    pub fn silk_layer(self) -> &'static str {
        match self {
            Side::Front => "F.SilkS",
            Side::Back => "B.SilkS",
        }
    }

    pub fn parse(text: &str) -> Option<Side> {
        match text.to_ascii_lowercase().as_str() {
            "front" | "f" | "top" => Some(Side::Front),
            "back" | "b" | "bottom" => Some(Side::Back),
            _ => None,
        }
    }
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Side::Front => "front",
            Side::Back => "back",
        })
    }
}

/// Colors as RGBA.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Style {
    /// Outside the board outline.
    pub background: [u8; 4],
    /// The bare board inside the outline.
    pub substrate: [u8; 4],
    pub copper: [u8; 4],
    pub hole: [u8; 4],
    pub outline: Option<[u8; 4]>,
    pub silk: Option<[u8; 4]>,
    pub anti_alias: bool,
}

impl Style {
    /// Copper white on black, holes black, no anti-aliasing: a binary mask.
    pub fn copper_mask() -> Style {
        Style {
            background: [0, 0, 0, 255],
            substrate: [0, 0, 0, 255],
            copper: [255, 255, 255, 255],
            hole: [0, 0, 0, 255],
            outline: None,
            silk: None,
            anti_alias: false,
        }
    }

    /// Holes white on black, nothing else.
    pub fn hole_mask() -> Style {
        Style {
            background: [0, 0, 0, 255],
            substrate: [0, 0, 0, 255],
            copper: [0, 0, 0, 255],
            hole: [255, 255, 255, 255],
            outline: None,
            silk: None,
            anti_alias: false,
        }
    }

    /// Roughly what a bare, unmasked board looks like: copper on tan
    /// substrate with dark holes.
    pub fn photo() -> Style {
        Style {
            background: [48, 48, 52, 255],
            substrate: [182, 158, 102, 255],
            copper: [214, 142, 78, 255],
            hole: [20, 18, 16, 255],
            outline: None,
            silk: None,
            anti_alias: true,
        }
    }

    /// A finished board: gold pads in openings of a dark solder mask, white
    /// silkscreen.
    pub fn masked_photo() -> Style {
        Style {
            background: [196, 176, 140, 255],
            substrate: [58, 36, 72, 255],
            copper: [222, 184, 96, 255],
            hole: [20, 18, 16, 255],
            outline: None,
            silk: Some([236, 234, 228, 255]),
            anti_alias: true,
        }
    }

    /// For looking at: copper, holes, outline and silkscreen in distinct
    /// colors on a dark board.
    pub fn display() -> Style {
        Style {
            background: [0, 43, 54, 255],
            substrate: [7, 54, 66, 255],
            copper: [181, 137, 0, 255],
            hole: [0, 0, 0, 255],
            outline: Some([220, 50, 47, 255]),
            silk: Some([238, 232, 213, 255]),
            anti_alias: true,
        }
    }
}

/// What covers the copper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Finish {
    /// Every bit of copper shows, as on a board straight off a mill or
    /// laser.
    Bare,
    /// Solder mask covers the copper except where the mask layer opens it:
    /// pads, and whatever is drawn or poured on the mask layer.
    Masked,
}

impl Finish {
    pub fn parse(text: &str) -> Option<Finish> {
        match text.to_ascii_lowercase().as_str() {
            "bare" => Some(Finish::Bare),
            "masked" | "mask" => Some(Finish::Masked),
            _ => None,
        }
    }
}

impl std::fmt::Display for Finish {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Finish::Bare => "bare",
            Finish::Masked => "masked",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layers {
    pub copper: bool,
    pub holes: bool,
    pub outline: bool,
    pub silk: bool,
}

impl Default for Layers {
    fn default() -> Layers {
        Layers { copper: true, holes: true, outline: true, silk: true }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderOptions {
    pub side: Side,
    pub px_per_mm: f64,
    /// Around the board extent when no region is given.
    pub margin_mm: f64,
    /// The part of the board frame to draw; the board extent plus margin by
    /// default.
    pub region: Option<Rect>,
    pub style: Style,
    pub layers: Layers,
    pub finish: Finish,
}

impl RenderOptions {
    pub fn new(side: Side, px_per_mm: f64, style: Style) -> RenderOptions {
        RenderOptions {
            side,
            px_per_mm,
            margin_mm: 2.0,
            region: None,
            style,
            layers: Layers::default(),
            finish: Finish::Bare,
        }
    }

    pub fn with_finish(mut self, finish: Finish) -> RenderOptions {
        self.finish = finish;
        self
    }
}

/// The drawn picture and how its pixels map to the board frame.
pub struct Raster {
    pub pixmap: Pixmap,
    pub side: Side,
    pub px_per_mm: f64,
    /// The board frame rectangle the picture covers.
    pub region: Rect,
}

impl Raster {
    pub fn width(&self) -> usize {
        self.pixmap.width() as usize
    }

    pub fn height(&self) -> usize {
        self.pixmap.height() as usize
    }

    /// Board frame to continuous pixel coordinates (pixel centers at +0.5).
    pub fn to_px(&self, p: Point) -> (f64, f64) {
        let x = match self.side {
            Side::Front => (p.x - self.region.min.x) * self.px_per_mm,
            Side::Back => (self.region.max.x - p.x) * self.px_per_mm,
        };
        (x, (p.y - self.region.min.y) * self.px_per_mm)
    }

    pub fn to_board(&self, x: f64, y: f64) -> Point {
        let bx = match self.side {
            Side::Front => self.region.min.x + x / self.px_per_mm,
            Side::Back => self.region.max.x - x / self.px_per_mm,
        };
        Point::new(bx, self.region.min.y + y / self.px_per_mm)
    }

    /// The transform tiny-skia applies to board frame coordinates.
    fn transform(side: Side, px_per_mm: f64, region: Rect) -> Transform {
        let s = px_per_mm as f32;
        match side {
            Side::Front => Transform::from_row(s, 0.0, 0.0, s, (-region.min.x * px_per_mm) as f32, (-region.min.y * px_per_mm) as f32),
            Side::Back => Transform::from_row(-s, 0.0, 0.0, s, (region.max.x * px_per_mm) as f32, (-region.min.y * px_per_mm) as f32),
        }
    }

    /// Luma of every pixel, row major.
    pub fn gray(&self) -> Vec<u8> {
        self.pixmap
            .data()
            .chunks_exact(4)
            .map(|px| ((px[0] as u32 * 299 + px[1] as u32 * 587 + px[2] as u32 * 114) / 1000) as u8)
            .collect()
    }

    /// Interleaved RGB of every pixel, row major.
    pub fn rgb(&self) -> Vec<u8> {
        self.pixmap.data().chunks_exact(4).flat_map(|px| [px[0], px[1], px[2]]).collect()
    }

    pub fn save_png(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        self.pixmap.save_png(path).map_err(|e| e.to_string())
    }
}

fn paint(color: [u8; 4], anti_alias: bool) -> Paint<'static> {
    let mut p = Paint::default();
    p.set_color_rgba8(color[0], color[1], color[2], color[3]);
    p.anti_alias = anti_alias;
    p
}

fn stroke(width: f64) -> Stroke {
    Stroke { width: width as f32, line_cap: LineCap::Round, line_join: LineJoin::Round, ..Stroke::default() }
}

fn polygon_path(points: &[Point], close: bool) -> Option<Path> {
    if points.len() < 2 {
        return None;
    }
    let mut pb = PathBuilder::new();
    pb.move_to(points[0].x as f32, points[0].y as f32);
    for p in &points[1..] {
        pb.line_to(p.x as f32, p.y as f32);
    }
    if close {
        pb.close();
    }
    pb.finish()
}

/// A rectangle with rounded corners, centered on the origin.
fn round_rect_path(w: f64, h: f64, radius: f64) -> Option<Path> {
    let r = radius.min(w / 2.0).min(h / 2.0).max(0.0);
    let (x0, y0, x1, y1) = (-w / 2.0, -h / 2.0, w / 2.0, h / 2.0);
    if r <= 1e-9 {
        return tiny_skia::Rect::from_ltrb(x0 as f32, y0 as f32, x1 as f32, y1 as f32).map(PathBuilder::from_rect);
    }
    // Quarter circles as cubics.
    let k = 0.552_284_75 * r;
    let mut pb = PathBuilder::new();
    let f = |v: f64| v as f32;
    pb.move_to(f(x0 + r), f(y0));
    pb.line_to(f(x1 - r), f(y0));
    pb.cubic_to(f(x1 - r + k), f(y0), f(x1), f(y0 + r - k), f(x1), f(y0 + r));
    pb.line_to(f(x1), f(y1 - r));
    pb.cubic_to(f(x1), f(y1 - r + k), f(x1 - r + k), f(y1), f(x1 - r), f(y1));
    pb.line_to(f(x0 + r), f(y1));
    pb.cubic_to(f(x0 + r - k), f(y1), f(x0), f(y1 - r + k), f(x0), f(y1 - r));
    pb.line_to(f(x0), f(y0 + r));
    pb.cubic_to(f(x0), f(y0 + r - k), f(x0 + r - k), f(y0), f(x0 + r), f(y0));
    pb.close();
    pb.finish()
}

/// Edge.Cuts drawings chained into closed loops. Pieces that never close
/// are left out.
pub fn outline_loops(board: &Board) -> Vec<Vec<Point>> {
    let tolerance = 0.02;
    let mut loops: Vec<Vec<Point>> = Vec::new();
    let mut open: Vec<Vec<Point>> = Vec::new();
    for g in board.graphics_on("Edge.Cuts") {
        match &g.shape {
            Shape::Circle { center, radius } => {
                let n = ((radius * 64.0).ceil() as usize).clamp(24, 720);
                loops.push(
                    (0..n)
                        .map(|i| {
                            let a = std::f64::consts::TAU * i as f64 / n as f64;
                            Point::new(center.x + radius * a.cos(), center.y + radius * a.sin())
                        })
                        .collect(),
                );
            }
            Shape::Poly { points } => loops.push(points.clone()),
            Shape::Line { start, end } => open.push(vec![*start, *end]),
            Shape::Arc { start, mid, end } => open.push(arc_points(*start, *mid, *end, 0.005)),
            Shape::Polyline { points } => open.push(points.clone()),
        }
    }
    while let Some(mut chain) = open.pop() {
        loop {
            let end = *chain.last().unwrap();
            if chain.len() > 2 && end.dist(chain[0]) <= tolerance {
                chain.pop();
                loops.push(chain);
                break;
            }
            let next = open.iter().position(|c| c[0].dist(end) <= tolerance || c.last().unwrap().dist(end) <= tolerance);
            match next {
                Some(i) => {
                    let mut piece = open.swap_remove(i);
                    if piece[0].dist(end) > tolerance {
                        piece.reverse();
                    }
                    chain.extend(piece.into_iter().skip(1));
                }
                None => break,
            }
        }
    }
    loops
}

struct Painter<'a> {
    pixmap: &'a mut Pixmap,
    transform: Transform,
    anti_alias: bool,
    px_per_mm: f64,
}

impl Painter<'_> {
    fn fill(&mut self, path: &Path, color: [u8; 4], rule: FillRule, local: Transform) {
        let p = paint(color, self.anti_alias);
        self.pixmap.fill_path(path, &p, rule, self.transform.pre_concat(local), None);
    }

    fn stroke(&mut self, path: &Path, color: [u8; 4], width: f64, local: Transform) {
        let p = paint(color, self.anti_alias);
        let width = width.max(1.0 / self.px_per_mm);
        self.pixmap.stroke_path(path, &p, &stroke(width), self.transform.pre_concat(local), None);
    }

    fn shape(&mut self, shape: &Shape, width: f64, filled: bool, color: [u8; 4], local: Transform) {
        match shape {
            Shape::Line { start, end } => {
                if let Some(path) = polygon_path(&[*start, *end], false) {
                    self.stroke(&path, color, width, local);
                }
            }
            Shape::Arc { start, mid, end } => {
                let pts = arc_points(*start, *mid, *end, 0.25 / self.px_per_mm);
                if let Some(path) = polygon_path(&pts, false) {
                    self.stroke(&path, color, width, local);
                }
            }
            Shape::Circle { center, radius } => {
                if let Some(path) = PathBuilder::from_circle(center.x as f32, center.y as f32, *radius as f32) {
                    if filled {
                        self.fill(&path, color, FillRule::Winding, local);
                    }
                    if width > 0.0 || !filled {
                        self.stroke(&path, color, width, local);
                    }
                }
            }
            Shape::Poly { points } => {
                if let Some(path) = polygon_path(points, true) {
                    if filled {
                        self.fill(&path, color, FillRule::Winding, local);
                    }
                    if width > 0.0 || !filled {
                        self.stroke(&path, color, width, local);
                    }
                }
            }
            Shape::Polyline { points } => {
                if let Some(path) = polygon_path(points, false) {
                    self.stroke(&path, color, width, local);
                }
            }
        }
    }

    fn graphic(&mut self, g: &Graphic, color: [u8; 4]) {
        self.shape(&g.shape, g.width, g.filled, color, Transform::identity());
    }

    /// The pad's copper, in the pad's own frame.
    fn pad_copper(&mut self, pad: &Pad, color: [u8; 4]) {
        let local = pad_transform(pad);
        let (w, h) = pad.size;
        match &pad.shape {
            PadShape::Custom { anchor, primitives } => {
                self.pad_basic(anchor, w, h, color, local);
                for Primitive { shape, width, filled } in primitives {
                    self.shape(shape, *width, *filled, color, local);
                }
            }
            shape => self.pad_basic(shape, w, h, color, local),
        }
    }

    fn pad_basic(&mut self, shape: &PadShape, w: f64, h: f64, color: [u8; 4], local: Transform) {
        let path = match shape {
            PadShape::Circle => PathBuilder::from_circle(0.0, 0.0, (w / 2.0) as f32),
            PadShape::Rect | PadShape::Trapezoid { .. } => round_rect_path(w, h, 0.0),
            PadShape::Oval => round_rect_path(w, h, w.min(h) / 2.0),
            PadShape::RoundRect { ratio } => round_rect_path(w, h, ratio * w.min(h)),
            PadShape::Custom { anchor, .. } => return self.pad_basic(anchor, w, h, color, local),
        };
        if let Some(path) = path {
            self.fill(&path, color, FillRule::Winding, local);
        }
    }

    /// The pad's drilled hole, in the pad's own frame.
    fn pad_hole(&mut self, pad: &Pad, color: [u8; 4]) {
        let Some(drill) = pad.drill else {
            return;
        };
        if drill.width <= 0.0 {
            return;
        }
        let local = pad_transform(pad).pre_concat(Transform::from_translate(drill.offset.x as f32, drill.offset.y as f32));
        if let Some(path) = round_rect_path(drill.width, drill.height, drill.width.min(drill.height) / 2.0) {
            self.fill(&path, color, FillRule::Winding, local);
        }
    }
}

/// Pad frame to board frame: KiCad's counterclockwise rotation is a
/// negative angle for tiny-skia, whose positive rotation is clockwise on
/// screen.
fn pad_transform(pad: &Pad) -> Transform {
    Transform::from_translate(pad.at.x as f32, pad.at.y as f32).pre_concat(Transform::from_rotate(-pad.rotation as f32))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CoverageLayer {
    Copper,
    MaskOpenings,
}

/// White where the layer covers, black elsewhere, with anti-aliased edges
/// when the style asks for them.
fn coverage(board: &Board, opts: &RenderOptions, width: u32, height: u32, transform: Transform, which: CoverageLayer) -> Pixmap {
    let mut pixmap = Pixmap::new(width, height).expect("raster size");
    pixmap.fill(tiny_skia::Color::BLACK);
    let white = [255, 255, 255, 255];
    let mut painter = Painter { pixmap: &mut pixmap, transform, anti_alias: opts.style.anti_alias, px_per_mm: opts.px_per_mm };
    let layer = match which {
        CoverageLayer::Copper => opts.side.copper_layer(),
        CoverageLayer::MaskOpenings => opts.side.mask_layer(),
    };
    for zone in board.zones.iter().filter(|z| z.layer == layer) {
        for outline in &zone.outlines {
            if let Some(path) = polygon_path(outline, true) {
                painter.fill(&path, white, FillRule::Winding, Transform::identity());
            }
        }
    }
    if which == CoverageLayer::Copper {
        for t in board.tracks.iter().filter(|t| t.layer == layer) {
            painter.shape(&Shape::Line { start: t.start, end: t.end }, t.width, false, white, Transform::identity());
        }
        for a in board.arcs.iter().filter(|a| a.layer == layer) {
            painter.shape(&Shape::Arc { start: a.start, mid: a.mid, end: a.end }, a.width, false, white, Transform::identity());
        }
        for via in board.vias.iter().filter(|v| v.layers.iter().any(|l| l == layer)) {
            if let Some(path) = PathBuilder::from_circle(via.at.x as f32, via.at.y as f32, (via.size / 2.0) as f32) {
                painter.fill(&path, white, FillRule::Winding, Transform::identity());
            }
        }
    }
    for g in board.graphics_on(layer) {
        painter.graphic(g, white);
    }
    for pad in board.pads().filter(|p| p.on_layer(layer)) {
        painter.pad_copper(pad, white);
    }
    pixmap
}

/// Draw one side of the board.
pub fn render(board: &Board, opts: &RenderOptions) -> Raster {
    let extent = board.bbox().unwrap_or(Rect { min: Point::ZERO, max: Point::new(10.0, 10.0) });
    let region = opts.region.unwrap_or_else(|| extent.expand(opts.margin_mm));
    let width = ((region.width() * opts.px_per_mm).ceil() as u32).clamp(1, 16384);
    let height = ((region.height() * opts.px_per_mm).ceil() as u32).clamp(1, 16384);
    let mut pixmap = Pixmap::new(width, height).expect("raster size");
    let bg = opts.style.background;
    pixmap.fill(tiny_skia::Color::from_rgba8(bg[0], bg[1], bg[2], bg[3]));
    let transform = Raster::transform(opts.side, opts.px_per_mm, region);
    let mut painter = Painter { pixmap: &mut pixmap, transform, anti_alias: opts.style.anti_alias, px_per_mm: opts.px_per_mm };
    let style = &opts.style;

    // Substrate inside the outline, the board extent when no outline closes.
    let loops = outline_loops(board);
    if loops.is_empty() {
        if let Some(path) = polygon_path(&[extent.min, Point::new(extent.max.x, extent.min.y), extent.max, Point::new(extent.min.x, extent.max.y)], true) {
            painter.fill(&path, style.substrate, FillRule::Winding, Transform::identity());
        }
    } else {
        let mut pb = PathBuilder::new();
        for l in &loops {
            if let Some(path) = polygon_path(l, true) {
                pb.push_path(&path);
            }
        }
        if let Some(path) = pb.finish() {
            painter.fill(&path, style.substrate, FillRule::EvenOdd, Transform::identity());
        }
    }

    if opts.layers.copper {
        // Copper coverage, and for a masked board the mask openings, each
        // drawn white on black into their own layer; the copper color is
        // blended in where both cover a pixel.
        let copper = coverage(board, opts, width, height, transform, CoverageLayer::Copper);
        let openings = match opts.finish {
            Finish::Bare => None,
            Finish::Masked => Some(coverage(board, opts, width, height, transform, CoverageLayer::MaskOpenings)),
        };
        let data = painter.pixmap.data_mut();
        let c = style.copper;
        for (i, px) in data.chunks_exact_mut(4).enumerate() {
            let mut a = copper.data()[i * 4] as u32;
            if let Some(o) = &openings {
                a = a.min(o.data()[i * 4] as u32);
            }
            if a == 0 {
                continue;
            }
            for k in 0..3 {
                px[k] = ((c[k] as u32 * a + px[k] as u32 * (255 - a)) / 255) as u8;
            }
            px[3] = 255;
        }
    }

    if opts.layers.holes {
        for pad in board.pads() {
            painter.pad_hole(pad, style.hole);
        }
        for via in &board.vias {
            if let Some(path) = PathBuilder::from_circle(via.at.x as f32, via.at.y as f32, (via.drill / 2.0) as f32) {
                painter.fill(&path, style.hole, FillRule::Winding, Transform::identity());
            }
        }
    }

    if opts.layers.silk {
        if let Some(color) = style.silk {
            for g in board.graphics_on(opts.side.silk_layer()) {
                painter.graphic(g, color);
            }
        }
    }

    if opts.layers.outline {
        if let Some(color) = style.outline {
            for g in board.graphics_on("Edge.Cuts") {
                painter.shape(&g.shape, g.width.max(1.5 / opts.px_per_mm), false, color, Transform::identity());
            }
        }
    }

    Raster { pixmap, side: opts.side, px_per_mm: opts.px_per_mm, region }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_board;

    const BOARD: &str = r#"(kicad_pcb (version 20240108) (generator "test")
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
  (footprint "Lib:Part" (layer "F.Cu") (at 10 10)
    (pad "1" thru_hole circle (at 0 0) (size 3 3) (drill 1.5) (layers "*.Cu" "*.Mask"))
    (pad "2" smd rect (at 5 0 90) (size 1 3) (layers "F.Cu" "F.Mask"))
  )
  (segment (start 2 18) (end 18 18) (width 1) (layer "B.Cu"))
  (gr_rect (start 0 0) (end 20 20) (stroke (width 0.05) (type default)) (fill no) (layer "Edge.Cuts"))
)"#;

    fn pixel(r: &Raster, p: Point) -> u8 {
        let (x, y) = r.to_px(p);
        r.gray()[y as usize * r.width() + x as usize]
    }

    #[test]
    fn copper_holes_and_sides() {
        let board = parse_board(BOARD).unwrap();
        let mut opts = RenderOptions::new(Side::Front, 10.0, Style::copper_mask());
        opts.margin_mm = 1.0;
        let front = render(&board, &opts);
        // 20 mm of outline, its 0.05 stroke, and 1 mm of margin each side.
        assert_eq!((front.width(), front.height()), (221, 221));
        // Pad ring is copper, its drill is not, the substrate is not.
        assert_eq!(pixel(&front, Point::new(11.2, 10.0)), 255);
        assert_eq!(pixel(&front, Point::new(10.0, 10.0)), 0);
        assert_eq!(pixel(&front, Point::new(5.0, 5.0)), 0);
        // Pad 2 is 1 wide and 3 tall, turned 90 degrees: 3 wide, 1 tall.
        assert_eq!(pixel(&front, Point::new(16.2, 10.0)), 255);
        assert_eq!(pixel(&front, Point::new(15.0, 11.0)), 0);
        // The back track does not show on the front.
        assert_eq!(pixel(&front, Point::new(10.0, 18.0)), 0);

        opts.side = Side::Back;
        let back = render(&board, &opts);
        assert_eq!(pixel(&back, Point::new(10.0, 18.0)), 255);
        assert_eq!(pixel(&back, Point::new(16.2, 10.0)), 0);
        // Mirrored: board x near the right edge lands near the left of the
        // picture, 1 mm of margin and half the outline stroke in.
        let (x, _) = back.to_px(Point::new(20.0, 0.0));
        assert!((x - 10.25).abs() < 1e-6, "{x}");
        let round = back.to_board(x, 0.0);
        assert!((round.x - 20.0).abs() < 1e-9);
    }

    #[test]
    fn masked_finish_hides_covered_copper() {
        let board = parse_board(BOARD).unwrap();
        let mut opts = RenderOptions::new(Side::Back, 10.0, Style::copper_mask()).with_finish(Finish::Masked);
        opts.margin_mm = 1.0;
        let back = render(&board, &opts);
        // The back track has no mask opening; pad 1 opens both sides.
        assert_eq!(pixel(&back, Point::new(10.0, 18.0)), 0);
        assert_eq!(pixel(&back, Point::new(11.2, 10.0)), 255);
    }

    #[test]
    fn outline_fill_from_loops() {
        let board = parse_board(BOARD).unwrap();
        assert_eq!(outline_loops(&board).len(), 1);
        let opts = RenderOptions::new(Side::Front, 10.0, Style::photo());
        let r = render(&board, &opts);
        let rgb = r.rgb();
        let at = |p: Point| {
            let (x, y) = r.to_px(p);
            let i = (y as usize * r.width() + x as usize) * 3;
            [rgb[i], rgb[i + 1], rgb[i + 2]]
        };
        assert_eq!(at(Point::new(-1.0, -1.0)), [48, 48, 52]);
        assert_eq!(at(Point::new(5.0, 5.0)), [182, 158, 102]);
    }
}
