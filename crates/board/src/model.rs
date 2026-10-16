//! The board as the parser leaves it: every coordinate absolute, in
//! millimeters, in KiCad's board frame (x right, y down).

use serde::{Deserialize, Serialize};

use crate::geom::{Point, Rect};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Board {
    /// The file format version token, a date like 20260206 for recent KiCad.
    pub version: i64,
    pub generator: String,
    pub layers: Vec<LayerDef>,
    pub footprints: Vec<Footprint>,
    pub tracks: Vec<Track>,
    pub arcs: Vec<TrackArc>,
    pub vias: Vec<Via>,
    pub zones: Vec<ZoneFill>,
    /// Board level drawings (gr_line and friends), any layer.
    pub graphics: Vec<Graphic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDef {
    pub ordinal: i64,
    pub name: String,
    pub kind: String,
    pub user_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Footprint {
    pub lib_id: String,
    pub reference: String,
    pub value: String,
    /// "F.Cu" or "B.Cu".
    pub layer: String,
    pub at: Point,
    /// Degrees, KiCad sense (counterclockwise on screen).
    pub rotation: f64,
    pub pads: Vec<Pad>,
    /// Footprint drawings already moved into the board frame.
    pub graphics: Vec<Graphic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PadKind {
    Smd,
    ThruHole,
    NpThruHole,
    Connect,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PadShape {
    Circle,
    Rect,
    Oval,
    RoundRect {
        ratio: f64,
    },
    Trapezoid {
        delta: (f64, f64),
    },
    /// An anchor shape plus drawings in the pad's own frame, which turns
    /// with the pad.
    Custom {
        anchor: Box<PadShape>,
        primitives: Vec<Primitive>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pad {
    pub number: String,
    pub kind: PadKind,
    pub shape: PadShape,
    /// Center in the board frame.
    pub at: Point,
    /// Degrees in the board frame, the footprint's rotation included.
    pub rotation: f64,
    pub size: (f64, f64),
    pub drill: Option<Drill>,
    /// Layer names with the wildcards expanded.
    pub layers: Vec<String>,
    pub net: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Drill {
    pub width: f64,
    /// Equal to `width` for a round hole, longer for a slot.
    pub height: f64,
    /// Hole offset from the pad center, in the pad's own frame.
    pub offset: Point,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    pub start: Point,
    pub end: Point,
    pub width: f64,
    pub layer: String,
    pub net: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackArc {
    pub start: Point,
    pub mid: Point,
    pub end: Point,
    pub width: f64,
    pub layer: String,
    pub net: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Via {
    pub at: Point,
    pub size: f64,
    pub drill: f64,
    pub layers: Vec<String>,
    pub net: Option<String>,
}

/// One layer's filled outlines of a zone. KiCad fractures holes into the
/// outline, so each outline is a simple polygon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneFill {
    pub layer: String,
    pub net: Option<String>,
    pub outlines: Vec<Vec<Point>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Shape {
    Line {
        start: Point,
        end: Point,
    },
    Arc {
        start: Point,
        mid: Point,
        end: Point,
    },
    Circle {
        center: Point,
        radius: f64,
    },
    /// A closed outline.
    Poly {
        points: Vec<Point>,
    },
    /// An open path.
    Polyline {
        points: Vec<Point>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Graphic {
    pub layer: String,
    pub shape: Shape,
    /// Stroke width; zero for an unstroked fill.
    pub width: f64,
    pub filled: bool,
}

/// A drawing inside a custom pad, relative to the pad center and unrotated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Primitive {
    pub shape: Shape,
    pub width: f64,
    pub filled: bool,
}

/// A drilled hole: a through hole pad's drill, a via, or a mounting hole.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hole {
    pub center: Point,
    pub width: f64,
    pub height: f64,
    /// Degrees, for slots.
    pub rotation: f64,
    pub plated: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BoardStats {
    pub footprints: usize,
    pub pads: usize,
    pub holes: usize,
    pub tracks: usize,
    pub arcs: usize,
    pub vias: usize,
    pub zone_outlines: usize,
    pub graphics: usize,
    pub bbox: Option<Rect>,
}

impl Shape {
    /// Every defining point of the shape.
    pub fn points(&self) -> Vec<Point> {
        match self {
            Shape::Line { start, end } => vec![*start, *end],
            Shape::Arc { start, mid, end } => vec![*start, *mid, *end],
            Shape::Circle { center, radius } => vec![
                Point::new(center.x - radius, center.y - radius),
                Point::new(center.x + radius, center.y + radius),
            ],
            Shape::Poly { points } | Shape::Polyline { points } => points.clone(),
        }
    }

    /// Move every point through `f`; the radius of a circle is kept.
    pub fn map(&self, f: impl Fn(Point) -> Point) -> Shape {
        match self {
            Shape::Line { start, end } => Shape::Line {
                start: f(*start),
                end: f(*end),
            },
            Shape::Arc { start, mid, end } => Shape::Arc {
                start: f(*start),
                mid: f(*mid),
                end: f(*end),
            },
            Shape::Circle { center, radius } => Shape::Circle {
                center: f(*center),
                radius: *radius,
            },
            Shape::Poly { points } => Shape::Poly {
                points: points.iter().map(|p| f(*p)).collect(),
            },
            Shape::Polyline { points } => Shape::Polyline {
                points: points.iter().map(|p| f(*p)).collect(),
            },
        }
    }

    pub fn bbox(&self) -> Option<Rect> {
        match self {
            Shape::Arc { start, mid, end } => {
                let pts = crate::geom::arc_points(*start, *mid, *end, 0.01);
                Rect::from_points(&pts)
            }
            _ => Rect::from_points(&self.points()),
        }
    }
}

impl Pad {
    pub fn on_layer(&self, layer: &str) -> bool {
        self.layers.iter().any(|l| l == layer)
    }

    pub fn is_copper(&self) -> bool {
        self.layers.iter().any(|l| l.ends_with(".Cu"))
    }

    pub fn hole(&self) -> Option<Hole> {
        let drill = self.drill?;
        if drill.width <= 0.0 {
            return None;
        }
        let center = self.at.add(drill.offset.rotate_deg(self.rotation));
        Some(Hole {
            center,
            width: drill.width,
            height: drill.height,
            rotation: self.rotation,
            plated: self.kind != PadKind::NpThruHole,
        })
    }
}

impl Via {
    pub fn hole(&self) -> Hole {
        Hole {
            center: self.at,
            width: self.drill,
            height: self.drill,
            rotation: 0.0,
            plated: true,
        }
    }
}

impl Board {
    pub fn copper_layers(&self) -> Vec<&str> {
        self.layers
            .iter()
            .filter(|l| l.name.ends_with(".Cu"))
            .map(|l| l.name.as_str())
            .collect()
    }

    pub fn has_layer(&self, name: &str) -> bool {
        self.layers.iter().any(|l| l.name == name)
    }

    pub fn pads(&self) -> impl Iterator<Item = &Pad> {
        self.footprints.iter().flat_map(|f| f.pads.iter())
    }

    /// Every drilled hole: pad drills and vias.
    pub fn holes(&self) -> Vec<Hole> {
        let mut holes: Vec<Hole> = self.pads().filter_map(|p| p.hole()).collect();
        holes.extend(self.vias.iter().map(|v| v.hole()));
        holes
    }

    /// Board and footprint drawings on one layer.
    pub fn graphics_on<'a>(&'a self, layer: &'a str) -> impl Iterator<Item = &'a Graphic> + 'a {
        self.graphics
            .iter()
            .chain(self.footprints.iter().flat_map(|f| f.graphics.iter()))
            .filter(move |g| g.layer == layer)
    }

    /// The extent of the Edge.Cuts drawings, stroke widths included.
    pub fn outline_bbox(&self) -> Option<Rect> {
        let mut rect: Option<Rect> = None;
        for g in self.graphics_on("Edge.Cuts") {
            if let Some(b) = g.shape.bbox() {
                let b = b.expand(g.width / 2.0);
                rect = Some(rect.map_or(b, |r| r.union(b)));
            }
        }
        rect
    }

    /// The extent of everything with copper on it.
    pub fn copper_bbox(&self) -> Option<Rect> {
        let mut rect: Option<Rect> = None;
        let mut add = |r: Option<Rect>| {
            if let Some(b) = r {
                rect = Some(rect.map_or(b, |x| x.union(b)));
            }
        };
        for pad in self.pads() {
            let half = pad.size.0.max(pad.size.1) / 2.0;
            add(Some(
                Rect {
                    min: pad.at,
                    max: pad.at,
                }
                .expand(half),
            ));
        }
        for t in &self.tracks {
            add(Rect::from_points([&t.start, &t.end]).map(|r| r.expand(t.width / 2.0)));
        }
        for a in &self.arcs {
            add(Shape::Arc {
                start: a.start,
                mid: a.mid,
                end: a.end,
            }
            .bbox()
            .map(|r| r.expand(a.width / 2.0)));
        }
        for v in &self.vias {
            add(Some(
                Rect {
                    min: v.at,
                    max: v.at,
                }
                .expand(v.size / 2.0),
            ));
        }
        for z in &self.zones {
            for o in &z.outlines {
                add(Rect::from_points(o));
            }
        }
        rect
    }

    /// The outline when the board has one, otherwise the copper extent.
    pub fn bbox(&self) -> Option<Rect> {
        self.outline_bbox().or_else(|| self.copper_bbox())
    }

    pub fn stats(&self) -> BoardStats {
        BoardStats {
            footprints: self.footprints.len(),
            pads: self.pads().count(),
            holes: self.holes().len(),
            tracks: self.tracks.len(),
            arcs: self.arcs.len(),
            vias: self.vias.len(),
            zone_outlines: self.zones.iter().map(|z| z.outlines.len()).sum(),
            graphics: self.graphics.len()
                + self
                    .footprints
                    .iter()
                    .map(|f| f.graphics.len())
                    .sum::<usize>(),
            bbox: self.bbox(),
        }
    }
}
