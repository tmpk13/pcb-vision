//! Reading a `.kicad_pcb` file into a [`Board`].
//!
//! Every format from KiCad 5 (`module`, bare layer names, arcs given by a
//! center and an angle) through KiCad 10 is read by looking at the tokens
//! that are present rather than at the version number.

use std::collections::HashMap;
use std::path::Path;

use crate::geom::{arc_points, bezier_points, Point};
use crate::model::{
    Board, Drill, Footprint, Graphic, LayerDef, Pad, PadKind, PadShape, Primitive, Shape, Track,
    TrackArc, Via, ZoneFill,
};
use crate::sexpr::{Sexpr, SexprError};

#[derive(Debug, thiserror::Error)]
pub enum BoardError {
    #[error("{0}")]
    Syntax(#[from] SexprError),
    #[error("not a kicad_pcb file (top level is {0:?})")]
    NotABoard(Option<String>),
    #[error("reading {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
}

/// Chord tolerance when arcs inside polygons are flattened, mm.
const ARC_TOLERANCE: f64 = 0.005;

struct Ctx {
    copper: Vec<String>,
    nets: HashMap<i64, String>,
}

impl Board {
    pub fn load(path: impl AsRef<Path>) -> Result<Board, BoardError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| BoardError::Io {
            path: path.display().to_string(),
            source,
        })?;
        parse_board(&text)
    }
}

pub fn parse_board(text: &str) -> Result<Board, BoardError> {
    let root = Sexpr::parse(text)?;
    if !root.is("kicad_pcb") {
        return Err(BoardError::NotABoard(root.name().map(String::from)));
    }
    let mut board = Board {
        version: root.find_f64("version").unwrap_or(0.0) as i64,
        generator: root.find_str("generator").unwrap_or("").to_string(),
        ..Default::default()
    };
    if let Some(layers) = root.find("layers") {
        for l in layers.lists() {
            board.layers.push(LayerDef {
                ordinal: l.name().and_then(|s| s.parse().ok()).unwrap_or(-1),
                name: l.arg_str(0).unwrap_or("").to_string(),
                kind: l.arg_str(1).unwrap_or("").to_string(),
                user_name: l.arg_str(2).map(String::from),
            });
        }
    }
    let mut copper: Vec<String> = board
        .copper_layers()
        .iter()
        .map(|s| s.to_string())
        .collect();
    if copper.is_empty() {
        copper = vec!["F.Cu".to_string(), "B.Cu".to_string()];
    }
    let nets = root
        .find_all("net")
        .filter_map(|n| Some((n.arg_f64(0)? as i64, n.arg_str(1)?.to_string())))
        .collect();
    let ctx = Ctx { copper, nets };

    for item in root.lists() {
        match item.name() {
            Some("footprint") | Some("module") => {
                board.footprints.push(parse_footprint(item, &ctx))
            }
            Some("segment") => {
                if let Some(t) = parse_segment(item, &ctx) {
                    board.tracks.push(t);
                }
            }
            Some("arc") => {
                if let Some(a) = parse_track_arc(item, &ctx) {
                    board.arcs.push(a);
                }
            }
            Some("via") => {
                if let Some(v) = parse_via(item, &ctx) {
                    board.vias.push(v);
                }
            }
            Some("zone") => parse_zone(item, &ctx, &mut board.zones),
            Some(name) if name.starts_with("gr_") => {
                if let Some(g) = parse_graphic(item, name) {
                    board.graphics.push(g);
                }
            }
            _ => {}
        }
    }
    Ok(board)
}

/// `(at x y [rotation])`.
fn at_of(node: &Sexpr) -> (Point, f64) {
    match node.find("at") {
        Some(at) => (
            Point::new(at.arg_f64(0).unwrap_or(0.0), at.arg_f64(1).unwrap_or(0.0)),
            at.arg_f64(2).unwrap_or(0.0),
        ),
        None => (Point::ZERO, 0.0),
    }
}

fn xy_of(node: &Sexpr, name: &str) -> Option<Point> {
    node.find_xy(name).map(|(x, y)| Point::new(x, y))
}

/// `(net index ["name"])`, the name looked up when only the index is given.
fn net_name(node: Option<&Sexpr>, ctx: &Ctx) -> Option<String> {
    let node = node?;
    if let Some(name) = node.arg_str(1) {
        return Some(name.to_string());
    }
    let index = node.arg_f64(0)? as i64;
    ctx.nets.get(&index).cloned()
}

fn expand_layers<'a>(specs: impl Iterator<Item = &'a str>, copper: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for spec in specs {
        if let Some(suffix) = spec.strip_prefix("*.") {
            if suffix == "Cu" {
                out.extend(copper.iter().cloned());
            } else {
                out.push(format!("F.{suffix}"));
                out.push(format!("B.{suffix}"));
            }
        } else if let Some(suffix) = spec.strip_prefix("F&B.") {
            out.push(format!("F.{suffix}"));
            out.push(format!("B.{suffix}"));
        } else {
            out.push(spec.to_string());
        }
    }
    out
}

fn parse_footprint(node: &Sexpr, ctx: &Ctx) -> Footprint {
    let lib_id = node.arg_str(0).unwrap_or("").to_string();
    let layer = node.find_str("layer").unwrap_or("F.Cu").to_string();
    let (at, rotation) = at_of(node);
    let mut reference = String::new();
    let mut value = String::new();
    for p in node.find_all("property") {
        match p.arg_str(0) {
            Some("Reference") => reference = p.arg_str(1).unwrap_or("").to_string(),
            Some("Value") => value = p.arg_str(1).unwrap_or("").to_string(),
            _ => {}
        }
    }
    for t in node.find_all("fp_text") {
        match t.arg_str(0) {
            Some("reference") if reference.is_empty() => {
                reference = t.arg_str(1).unwrap_or("").to_string();
            }
            Some("value") if value.is_empty() => value = t.arg_str(1).unwrap_or("").to_string(),
            _ => {}
        }
    }
    let pads = node
        .find_all("pad")
        .filter_map(|p| parse_pad(p, at, rotation, ctx))
        .collect();
    let graphics = node
        .lists()
        .filter_map(|l| {
            let name = l.name()?;
            if !name.starts_with("fp_") || name.starts_with("fp_text") {
                return None;
            }
            parse_graphic(l, name)
        })
        .map(|g| Graphic {
            shape: g.shape.map(|p| at.add(p.rotate_deg(rotation))),
            ..g
        })
        .collect();
    Footprint {
        lib_id,
        reference,
        value,
        layer,
        at,
        rotation,
        pads,
        graphics,
    }
}

fn parse_pad(node: &Sexpr, fp_at: Point, fp_rotation: f64, ctx: &Ctx) -> Option<Pad> {
    let number = node.arg_str(0)?.to_string();
    let kind = match node.arg_str(1)? {
        "smd" => PadKind::Smd,
        "thru_hole" => PadKind::ThruHole,
        "np_thru_hole" => PadKind::NpThruHole,
        "connect" => PadKind::Connect,
        _ => return None,
    };
    let shape_name = node.arg_str(2)?;
    let (local, rotation) = at_of(node);
    let at = fp_at.add(local.rotate_deg(fp_rotation));
    let size = node.find_xy("size").unwrap_or((0.0, 0.0));
    let drill = node.find("drill").and_then(parse_drill);
    let layers = match node.find("layers") {
        Some(l) => expand_layers(l.atom_args(), &ctx.copper),
        None => Vec::new(),
    };
    let net = net_name(node.find("net"), ctx);
    let shape = match shape_name {
        "circle" => PadShape::Circle,
        "oval" => PadShape::Oval,
        "roundrect" => PadShape::RoundRect {
            ratio: node.find_f64("roundrect_rratio").unwrap_or(0.25),
        },
        "trapezoid" => PadShape::Trapezoid {
            delta: node.find_xy("rect_delta").unwrap_or((0.0, 0.0)),
        },
        "custom" => {
            let anchor = match node.find("options").and_then(|o| o.find_str("anchor")) {
                Some("circle") => PadShape::Circle,
                _ => PadShape::Rect,
            };
            let primitives = node
                .find("primitives")
                .map(|p| {
                    p.lists()
                        .filter_map(|l| parse_graphic(l, l.name()?))
                        .map(|g| Primitive {
                            shape: g.shape,
                            width: g.width,
                            filled: g.filled,
                        })
                        .collect()
                })
                .unwrap_or_default();
            PadShape::Custom {
                anchor: Box::new(anchor),
                primitives,
            }
        }
        _ => PadShape::Rect,
    };
    Some(Pad {
        number,
        kind,
        shape,
        at,
        rotation,
        size,
        drill,
        layers,
        net,
    })
}

/// `(drill [oval] w [h] [(offset x y)])`.
fn parse_drill(node: &Sexpr) -> Option<Drill> {
    let mut oval = false;
    let mut numbers = Vec::new();
    let mut offset = Point::ZERO;
    for arg in node.args() {
        match arg {
            Sexpr::Atom(s) if s == "oval" => oval = true,
            Sexpr::Atom(s) => {
                if let Ok(v) = s.parse::<f64>() {
                    numbers.push(v);
                }
            }
            Sexpr::List(_) if arg.is("offset") => {
                offset = Point::new(arg.arg_f64(0).unwrap_or(0.0), arg.arg_f64(1).unwrap_or(0.0));
            }
            Sexpr::List(_) => {}
        }
    }
    let width = *numbers.first()?;
    let height = if oval {
        numbers.get(1).copied().unwrap_or(width)
    } else {
        width
    };
    Some(Drill {
        width,
        height,
        offset,
    })
}

fn parse_segment(node: &Sexpr, ctx: &Ctx) -> Option<Track> {
    Some(Track {
        start: xy_of(node, "start")?,
        end: xy_of(node, "end")?,
        width: node.find_f64("width").unwrap_or(0.0),
        layer: node.find_str("layer").unwrap_or("").to_string(),
        net: net_name(node.find("net"), ctx),
    })
}

fn parse_track_arc(node: &Sexpr, ctx: &Ctx) -> Option<TrackArc> {
    Some(TrackArc {
        start: xy_of(node, "start")?,
        mid: xy_of(node, "mid")?,
        end: xy_of(node, "end")?,
        width: node.find_f64("width").unwrap_or(0.0),
        layer: node.find_str("layer").unwrap_or("").to_string(),
        net: net_name(node.find("net"), ctx),
    })
}

fn parse_via(node: &Sexpr, ctx: &Ctx) -> Option<Via> {
    let (at, _) = at_of(node);
    node.find("at")?;
    let layers = match node.find("layers") {
        Some(l) => expand_layers(l.atom_args(), &ctx.copper),
        None => ctx.copper.clone(),
    };
    Some(Via {
        at,
        size: node.find_f64("size").unwrap_or(0.0),
        drill: node.find_f64("drill").unwrap_or(0.0),
        layers,
        net: net_name(node.find("net"), ctx),
    })
}

fn parse_zone(node: &Sexpr, ctx: &Ctx, out: &mut Vec<ZoneFill>) {
    let net = net_name(node.find("net"), ctx)
        .or_else(|| node.find_str("net_name").map(String::from))
        .filter(|n| !n.is_empty());
    let zone_layers: Vec<String> = match node.find("layers") {
        Some(l) => expand_layers(l.atom_args(), &ctx.copper),
        None => node
            .find_str("layer")
            .map(|s| vec![s.to_string()])
            .unwrap_or_default(),
    };
    let mut fills: Vec<ZoneFill> = Vec::new();
    for fp in node.find_all("filled_polygon") {
        let layer = fp
            .find_str("layer")
            .map(String::from)
            .or_else(|| zone_layers.first().cloned())
            .unwrap_or_default();
        let points = parse_pts(fp.find("pts"));
        if points.len() < 3 {
            continue;
        }
        match fills.iter_mut().find(|z| z.layer == layer) {
            Some(z) => z.outlines.push(points),
            None => fills.push(ZoneFill {
                layer,
                net: net.clone(),
                outlines: vec![points],
            }),
        }
    }
    out.extend(fills);
}

/// `(pts (xy x y) ... (arc (start) (mid) (end)) ...)` flattened to points.
fn parse_pts(node: Option<&Sexpr>) -> Vec<Point> {
    let mut out = Vec::new();
    let Some(node) = node else {
        return out;
    };
    for item in node.lists() {
        match item.name() {
            Some("xy") => {
                if let (Some(x), Some(y)) = (item.arg_f64(0), item.arg_f64(1)) {
                    out.push(Point::new(x, y));
                }
            }
            Some("arc") => {
                if let (Some(s), Some(m), Some(e)) =
                    (xy_of(item, "start"), xy_of(item, "mid"), xy_of(item, "end"))
                {
                    for p in arc_points(s, m, e, ARC_TOLERANCE) {
                        if out.last().is_none_or(|last: &Point| last.dist(p) > 1e-9) {
                            out.push(p);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// A `gr_*`, `fp_*` or custom pad primitive drawing, in the frame it is
/// written in.
fn parse_graphic(node: &Sexpr, name: &str) -> Option<Graphic> {
    let base = name
        .strip_prefix("gr_")
        .or_else(|| name.strip_prefix("fp_"))
        .unwrap_or(name);
    let layer = node
        .find_str("layer")
        .or_else(|| node.find("layers").and_then(|l| l.arg_str(0)))
        .unwrap_or("")
        .to_string();
    let width = node
        .find("stroke")
        .and_then(|s| s.find_f64("width"))
        .or_else(|| node.find_f64("width"))
        .unwrap_or(0.0);
    let fill = node.find("fill").map(|f| match f.arg(0) {
        Some(Sexpr::Atom(v)) => matches!(v.as_str(), "yes" | "solid"),
        Some(Sexpr::List(_)) => f.find_str("type").is_some_and(|t| t != "none"),
        None => false,
    });
    let shape = match base {
        "line" => Shape::Line {
            start: xy_of(node, "start")?,
            end: xy_of(node, "end")?,
        },
        "arc" => {
            let start = xy_of(node, "start")?;
            let end = xy_of(node, "end")?;
            if let Some(mid) = xy_of(node, "mid") {
                Shape::Arc { start, mid, end }
            } else {
                // KiCad 5: start is the center, end the first point of the
                // arc, and the sweep runs clockwise on screen for a positive
                // angle.
                let angle = node.find_f64("angle")?;
                let center = start;
                let first = end;
                Shape::Arc {
                    start: first,
                    mid: first.rotate_about(center, -angle / 2.0),
                    end: first.rotate_about(center, -angle),
                }
            }
        }
        "circle" => {
            let center = xy_of(node, "center")?;
            let end = xy_of(node, "end")?;
            Shape::Circle {
                center,
                radius: center.dist(end),
            }
        }
        "rect" => {
            let s = xy_of(node, "start")?;
            let e = xy_of(node, "end")?;
            Shape::Poly {
                points: vec![s, Point::new(e.x, s.y), e, Point::new(s.x, e.y)],
            }
        }
        "poly" => {
            let points = parse_pts(node.find("pts"));
            if points.len() < 3 {
                return None;
            }
            Shape::Poly { points }
        }
        "curve" => {
            let pts = parse_pts(node.find("pts"));
            let points = if pts.len() == 4 {
                bezier_points(pts[0], pts[1], pts[2], pts[3], 16)
            } else {
                pts
            };
            if points.len() < 2 {
                return None;
            }
            Shape::Polyline { points }
        }
        _ => return None,
    };
    let filled = fill.unwrap_or(base == "poly");
    Some(Graphic {
        layer,
        shape,
        width,
        filled,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI: &str = r#"(kicad_pcb (version 20240108) (generator "test")
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
  (net 0 "") (net 1 "GND")
  (footprint "Lib:Part" (layer "F.Cu") (at 10 20 90)
    (property "Reference" "R1") (property "Value" "10k")
    (fp_line (start 0 1) (end 2 1) (stroke (width 0.1) (type solid)) (layer "F.SilkS"))
    (pad "1" smd rect (at 1 0 90) (size 1 2) (layers "F.Cu" "F.Mask") (net 1 "GND"))
    (pad "2" thru_hole circle (at -1 0 90) (size 1.5 1.5) (drill oval 0.6 1.2 (offset 0.1 0)) (layers "*.Cu" "*.Mask") (net 0 ""))
  )
  (segment (start 0 0) (end 5 0) (width 0.25) (layer "F.Cu") (net 1))
  (via (at 3 3) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (net 1))
  (gr_rect (start 0 0) (end 30 40) (stroke (width 0.05) (type default)) (fill no) (layer "Edge.Cuts"))
  (zone (net 1) (net_name "GND") (layers "F&B.Cu")
    (filled_polygon (layer "F.Cu") (pts (xy 0 0) (xy 1 0) (xy 1 1)))
    (filled_polygon (layer "B.Cu") (pts (xy 0 0) (xy 2 0) (arc (start 2 0) (mid 2.5 0.5) (end 2 1))))
  )
)"#;

    #[test]
    fn reads_a_small_board() {
        let board = parse_board(MINI).unwrap();
        assert_eq!(board.version, 20240108);
        assert_eq!(board.copper_layers(), vec!["F.Cu", "B.Cu"]);
        let fp = &board.footprints[0];
        assert_eq!(fp.reference, "R1");
        assert_eq!(fp.value, "10k");
        assert_eq!(fp.rotation, 90.0);
        // Local (1, 0) turned 90 degrees counterclockwise on screen lands
        // one unit up (negative y) from the footprint center.
        let p1 = &fp.pads[0];
        assert!(
            (p1.at.x - 10.0).abs() < 1e-9 && (p1.at.y - 19.0).abs() < 1e-9,
            "{:?}",
            p1.at
        );
        assert_eq!(p1.rotation, 90.0);
        assert_eq!(p1.net.as_deref(), Some("GND"));
        assert_eq!(p1.layers, vec!["F.Cu", "F.Mask"]);
        let p2 = &fp.pads[1];
        assert_eq!(p2.kind, PadKind::ThruHole);
        assert_eq!(p2.layers, vec!["F.Cu", "B.Cu", "F.Mask", "B.Mask"]);
        let drill = p2.drill.unwrap();
        assert_eq!((drill.width, drill.height), (0.6, 1.2));
        assert_eq!(drill.offset, Point::new(0.1, 0.0));
        assert_eq!(p2.net, Some(String::new()));
        let line = &fp.graphics[0];
        assert_eq!(line.layer, "F.SilkS");
        assert_eq!(line.width, 0.1);
        assert_eq!(board.tracks[0].net.as_deref(), Some("GND"));
        assert_eq!(board.tracks[0].width, 0.25);
        assert_eq!(board.vias[0].layers, vec!["F.Cu", "B.Cu"]);
        assert_eq!(board.zones.len(), 2);
        assert_eq!(board.zones[1].layer, "B.Cu");
        assert!(board.zones[1].outlines[0].len() > 5);
        let bbox = board.outline_bbox().unwrap();
        assert!((bbox.min.x + 0.025).abs() < 1e-9 && (bbox.max.y - 40.025).abs() < 1e-9);
        assert_eq!(board.holes().len(), 2);
    }

    #[test]
    fn reads_kicad5_module_form() {
        let text = r#"(kicad_pcb (version 20171130) (host pcbnew 5.1.0)
  (layers (0 F.Cu signal) (31 B.Cu signal) (44 Edge.Cuts user))
  (net 0 "") (net 1 VCC)
  (module Lib:Part (layer F.Cu) (tedit 0) (tstamp 0) (at 5 5)
    (fp_text reference C1 (at 0 0) (layer F.SilkS))
    (fp_text value 1u (at 0 0) (layer F.Fab))
    (fp_arc (start 0 0) (end 1 0) (angle 90) (layer F.SilkS) (width 0.12))
    (pad 1 smd rect (at 1 0) (size 1 1) (layers F.Cu F.Paste F.Mask) (net 1 VCC))
  )
  (gr_circle (center 0 0) (end 3 0) (layer Edge.Cuts) (width 0.1))
  (gr_poly (pts (xy 0 0) (xy 1 0) (xy 1 1)) (layer F.SilkS) (width 0.1))
)"#;
        let board = parse_board(text).unwrap();
        let fp = &board.footprints[0];
        assert_eq!(fp.reference, "C1");
        assert_eq!(fp.value, "1u");
        assert_eq!(fp.pads[0].at, Point::new(6.0, 5.0));
        assert_eq!(fp.pads[0].net.as_deref(), Some("VCC"));
        match &fp.graphics[0].shape {
            Shape::Arc { start, mid, end } => {
                assert_eq!(*start, Point::new(6.0, 5.0));
                // A positive legacy angle sweeps clockwise on screen: from
                // the right of the center down to below it.
                assert!(
                    (end.x - 5.0).abs() < 1e-9 && (end.y - 6.0).abs() < 1e-9,
                    "{end:?}"
                );
                assert!(mid.x > 5.0 && mid.y > 5.0);
            }
            other => panic!("{other:?}"),
        }
        assert!(
            matches!(board.graphics[0].shape, Shape::Circle { radius, .. } if (radius - 3.0).abs() < 1e-9)
        );
        assert!(board.graphics[1].filled);
    }

    #[test]
    fn rejects_other_files() {
        assert!(matches!(
            parse_board("(kicad_sch (version 1))"),
            Err(BoardError::NotABoard(_))
        ));
        assert!(matches!(
            parse_board("(kicad_pcb"),
            Err(BoardError::Syntax(_))
        ));
    }
}
