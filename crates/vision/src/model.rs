//! What the camera can expect to see of a board side: the centroids of the
//! visible copper islands and the drilled holes, in a view frame.
//!
//! The view frame is the board frame for the front side and the board frame
//! mirrored in x for the back, so a picture of either side maps to it with
//! a plain rotation, scale and perspective.

use pcbv_board::{outline_loops, render, Board, Finish, Point, Raster, Rect, RenderOptions, Side, Style};
use serde::{Deserialize, Serialize};

use crate::blobs::components;
use crate::image::Gray;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureKind {
    /// A connected patch of visible copper.
    Island,
    /// A drilled hole.
    Hole,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Feature {
    /// View frame, mm.
    pub x: f64,
    pub y: f64,
    pub kind: FeatureKind,
    /// mm^2.
    pub area: f64,
    /// Equivalent ellipse axes, mm.
    pub major: f64,
    pub minor: f64,
}

pub struct BoardModel {
    pub side: Side,
    pub finish: Finish,
    pub features: Vec<Feature>,
    /// View frame extent of the board.
    pub bbox: Rect,
    /// The board outline in the view frame.
    pub outline: Vec<Vec<Point>>,
    /// A picture of the side for display, with its pixel mapping.
    pub raster: Raster,
}

/// Board frame to view frame; its own inverse.
pub fn view_of(side: Side, p: Point) -> Point {
    match side {
        Side::Front => p,
        Side::Back => Point::new(-p.x, p.y),
    }
}

impl BoardModel {
    /// `px_per_mm` sets how finely the copper is sampled for island
    /// centroids and how large the display picture is.
    pub fn build(board: &Board, side: Side, finish: Finish, px_per_mm: f64) -> BoardModel {
        let mut mask_opts = RenderOptions::new(side, px_per_mm, Style::copper_mask()).with_finish(finish);
        mask_opts.layers.silk = false;
        mask_opts.layers.outline = false;
        let mask = render(board, &mask_opts);
        let gray = Gray::from_vec(mask.width(), mask.height(), mask.gray()).expect("raster size");
        let scale = px_per_mm;
        let mut features: Vec<Feature> = components(&gray, 128)
            .into_iter()
            .filter(|b| b.area >= 3.0)
            .map(|b| {
                let p = view_of(side, mask.to_board(b.x, b.y));
                Feature {
                    x: p.x,
                    y: p.y,
                    kind: FeatureKind::Island,
                    area: b.area / (scale * scale),
                    major: b.major / scale,
                    minor: b.minor / scale,
                }
            })
            .collect();
        for hole in board.holes() {
            let p = view_of(side, hole.center);
            features.push(Feature {
                x: p.x,
                y: p.y,
                kind: FeatureKind::Hole,
                area: std::f64::consts::PI * hole.width * hole.height / 4.0,
                major: hole.width.max(hole.height),
                minor: hole.width.min(hole.height),
            });
        }
        let style = match finish {
            Finish::Bare => Style::photo(),
            Finish::Masked => Style::masked_photo(),
        };
        let extent = board.bbox().unwrap_or(Rect { min: Point::ZERO, max: Point::new(10.0, 10.0) });
        let mut display = RenderOptions::new(side, px_per_mm, style).with_finish(finish);
        // Room around the board for a field of view that reaches past it.
        display.margin_mm = extent.width().max(extent.height()) * 0.3;
        let raster = render(board, &display);
        let bbox = Rect::from_points(&[view_of(side, extent.min), view_of(side, extent.max)]).unwrap();
        let outline = outline_loops(board)
            .into_iter()
            .map(|l| l.into_iter().map(|p| view_of(side, p)).collect())
            .collect();
        BoardModel { side, finish, features, bbox, outline, raster }
    }

    /// View frame to the display picture's pixels.
    pub fn view_to_raster(&self, p: Point) -> (f64, f64) {
        self.raster.to_px(view_of(self.side, p))
    }

    pub fn islands(&self) -> usize {
        self.features.iter().filter(|f| f.kind == FeatureKind::Island).count()
    }

    pub fn holes(&self) -> usize {
        self.features.iter().filter(|f| f.kind == FeatureKind::Hole).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pcbv_board::parse_board;

    const BOARD: &str = r#"(kicad_pcb (version 20240108) (generator "test")
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
  (footprint "Lib:Part" (layer "F.Cu") (at 10 10)
    (pad "1" thru_hole circle (at 0 0) (size 3 3) (drill 1.5) (layers "*.Cu" "*.Mask"))
    (pad "2" smd rect (at 5 0) (size 2 1) (layers "F.Cu" "F.Mask"))
    (pad "3" smd rect (at 5 5) (size 1 1) (layers "B.Cu" "B.Mask"))
  )
  (segment (start 2 18) (end 18 18) (width 1) (layer "B.Cu"))
  (gr_rect (start 0 0) (end 20 20) (stroke (width 0.05) (type default)) (fill no) (layer "Edge.Cuts"))
)"#;

    #[test]
    fn front_islands_and_holes() {
        let board = parse_board(BOARD).unwrap();
        let model = BoardModel::build(&board, Side::Front, Finish::Bare, 20.0);
        assert_eq!(model.islands(), 2, "{:?}", model.features);
        assert_eq!(model.holes(), 1);
        let pad2 = model
            .features
            .iter()
            .find(|f| f.kind == FeatureKind::Island && (f.x - 15.0).abs() < 0.1)
            .expect("pad 2");
        assert!((pad2.y - 10.0).abs() < 0.05);
        assert!((pad2.area - 2.0).abs() < 0.15, "{}", pad2.area);
        assert!((pad2.major - 2.0 * 1.1547).abs() < 0.2, "{}", pad2.major);
        let ring = model.features.iter().find(|f| f.kind == FeatureKind::Island && (f.x - 10.0).abs() < 0.1).unwrap();
        let ring_area = std::f64::consts::PI * (1.5f64.powi(2) - 0.75f64.powi(2));
        assert!((ring.area - ring_area).abs() < 0.3, "{} vs {ring_area}", ring.area);
        assert_eq!(model.outline.len(), 1);
    }

    #[test]
    fn back_is_mirrored() {
        let board = parse_board(BOARD).unwrap();
        let model = BoardModel::build(&board, Side::Back, Finish::Masked, 20.0);
        // Pad 3 opens the back mask; the track is covered.
        assert_eq!(model.islands(), 2, "{:?}", model.features);
        let pad3 = model.features.iter().find(|f| f.kind == FeatureKind::Island && (f.y - 15.0).abs() < 0.1).unwrap();
        assert!((pad3.x + 15.0).abs() < 0.05, "{pad3:?}");
        assert!(model.bbox.min.x < -19.0 && model.bbox.max.x < 1.0);
        let (px, _) = model.view_to_raster(Point::new(-15.0, 15.0));
        let (bx, _) = model.raster.to_px(Point::new(15.0, 15.0));
        assert!((px - bx).abs() < 1e-9);
    }
}
