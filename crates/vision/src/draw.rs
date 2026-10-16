//! Marks drawn over a picture: spots, pairs, the projected board.

use pcbv_board::Point;
use tiny_skia::{LineCap, Paint, PathBuilder, Pixmap, Stroke, Transform};

use crate::image::Rgb;
use crate::localizer::Observation;
use crate::model::BoardModel;

/// Solarized accent colors.
pub const YELLOW: [u8; 4] = [181, 137, 0, 255];
pub const ORANGE: [u8; 4] = [203, 75, 22, 255];
pub const RED: [u8; 4] = [220, 50, 47, 255];
pub const MAGENTA: [u8; 4] = [211, 54, 130, 255];
pub const VIOLET: [u8; 4] = [108, 113, 196, 255];
pub const BLUE: [u8; 4] = [38, 139, 210, 255];
pub const CYAN: [u8; 4] = [42, 161, 152, 255];
pub const GREEN: [u8; 4] = [133, 153, 0, 255];

pub struct Canvas {
    pixmap: Pixmap,
}

impl Canvas {
    pub fn from_rgb(img: &Rgb) -> Canvas {
        let mut pixmap = Pixmap::new(img.width as u32, img.height as u32).expect("canvas size");
        for (dst, src) in pixmap.data_mut().chunks_exact_mut(4).zip(img.data.chunks_exact(3)) {
            dst[..3].copy_from_slice(src);
            dst[3] = 255;
        }
        Canvas { pixmap }
    }

    pub fn into_rgb(self) -> Rgb {
        let data = self.pixmap.data().chunks_exact(4).flat_map(|px| [px[0], px[1], px[2]]).collect();
        Rgb { width: self.pixmap.width() as usize, height: self.pixmap.height() as usize, data }
    }

    fn paint(color: [u8; 4]) -> Paint<'static> {
        let mut p = Paint::default();
        p.set_color_rgba8(color[0], color[1], color[2], color[3]);
        p.anti_alias = true;
        p
    }

    pub fn circle(&mut self, x: f64, y: f64, radius: f64, color: [u8; 4], width: f64) {
        if let Some(path) = PathBuilder::from_circle(x as f32, y as f32, radius as f32) {
            let stroke = Stroke { width: width as f32, ..Stroke::default() };
            self.pixmap.stroke_path(&path, &Self::paint(color), &stroke, Transform::identity(), None);
        }
    }

    pub fn line(&mut self, a: (f64, f64), b: (f64, f64), color: [u8; 4], width: f64) {
        let mut pb = PathBuilder::new();
        pb.move_to(a.0 as f32, a.1 as f32);
        pb.line_to(b.0 as f32, b.1 as f32);
        if let Some(path) = pb.finish() {
            let stroke = Stroke { width: width as f32, line_cap: LineCap::Round, ..Stroke::default() };
            self.pixmap.stroke_path(&path, &Self::paint(color), &stroke, Transform::identity(), None);
        }
    }

    pub fn polyline(&mut self, points: &[(f64, f64)], close: bool, color: [u8; 4], width: f64) {
        if points.len() < 2 {
            return;
        }
        let mut pb = PathBuilder::new();
        pb.move_to(points[0].0 as f32, points[0].1 as f32);
        for p in &points[1..] {
            pb.line_to(p.0 as f32, p.1 as f32);
        }
        if close {
            pb.close();
        }
        if let Some(path) = pb.finish() {
            let stroke = Stroke { width: width as f32, line_cap: LineCap::Round, ..Stroke::default() };
            self.pixmap.stroke_path(&path, &Self::paint(color), &stroke, Transform::identity(), None);
        }
    }
}

/// Spots, their pairings, the projected features and outline, and a
/// crosshair, drawn over the frame the observation came from.
pub fn overlay(frame: &Rgb, obs: &Observation, model: &BoardModel) -> Rgb {
    let mut canvas = Canvas::from_rgb(frame);
    let unit = (frame.width.max(frame.height) as f64 / 400.0).max(1.0);
    let paired: std::collections::HashSet<usize> = obs.pairs.iter().map(|p| p.0).collect();
    for (i, s) in obs.spots.iter().enumerate() {
        let r = (s.area / std::f64::consts::PI).sqrt().max(unit);
        let color = if paired.contains(&i) { GREEN } else { YELLOW };
        canvas.circle(s.x, s.y, r, color, unit * 0.6);
    }
    if let Some(pose) = &obs.pose {
        if let Some(view_to_image) = pose.image_to_view.inverse() {
            for f in &model.features {
                let (x, y) = view_to_image.apply(f.x, f.y);
                if x < -10.0 || y < -10.0 || x > frame.width as f64 + 10.0 || y > frame.height as f64 + 10.0 {
                    continue;
                }
                canvas.circle(x, y, unit * 0.8, CYAN, unit * 0.4);
            }
            for &(i, f) in &obs.pairs {
                let s = &obs.spots[i];
                let (x, y) = view_to_image.apply(model.features[f].x, model.features[f].y);
                canvas.line((s.x, s.y), (x, y), RED, unit * 0.5);
            }
            for l in &model.outline {
                let pts: Vec<(f64, f64)> = l.iter().map(|p: &Point| view_to_image.apply(p.x, p.y)).collect();
                canvas.polyline(&pts, true, MAGENTA, unit * 0.7);
            }
        }
        let (cx, cy) = (frame.width as f64 / 2.0, frame.height as f64 / 2.0);
        canvas.line((cx - unit * 8.0, cy), (cx + unit * 8.0, cy), BLUE, unit * 0.6);
        canvas.line((cx, cy - unit * 8.0), (cx, cy + unit * 8.0), BLUE, unit * 0.6);
    }
    canvas.into_rgb()
}

/// The board picture with the camera's field of view drawn on it.
pub fn board_map(model: &BoardModel, obs: &Observation) -> Rgb {
    let raster = &model.raster;
    let base = Rgb { width: raster.width(), height: raster.height(), data: raster.rgb() };
    let mut canvas = Canvas::from_rgb(&base);
    let unit = (raster.width().max(raster.height()) as f64 / 400.0).max(1.0);
    if let Some(pose) = &obs.pose {
        let pts: Vec<(f64, f64)> = pose.corners.iter().map(|c| raster.to_px(*c)).collect();
        canvas.polyline(&pts, true, GREEN, unit * 1.2);
        let (cx, cy) = raster.to_px(pose.center);
        canvas.circle(cx, cy, unit * 3.0, BLUE, unit * 0.8);
    }
    canvas.into_rgb()
}
