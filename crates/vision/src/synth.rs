//! Camera pictures made from the board model, for tests and demos.

use pcbv_board::{Point, Raster};
use serde::{Deserialize, Serialize};
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Transform};

use crate::image::{Gray, Rgb};
use crate::model::BoardModel;
use crate::rng::Rng;
use crate::xform::{Homography, Similarity};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthParams {
    pub width: usize,
    pub height: usize,
    /// Box blur radius, px.
    pub blur: usize,
    /// Standard deviation of the added noise, gray levels.
    pub noise: f64,
    /// Brightness change from the left edge to the right, as a fraction.
    pub gradient: f64,
    /// Darkening at the corners, as a fraction.
    pub vignette: f64,
    pub background: [u8; 3],
    pub seed: u64,
}

impl Default for SynthParams {
    fn default() -> SynthParams {
        SynthParams { width: 1280, height: 720, blur: 1, noise: 4.0, gradient: 0.25, vignette: 0.2, background: [196, 176, 140], seed: 1 }
    }
}

/// The map from view frame millimeters to image pixels for a camera that
/// sees `center` at the middle of the frame, turned by `angle_deg`
/// (counterclockwise on screen), at `px_per_mm`, with `tilt` adding
/// perspective (fractions per pixel, small).
pub fn camera(center: Point, angle_deg: f64, px_per_mm: f64, tilt: (f64, f64), width: usize, height: usize) -> Homography {
    let s = Similarity { scale: px_per_mm, angle: -angle_deg.to_radians(), tx: 0.0, ty: 0.0 };
    let (cx, cy) = s.apply(center.x, center.y);
    let base = Similarity { tx: width as f64 / 2.0 - cx, ty: height as f64 / 2.0 - cy, ..s }.to_homography();
    // Perspective about the frame center: w = 1 + tx (x - w/2) + ty (y - h/2).
    let (hw, hh) = (width as f64 / 2.0, height as f64 / 2.0);
    let persp = Homography([
        1.0 + tilt.0 * hw,
        tilt.1 * hw,
        -tilt.0 * hw * hw - tilt.1 * hw * hh,
        tilt.0 * hh,
        1.0 + tilt.1 * hh,
        -tilt.0 * hw * hh - tilt.1 * hh * hh,
        tilt.0,
        tilt.1,
        1.0 - tilt.0 * hw - tilt.1 * hh,
    ]);
    base.then(&persp)
}

fn sample_rgb(raster: &Raster, x: f64, y: f64) -> Option<[f64; 3]> {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let w = raster.width();
    let h = raster.height();
    if fx < 0.0 || fy < 0.0 || fx > (w - 1) as f64 || fy > (h - 1) as f64 {
        return None;
    }
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = fx - x0 as f64;
    let ty = fy - y0 as f64;
    let data = raster.pixmap.data();
    let px = |x: usize, y: usize, c: usize| data[(y * w + x) * 4 + c] as f64;
    let mut out = [0.0; 3];
    for (c, v) in out.iter_mut().enumerate() {
        let top = px(x0, y0, c) * (1.0 - tx) + px(x1, y0, c) * tx;
        let bottom = px(x0, y1, c) * (1.0 - tx) + px(x1, y1, c) * tx;
        *v = top * (1.0 - ty) + bottom * ty;
    }
    Some(out)
}

/// Where the board is in the model's picture: 255 inside the outline.
fn board_mask(model: &BoardModel) -> Gray {
    let raster = &model.raster;
    let mut pixmap = Pixmap::new(raster.width() as u32, raster.height() as u32).expect("mask size");
    let mut pb = PathBuilder::new();
    for l in &model.outline {
        if l.len() < 3 {
            continue;
        }
        let (x, y) = model.view_to_raster(l[0]);
        pb.move_to(x as f32, y as f32);
        for p in &l[1..] {
            let (x, y) = model.view_to_raster(*p);
            pb.line_to(x as f32, y as f32);
        }
        pb.close();
    }
    let mut paint = Paint::default();
    paint.set_color_rgba8(255, 255, 255, 255);
    paint.anti_alias = false;
    match pb.finish() {
        Some(path) => pixmap.fill_path(&path, &paint, FillRule::EvenOdd, Transform::identity(), None),
        None => pixmap.fill(tiny_skia::Color::WHITE),
    }
    Gray { width: raster.width(), height: raster.height(), data: pixmap.data().chunks_exact(4).map(|px| px[0]).collect() }
}

/// A picture of the model's board as seen by a camera with the given view
/// frame to pixel map; the background shows wherever the outline does
/// not.
pub fn synthesize(model: &BoardModel, view_to_px: &Homography, params: &SynthParams) -> Rgb {
    let inv = view_to_px.inverse().expect("invertible camera map");
    let raster = &model.raster;
    let mask = board_mask(model);
    let mut rng = Rng::new(params.seed);
    let (w, h) = (params.width, params.height);
    let mut img = Rgb::new(w, h);
    let bg = params.background.map(|v| v as f64);
    let diag = ((w * w + h * h) as f64).sqrt() / 2.0;
    for y in 0..h {
        for x in 0..w {
            let (vx, vy) = inv.apply(x as f64 + 0.5, y as f64 + 0.5);
            let (rx, ry) = model.view_to_raster(Point::new(vx, vy));
            let inside = mask.sample(rx, ry).is_some_and(|v| v > 127.0);
            let base = if inside { sample_rgb(raster, rx, ry).unwrap_or(bg) } else { bg };
            let light = 1.0 + params.gradient * ((x as f64 / w as f64) - 0.5);
            let dx = x as f64 - w as f64 / 2.0;
            let dy = y as f64 - h as f64 / 2.0;
            let vignette = 1.0 - params.vignette * ((dx * dx + dy * dy) / (diag * diag));
            let gain = light * vignette;
            let mut px = [0u8; 3];
            for c in 0..3 {
                let v = base[c] * gain + rng.normal() * params.noise;
                px[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            img.put(x, y, px);
        }
    }
    if params.blur > 0 {
        img = box_blur(&img, params.blur);
    }
    img
}

fn box_blur(img: &Rgb, radius: usize) -> Rgb {
    let mut out = img.clone();
    let (w, h) = (img.width, img.height);
    let mut tmp = vec![0u16; w * h * 3];
    // Horizontal.
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(w);
            for c in 0..3 {
                let sum: u32 = (x0..x1).map(|xx| img.data[(y * w + xx) * 3 + c] as u32).sum();
                tmp[(y * w + x) * 3 + c] = (sum / (x1 - x0) as u32) as u16;
            }
        }
    }
    // Vertical.
    for y in 0..h {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius + 1).min(h);
        for x in 0..w {
            for c in 0..3 {
                let sum: u32 = (y0..y1).map(|yy| tmp[(yy * w + x) * 3 + c] as u32).sum();
                out.data[(y * w + x) * 3 + c] = (sum / (y1 - y0) as u32) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_map_puts_center_in_the_middle() {
        let cam = camera(Point::new(30.0, 20.0), 90.0, 10.0, (0.0, 0.0), 640, 480);
        let (x, y) = cam.apply(30.0, 20.0);
        assert!((x - 320.0).abs() < 1e-9 && (y - 240.0).abs() < 1e-9);
        // One mm along view x, with the camera turned 90 degrees
        // counterclockwise on screen, goes up the frame.
        let (x, y) = cam.apply(31.0, 20.0);
        assert!((x - 320.0).abs() < 1e-9 && (y - 230.0).abs() < 1e-9, "{x} {y}");
        let tilted = camera(Point::new(30.0, 20.0), 0.0, 10.0, (1e-4, 0.0), 640, 480);
        let (x, y) = tilted.apply(30.0, 20.0);
        assert!((x - 320.0).abs() < 1e-9 && (y - 240.0).abs() < 1e-9, "{x} {y}");
        // The map's input is the view frame, so the scale is read there.
        assert!((tilted.local_scale(30.0, 20.0) - 10.0).abs() < 1e-6);
    }
}
