//! Compact bright or dark spots: pads, holes, vias.
//!
//! A pixel is foreground when it stands out from the mean of its
//! neighborhood by more than a threshold, which rides over lighting that
//! changes across the frame. Foreground pixels are grouped into connected
//! components whose centroid, size and shape are measured.

use serde::{Deserialize, Serialize};

use crate::image::{otsu_threshold, Gray};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Polarity {
    Bright,
    Dark,
}

impl Polarity {
    pub fn parse(text: &str) -> Option<Polarity> {
        match text.to_ascii_lowercase().as_str() {
            "bright" | "light" => Some(Polarity::Bright),
            "dark" => Some(Polarity::Dark),
            _ => None,
        }
    }
}

impl std::fmt::Display for Polarity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Polarity::Bright => "bright",
            Polarity::Dark => "dark",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlobParams {
    pub polarity: Polarity,
    /// Half width of the square the local mean is taken over, px.
    pub radius: usize,
    /// Least contrast against the local mean, 0 to 255.
    pub threshold: u8,
    pub min_area: usize,
    pub max_area: usize,
    /// Least area over bounding box area; rejects thin lines and noise.
    pub min_fill: f64,
    /// Whether blobs cut by the frame edge are kept.
    pub keep_border: bool,
    /// For the global segmentation: pixels below this are never
    /// foreground, and the threshold is sought only above it.
    pub global_floor: u8,
}

impl Default for BlobParams {
    fn default() -> BlobParams {
        BlobParams {
            polarity: Polarity::Bright,
            radius: 40,
            threshold: 40,
            min_area: 6,
            max_area: 40_000,
            min_fill: 0.3,
            keep_border: false,
            global_floor: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Blob {
    /// Contrast weighted centroid, continuous pixel coordinates.
    pub x: f64,
    pub y: f64,
    /// Pixel count.
    pub area: f64,
    /// `[x0, y0, x1, y1]`, half open.
    pub bbox: [usize; 4],
    /// Lengths of the equivalent ellipse's axes, px.
    pub major: f64,
    pub minor: f64,
    /// Direction of the major axis, radians, y down.
    pub angle: f64,
    pub touches_border: bool,
    /// Mean contrast of the pixels, 0 to 255.
    pub contrast: f64,
}

impl Blob {
    pub fn fill_ratio(&self) -> f64 {
        let w = (self.bbox[2] - self.bbox[0]) as f64;
        let h = (self.bbox[3] - self.bbox[1]) as f64;
        self.area / (w * h).max(1.0)
    }

    pub fn elongation(&self) -> f64 {
        self.major / self.minor.max(1e-9)
    }
}

/// How far each pixel stands out from its neighborhood in the given
/// direction, 0 to 255.
pub fn local_contrast(channel: &Gray, radius: usize, polarity: Polarity) -> Gray {
    let mean = channel.box_mean(radius);
    let data = channel
        .data
        .iter()
        .zip(&mean.data)
        .map(|(&v, &m)| match polarity {
            Polarity::Bright => v.saturating_sub(m),
            Polarity::Dark => m.saturating_sub(v),
        })
        .collect();
    Gray { width: channel.width, height: channel.height, data }
}

struct Stats {
    n: u64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
    w: f64,
    wx: f64,
    wy: f64,
    bbox: [usize; 4],
    border: bool,
}

fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let p = parent[i as usize];
        parent[i as usize] = parent[p as usize];
        i = p;
    }
    i
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[hi as usize] = lo;
    }
}

/// Eight-connected components of the pixels whose contrast reaches
/// `threshold`, with no filtering.
pub fn components(contrast: &Gray, threshold: u8) -> Vec<Blob> {
    let threshold = threshold.max(1);
    let w = contrast.width;
    let h = contrast.height;
    let mut labels = vec![0u32; w * h];
    let mut parent: Vec<u32> = vec![0];
    for y in 0..h {
        for x in 0..w {
            if contrast.data[y * w + x] < threshold {
                continue;
            }
            let mut best = 0u32;
            let mut neighbors = [0u32; 4];
            let mut count = 0;
            if x > 0 {
                neighbors[count] = labels[y * w + x - 1];
                count += 1;
            }
            if y > 0 {
                neighbors[count] = labels[(y - 1) * w + x];
                count += 1;
                if x > 0 {
                    neighbors[count] = labels[(y - 1) * w + x - 1];
                    count += 1;
                }
                if x + 1 < w {
                    neighbors[count] = labels[(y - 1) * w + x + 1];
                    count += 1;
                }
            }
            for &n in &neighbors[..count] {
                if n != 0 && (best == 0 || n < best) {
                    best = n;
                }
            }
            if best == 0 {
                best = parent.len() as u32;
                parent.push(best);
            } else {
                for &n in &neighbors[..count] {
                    if n != 0 && n != best {
                        union(&mut parent, best, n);
                    }
                }
            }
            labels[y * w + x] = best;
        }
    }
    let mut root_index: Vec<u32> = vec![u32::MAX; parent.len()];
    let mut stats: Vec<Stats> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let label = labels[y * w + x];
            if label == 0 {
                continue;
            }
            let root = find(&mut parent, label);
            let idx = if root_index[root as usize] == u32::MAX {
                root_index[root as usize] = stats.len() as u32;
                stats.push(Stats {
                    n: 0,
                    sx: 0.0,
                    sy: 0.0,
                    sxx: 0.0,
                    syy: 0.0,
                    sxy: 0.0,
                    w: 0.0,
                    wx: 0.0,
                    wy: 0.0,
                    bbox: [x, y, x + 1, y + 1],
                    border: false,
                });
                stats.len() - 1
            } else {
                root_index[root as usize] as usize
            };
            let s = &mut stats[idx];
            let cx = x as f64 + 0.5;
            let cy = y as f64 + 0.5;
            let weight = contrast.data[y * w + x] as f64;
            s.n += 1;
            s.sx += cx;
            s.sy += cy;
            s.sxx += cx * cx;
            s.syy += cy * cy;
            s.sxy += cx * cy;
            s.w += weight;
            s.wx += weight * cx;
            s.wy += weight * cy;
            s.bbox[0] = s.bbox[0].min(x);
            s.bbox[1] = s.bbox[1].min(y);
            s.bbox[2] = s.bbox[2].max(x + 1);
            s.bbox[3] = s.bbox[3].max(y + 1);
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                s.border = true;
            }
        }
    }
    stats
        .into_iter()
        .map(|s| {
            let n = s.n as f64;
            let mx = s.sx / n;
            let my = s.sy / n;
            let cxx = (s.sxx / n - mx * mx).max(0.0);
            let cyy = (s.syy / n - my * my).max(0.0);
            let cxy = s.sxy / n - mx * my;
            let tr = cxx + cyy;
            let det = (cxx * cyy - cxy * cxy).max(0.0);
            let disc = ((tr * tr / 4.0) - det).max(0.0).sqrt();
            let l1 = (tr / 2.0 + disc).max(0.0);
            let l2 = (tr / 2.0 - disc).max(0.0);
            // A uniform disc of radius r has variance r^2 / 4 per axis, and
            // a single pixel still has width one.
            let major = (4.0 * (l1 + 1.0 / 12.0).sqrt()).max(1.0);
            let minor = (4.0 * (l2 + 1.0 / 12.0).sqrt()).max(1.0);
            Blob {
                x: s.wx / s.w,
                y: s.wy / s.w,
                area: n,
                bbox: s.bbox,
                major,
                minor,
                angle: 0.5 * (2.0 * cxy).atan2(cxx - cyy),
                touches_border: s.border,
                contrast: s.w / n,
            }
        })
        .collect()
}

/// How foreground is told from background.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Segmentation {
    /// Against the local mean: spots that stand out from their
    /// surroundings.
    Local,
    /// Against one threshold for the whole frame, found by Otsu's method:
    /// regions of one material against another.
    Global,
}

impl Segmentation {
    pub fn parse(text: &str) -> Option<Segmentation> {
        match text.to_ascii_lowercase().as_str() {
            "local" => Some(Segmentation::Local),
            "global" | "otsu" => Some(Segmentation::Global),
            _ => None,
        }
    }
}

impl std::fmt::Display for Segmentation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Segmentation::Local => "local",
            Segmentation::Global => "global",
        })
    }
}

/// Contrast, components, and the size and shape filters.
pub fn detect(channel: &Gray, params: &BlobParams) -> Vec<Blob> {
    let contrast = local_contrast(channel, params.radius, params.polarity);
    detect_in_contrast(&contrast, params)
}

/// The threshold for the whole frame: Otsu's split of the values above
/// the floor when they fall in two groups, the floor itself when they do
/// not (one material above it) or when there is nothing above it.
pub fn global_threshold(channel: &Gray, floor: u8) -> u8 {
    let above: Vec<u8> = channel.data.iter().copied().filter(|&v| v >= floor).collect();
    if above.is_empty() || above.len() * 200 < channel.data.len() {
        return 255;
    }
    let n = above.len() as f64;
    let mean = above.iter().map(|&v| v as f64).sum::<f64>() / n;
    let variance = above.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let sub = Gray { width: above.len(), height: 1, data: above };
    let t = otsu_threshold(&sub);
    // Separability: how much of the spread the split explains.
    let (mut n0, mut s0) = (0.0, 0.0);
    for &v in &sub.data {
        if v <= t {
            n0 += 1.0;
            s0 += v as f64;
        }
    }
    let n1 = n - n0;
    if n0 == 0.0 || n1 == 0.0 || variance < 1e-9 {
        return floor;
    }
    let m0 = s0 / n0;
    let m1 = (mean * n - s0) / n1;
    let between = n0 / n * n1 / n * (m0 - m1).powi(2);
    if between / variance < 0.5 {
        floor
    } else {
        t.max(floor)
    }
}

/// One threshold for the whole frame, then components. Pixels count with
/// their distance from the threshold as weight, so centroids lean toward
/// the clearly foreground pixels. `threshold` in the params is the least
/// separation from the threshold a pixel needs.
pub fn detect_global(channel: &Gray, params: &BlobParams) -> Vec<Blob> {
    let t = global_threshold(channel, params.global_floor) as i32;
    let data = channel
        .data
        .iter()
        .map(|&v| {
            let d = match params.polarity {
                Polarity::Bright => v as i32 - t,
                Polarity::Dark => t - v as i32,
            };
            d.clamp(0, 255) as u8
        })
        .collect();
    let contrast = Gray { width: channel.width, height: channel.height, data };
    detect_in_contrast(&contrast, &BlobParams { threshold: params.threshold.max(1), ..params.clone() })
}

/// Detection by either segmentation.
pub fn detect_with(channel: &Gray, params: &BlobParams, segmentation: Segmentation) -> Vec<Blob> {
    match segmentation {
        Segmentation::Local => detect(channel, params),
        Segmentation::Global => detect_global(channel, params),
    }
}

pub fn detect_in_contrast(contrast: &Gray, params: &BlobParams) -> Vec<Blob> {
    components(contrast, params.threshold)
        .into_iter()
        .filter(|b| {
            let area = b.area as usize;
            area >= params.min_area
                && area <= params.max_area
                && b.fill_ratio() >= params.min_fill
                && (params.keep_border || !b.touches_border)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc_image(w: usize, h: usize, discs: &[(f64, f64, f64)], fg: u8, bg: u8) -> Gray {
        let mut g = Gray::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let cx = x as f64 + 0.5;
                let cy = y as f64 + 0.5;
                let inside = discs.iter().any(|&(dx, dy, r)| (cx - dx).hypot(cy - dy) <= r);
                g.data[y * w + x] = if inside { fg } else { bg };
            }
        }
        g
    }

    #[test]
    fn finds_discs_with_subpixel_centers() {
        let g = disc_image(120, 80, &[(30.3, 40.6, 6.0), (90.0, 20.0, 3.0)], 200, 50);
        let blobs = detect(&g, &BlobParams { radius: 20, threshold: 30, ..BlobParams::default() });
        assert_eq!(blobs.len(), 2, "{blobs:?}");
        let big = blobs.iter().max_by(|a, b| a.area.partial_cmp(&b.area).unwrap()).unwrap();
        assert!((big.x - 30.3).abs() < 0.3 && (big.y - 40.6).abs() < 0.3, "{big:?}");
        assert!((big.area - std::f64::consts::PI * 36.0).abs() < 12.0, "{}", big.area);
        assert!((big.major - 12.0).abs() < 1.5 && (big.minor - 12.0).abs() < 1.5, "{big:?}");
        assert!(big.fill_ratio() > 0.7);
    }

    #[test]
    fn dark_polarity_and_border_filter() {
        let mut g = disc_image(60, 60, &[(30.0, 30.0, 5.0)], 20, 180);
        // A dark bar along the top edge.
        for x in 0..60 {
            g.data[x] = 20;
            g.data[60 + x] = 20;
        }
        let params = BlobParams { polarity: Polarity::Dark, radius: 15, threshold: 30, min_fill: 0.0, ..BlobParams::default() };
        let blobs = detect(&g, &params);
        assert_eq!(blobs.len(), 1);
        assert!(!blobs[0].touches_border);
        let all = detect(&g, &BlobParams { keep_border: true, ..params });
        assert_eq!(all.len(), 2);
        assert!(all.iter().any(|b| b.touches_border && b.fill_ratio() > 0.9));
    }

    #[test]
    fn global_threshold_finds_regions_inside_a_bright_field() {
        // Bright pads inside a bright field, parted by thin dark lines: no
        // local contrast, but one threshold separates them.
        let mut g = Gray::new(100, 60);
        for v in g.data.iter_mut() {
            *v = 200;
        }
        for y in 0..60 {
            for x in 0..100 {
                let on_line = x % 25 == 0 || x % 25 == 1 || y % 30 == 0 || y % 30 == 1;
                if on_line {
                    g.data[y * 100 + x] = 90;
                }
            }
        }
        let params = BlobParams { threshold: 5, min_area: 10, keep_border: false, ..BlobParams::default() };
        let global = detect_global(&g, &params);
        // Cells not touching the border: 3 across, 1 down.
        assert_eq!(global.len(), 3, "{global:?}");
        assert!(global.iter().all(|b| (b.area - 23.0 * 28.0).abs() < 1.0), "{global:?}");
    }

    #[test]
    fn global_threshold_floor_and_split() {
        // Dark background, mid substrate, bright copper: the split lands
        // between substrate and copper once the background is floored off.
        let mut data = vec![10u8; 600];
        data.extend(vec![110u8; 200]);
        data.extend(vec![190u8; 200]);
        let g = Gray::from_vec(100, 10, data).unwrap();
        let t = global_threshold(&g, 90);
        assert!((110..190).contains(&t), "{t}");
        // Only one material above the floor: the floor is the threshold.
        let mut data = vec![10u8; 800];
        data.extend(vec![190u8; 200]);
        let g = Gray::from_vec(100, 10, data).unwrap();
        assert_eq!(global_threshold(&g, 90), 90);
        // Nothing above the floor.
        let g = Gray::from_vec(10, 10, vec![10u8; 100]).unwrap();
        assert_eq!(global_threshold(&g, 90), 255);
    }

    #[test]
    fn diagonal_pixels_connect() {
        let mut g = Gray::new(5, 5);
        for i in 0..5 {
            g.data[i * 5 + i] = 255;
        }
        let blobs = components(&g, 100);
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].area, 5.0);
        assert!((blobs[0].angle - std::f64::consts::FRAC_PI_4).abs() < 1e-6);
    }
}
