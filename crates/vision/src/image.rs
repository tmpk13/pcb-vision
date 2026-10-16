//! Image containers and the pixel operations the pipeline needs.
//!
//! Pixel `(x, y)` covers the unit square from `(x, y)` to `(x + 1, y + 1)`,
//! so its center is at `(x + 0.5, y + 0.5)` in continuous coordinates.

use std::path::Path;

use image::ImageDecoder;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("{path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("{path}: {message}")]
    Format { path: String, message: String },
    #[error("buffer does not match {width}x{height}")]
    Dimensions { width: usize, height: usize },
}

/// Which scalar the detector looks at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Channel {
    /// Brightness; separates gold, silver and copper from dark mask or
    /// substrate.
    Luma,
    /// Red minus blue; separates gold and copper from white silkscreen and
    /// from pale substrate.
    Warmth,
    /// Red minus blue over the sum, scaled: warmth that does not change
    /// with how bright the light is, so one threshold holds across a frame.
    Chroma,
}

impl Channel {
    pub fn parse(text: &str) -> Option<Channel> {
        match text.to_ascii_lowercase().as_str() {
            "luma" | "gray" | "grey" => Some(Channel::Luma),
            "warmth" | "warm" | "rb" => Some(Channel::Warmth),
            "chroma" => Some(Channel::Chroma),
            _ => None,
        }
    }
}

impl std::fmt::Display for Channel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Channel::Luma => "luma",
            Channel::Warmth => "warmth",
            Channel::Chroma => "chroma",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gray {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    /// Interleaved, row major.
    pub data: Vec<u8>,
}

/// Summed area table; `sum` gives the total of any rectangle in constant
/// time.
pub struct Integral {
    pub width: usize,
    pub height: usize,
    stride: usize,
    data: Vec<u32>,
}

fn io_error(path: &Path, source: std::io::Error) -> ImageError {
    ImageError::Io {
        path: path.display().to_string(),
        source,
    }
}

fn format_error(path: &Path, e: impl std::fmt::Display) -> ImageError {
    ImageError::Format {
        path: path.display().to_string(),
        message: e.to_string(),
    }
}

pub fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 299 + g as u32 * 587 + b as u32 * 114) / 1000) as u8
}

impl Gray {
    pub fn new(width: usize, height: usize) -> Gray {
        Gray {
            width,
            height,
            data: vec![0; width * height],
        }
    }

    pub fn from_vec(width: usize, height: usize, data: Vec<u8>) -> Result<Gray, ImageError> {
        if data.len() != width * height {
            return Err(ImageError::Dimensions { width, height });
        }
        Ok(Gray {
            width,
            height,
            data,
        })
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Gray, ImageError> {
        Ok(Rgb::load(path)?.to_gray())
    }

    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), ImageError> {
        let path = path.as_ref();
        let img =
            image::GrayImage::from_raw(self.width as u32, self.height as u32, self.data.clone())
                .ok_or(ImageError::Dimensions {
                    width: self.width,
                    height: self.height,
                })?;
        img.save(path).map_err(|e| format_error(path, e))
    }

    pub fn to_rgb(&self) -> Rgb {
        Rgb {
            width: self.width,
            height: self.height,
            data: self.data.iter().flat_map(|&v| [v, v, v]).collect(),
        }
    }

    /// Average of each `factor` by `factor` block; a partial block at the
    /// right or bottom is dropped.
    pub fn downscale(&self, factor: usize) -> Gray {
        let factor = factor.max(1);
        if factor == 1 {
            return self.clone();
        }
        let w = self.width / factor;
        let h = self.height / factor;
        let mut out = Vec::with_capacity(w * h);
        let n = (factor * factor) as u32;
        for by in 0..h {
            for bx in 0..w {
                let mut sum = 0u32;
                for y in by * factor..(by + 1) * factor {
                    let row = &self.data
                        [y * self.width + bx * factor..y * self.width + (bx + 1) * factor];
                    sum += row.iter().map(|&v| v as u32).sum::<u32>();
                }
                out.push(((sum + n / 2) / n) as u8);
            }
        }
        Gray {
            width: w,
            height: h,
            data: out,
        }
    }

    pub fn integral(&self) -> Integral {
        let stride = self.width + 1;
        let mut data = vec![0u32; stride * (self.height + 1)];
        for y in 0..self.height {
            let mut row_sum = 0u32;
            for x in 0..self.width {
                row_sum += self.get(x, y) as u32;
                data[(y + 1) * stride + x + 1] = data[y * stride + x + 1] + row_sum;
            }
        }
        Integral {
            width: self.width,
            height: self.height,
            stride,
            data,
        }
    }

    /// Mean over the square of the given radius around each pixel, the
    /// square clipped at the borders.
    pub fn box_mean(&self, radius: usize) -> Gray {
        let integral = self.integral();
        let mut out = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius + 1).min(self.height);
            for x in 0..self.width {
                let x0 = x.saturating_sub(radius);
                let x1 = (x + radius + 1).min(self.width);
                let n = ((x1 - x0) * (y1 - y0)) as u32;
                out.push(((integral.sum(x0, y0, x1, y1) + n / 2) / n) as u8);
            }
        }
        Gray {
            width: self.width,
            height: self.height,
            data: out,
        }
    }

    /// Bilinear sample at a continuous position; `None` outside the image.
    pub fn sample(&self, x: f64, y: f64) -> Option<f64> {
        let fx = x - 0.5;
        let fy = y - 0.5;
        if fx < 0.0 || fy < 0.0 || fx > (self.width - 1) as f64 || fy > (self.height - 1) as f64 {
            return None;
        }
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;
        let top = self.get(x0, y0) as f64 * (1.0 - tx) + self.get(x1, y0) as f64 * tx;
        let bottom = self.get(x0, y1) as f64 * (1.0 - tx) + self.get(x1, y1) as f64 * tx;
        Some(top * (1.0 - ty) + bottom * ty)
    }
}

/// Otsu's threshold: the level that best splits the histogram in two.
pub fn otsu_threshold(channel: &Gray) -> u8 {
    let mut hist = [0u64; 256];
    for &v in &channel.data {
        hist[v as usize] += 1;
    }
    let total: u64 = channel.data.len() as u64;
    if total == 0 {
        return 128;
    }
    let sum_all: u64 = hist.iter().enumerate().map(|(v, &n)| v as u64 * n).sum();
    let (mut w0, mut sum0) = (0u64, 0u64);
    let mut best = (0.0f64, 128u8);
    for (t, &count) in hist.iter().enumerate().take(255) {
        w0 += count;
        sum0 += t as u64 * count;
        let w1 = total - w0;
        if w0 == 0 || w1 == 0 {
            continue;
        }
        let m0 = sum0 as f64 / w0 as f64;
        let m1 = (sum_all - sum0) as f64 / w1 as f64;
        let between = w0 as f64 * w1 as f64 * (m0 - m1).powi(2);
        if between > best.0 {
            best = (between, t as u8);
        }
    }
    best.1
}

impl Integral {
    /// Sum over the half open rectangle `[x0, x1) x [y0, y1)`.
    #[inline]
    pub fn sum(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> u32 {
        let s = self.stride;
        self.data[y1 * s + x1] + self.data[y0 * s + x0]
            - self.data[y0 * s + x1]
            - self.data[y1 * s + x0]
    }
}

impl Rgb {
    pub fn new(width: usize, height: usize) -> Rgb {
        Rgb {
            width,
            height,
            data: vec![0; width * height * 3],
        }
    }

    pub fn from_vec(width: usize, height: usize, data: Vec<u8>) -> Result<Rgb, ImageError> {
        if data.len() != width * height * 3 {
            return Err(ImageError::Dimensions { width, height });
        }
        Ok(Rgb {
            width,
            height,
            data,
        })
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.width + x) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    #[inline]
    pub fn put(&mut self, x: usize, y: usize, px: [u8; 3]) {
        let i = (y * self.width + x) * 3;
        self.data[i..i + 3].copy_from_slice(&px);
    }

    /// Read a PNG or JPEG, turned the way its orientation tag says.
    pub fn load(path: impl AsRef<Path>) -> Result<Rgb, ImageError> {
        let path = path.as_ref();
        let reader = image::ImageReader::open(path).map_err(|e| io_error(path, e))?;
        let reader = reader
            .with_guessed_format()
            .map_err(|e| io_error(path, e))?;
        let mut decoder = reader.into_decoder().map_err(|e| format_error(path, e))?;
        let orientation = decoder.orientation().map_err(|e| format_error(path, e))?;
        let mut img =
            image::DynamicImage::from_decoder(decoder).map_err(|e| format_error(path, e))?;
        img.apply_orientation(orientation);
        let rgb = img.into_rgb8();
        Ok(Rgb {
            width: rgb.width() as usize,
            height: rgb.height() as usize,
            data: rgb.into_raw(),
        })
    }

    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), ImageError> {
        let path = path.as_ref();
        let img =
            image::RgbImage::from_raw(self.width as u32, self.height as u32, self.data.clone())
                .ok_or(ImageError::Dimensions {
                    width: self.width,
                    height: self.height,
                })?;
        img.save(path).map_err(|e| format_error(path, e))
    }

    pub fn to_gray(&self) -> Gray {
        self.channel(Channel::Luma)
    }

    pub fn channel(&self, channel: Channel) -> Gray {
        let data = self
            .data
            .chunks_exact(3)
            .map(|px| match channel {
                Channel::Luma => luma(px[0], px[1], px[2]),
                Channel::Warmth => px[0].saturating_sub(px[2]),
                Channel::Chroma => {
                    let sum = px[0] as i32 + px[1] as i32 + px[2] as i32 + 1;
                    ((px[0] as i32 - px[2] as i32) * 640 / sum).clamp(0, 255) as u8
                }
            })
            .collect();
        Gray {
            width: self.width,
            height: self.height,
            data,
        }
    }

    /// Average of each `factor` by `factor` block.
    pub fn downscale(&self, factor: usize) -> Rgb {
        let factor = factor.max(1);
        if factor == 1 {
            return self.clone();
        }
        let w = self.width / factor;
        let h = self.height / factor;
        let mut out = Vec::with_capacity(w * h * 3);
        let n = (factor * factor) as u32;
        for by in 0..h {
            for bx in 0..w {
                let mut sum = [0u32; 3];
                for y in by * factor..(by + 1) * factor {
                    let start = (y * self.width + bx * factor) * 3;
                    for px in self.data[start..start + factor * 3].chunks_exact(3) {
                        sum[0] += px[0] as u32;
                        sum[1] += px[1] as u32;
                        sum[2] += px[2] as u32;
                    }
                }
                out.extend(sum.iter().map(|&s| ((s + n / 2) / n) as u8));
            }
        }
        Rgb {
            width: w,
            height: h,
            data: out,
        }
    }

    /// The smallest integer factor that brings the width to `max_width` or
    /// less.
    pub fn factor_for(&self, max_width: usize) -> usize {
        if max_width == 0 {
            return 1;
        }
        self.width.div_ceil(max_width).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_and_box_mean() {
        let g = Gray::from_vec(4, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
        let i = g.integral();
        assert_eq!(i.sum(0, 0, 4, 3), 78);
        assert_eq!(i.sum(1, 1, 3, 3), 6 + 7 + 10 + 11);
        let m = g.box_mean(1);
        // Center pixel (1, 1): mean of the 3x3 block = 6.
        assert_eq!(m.get(1, 1), 6);
        // Corner (0, 0): mean of the clipped 2x2 block = (1+2+5+6)/4 = 3.5 -> 4.
        assert_eq!(m.get(0, 0), 4);
    }

    #[test]
    fn downscale_averages_blocks() {
        let g = Gray::from_vec(4, 2, vec![0, 100, 10, 20, 50, 50, 30, 40]).unwrap();
        let d = g.downscale(2);
        assert_eq!((d.width, d.height), (2, 1));
        assert_eq!(d.data, vec![50, 25]);
        let rgb =
            Rgb::from_vec(2, 2, vec![0, 0, 0, 200, 100, 0, 100, 100, 100, 100, 200, 0]).unwrap();
        assert_eq!(rgb.downscale(2).data, vec![100, 100, 25]);
        assert_eq!(rgb.factor_for(1), 2);
    }

    #[test]
    fn channels() {
        let rgb = Rgb::from_vec(3, 1, vec![220, 180, 60, 240, 240, 240, 110, 90, 30]).unwrap();
        assert_eq!(rgb.channel(Channel::Warmth).data, vec![160, 0, 80]);
        assert_eq!(rgb.to_gray().data[1], 240);
        // Chroma does not care that the third pixel is half as bright.
        let chroma = rgb.channel(Channel::Chroma).data;
        assert_eq!(chroma[1], 0);
        assert!(
            (chroma[0] as i32 - chroma[2] as i32).abs() <= 2,
            "{chroma:?}"
        );
    }

    #[test]
    fn otsu_splits_two_levels() {
        let mut data = vec![30u8; 500];
        data.extend(vec![200u8; 300]);
        let g = Gray::from_vec(80, 10, data).unwrap();
        let t = otsu_threshold(&g);
        assert!((30..200).contains(&t), "{t}");
    }

    #[test]
    fn bilinear_sampling() {
        let g = Gray::from_vec(2, 2, vec![0, 100, 100, 200]).unwrap();
        assert_eq!(g.sample(0.5, 0.5), Some(0.0));
        assert_eq!(g.sample(1.0, 1.0), Some(100.0));
        assert_eq!(g.sample(1.5, 1.5), Some(200.0));
        assert_eq!(g.sample(-0.1, 0.5), None);
    }
}
