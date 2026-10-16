//! V4L2 camera capture: device listing, mode probing, and a capture loop
//! that hands out RGB frames decoded from MJPEG or converted from YUYV.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use pcbv_vision::Rgb;
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::capture::Parameters;
use v4l::video::Capture;
use v4l::{Device, Format, FourCC};
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

#[derive(Debug, thiserror::Error)]
pub enum CameraError {
    #[error("opening {path}: {source}")]
    Open { path: String, source: std::io::Error },
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("the driver offers {0} rather than MJPG or YUYV")]
    Format(String),
    #[error("jpeg: {0}")]
    Jpeg(String),
    #[error("capture failed {0} times in a row")]
    Stalled(u32),
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Mode {
    pub fourcc: String,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct DeviceInfo {
    pub path: String,
    pub name: String,
    pub modes: Vec<Mode>,
}

/// Video capture devices and the modes each offers.
pub fn list_devices() -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    for node in v4l::context::enum_devices() {
        let path = node.path().to_path_buf();
        let Ok(dev) = Device::with_path(&path) else {
            continue;
        };
        let Ok(caps) = dev.query_caps() else {
            continue;
        };
        if !caps.capabilities.contains(v4l::capability::Flags::VIDEO_CAPTURE) {
            continue;
        }
        let modes = probe_modes(&dev);
        if modes.is_empty() {
            continue;
        }
        out.push(DeviceInfo { path: path.display().to_string(), name: caps.card.clone(), modes });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

fn probe_modes(dev: &Device) -> Vec<Mode> {
    let mut modes = Vec::new();
    let Ok(formats) = dev.enum_formats() else {
        return modes;
    };
    for fmt in formats {
        let fourcc = fmt.fourcc.to_string();
        if fourcc != "MJPG" && fourcc != "YUYV" {
            continue;
        }
        let Ok(sizes) = dev.enum_framesizes(fmt.fourcc) else {
            continue;
        };
        for size in sizes {
            let v4l::framesize::FrameSizeEnum::Discrete(d) = size.size else {
                continue;
            };
            let fps = dev
                .enum_frameintervals(fmt.fourcc, d.width, d.height)
                .map(|list| {
                    list.iter()
                        .filter_map(|i| match i.interval {
                            v4l::frameinterval::FrameIntervalEnum::Discrete(f) if f.numerator > 0 => {
                                Some(f.denominator as f64 / f.numerator as f64)
                            }
                            _ => None,
                        })
                        .fold(0.0, f64::max)
                })
                .unwrap_or(0.0);
            modes.push(Mode { fourcc: fourcc.clone(), width: d.width, height: d.height, fps });
        }
    }
    modes
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptureConfig {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl Default for CaptureConfig {
    fn default() -> CaptureConfig {
        CaptureConfig { path: "/dev/video0".to_string(), width: 1280, height: 720, fps: 30 }
    }
}

pub struct Frame {
    pub rgb: Rgb,
    pub sequence: u64,
    pub captured: Instant,
    pub decode_ms: f64,
}

fn decode_mjpeg(data: &[u8]) -> Result<Rgb, CameraError> {
    let opts = DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), opts);
    let pixels = decoder.decode().map_err(|e| CameraError::Jpeg(format!("{e:?}")))?;
    let (w, h) = decoder.dimensions().ok_or_else(|| CameraError::Jpeg("no dimensions".into()))?;
    Rgb::from_vec(w, h, pixels).map_err(|e| CameraError::Jpeg(e.to_string()))
}

fn yuyv_to_rgb(data: &[u8], width: usize, height: usize) -> Option<Rgb> {
    if data.len() < width * height * 2 {
        return None;
    }
    let mut out = Vec::with_capacity(width * height * 3);
    let convert = |y: f32, u: f32, v: f32| -> [u8; 3] {
        let r = y + 1.402 * (v - 128.0);
        let g = y - 0.344 * (u - 128.0) - 0.714 * (v - 128.0);
        let b = y + 1.772 * (u - 128.0);
        [r.clamp(0.0, 255.0) as u8, g.clamp(0.0, 255.0) as u8, b.clamp(0.0, 255.0) as u8]
    };
    for chunk in data[..width * height * 2].chunks_exact(4) {
        let (y0, u, y1, v) = (chunk[0] as f32, chunk[1] as f32, chunk[2] as f32, chunk[3] as f32);
        out.extend_from_slice(&convert(y0, u, v));
        out.extend_from_slice(&convert(y1, u, v));
    }
    Rgb::from_vec(width, height, out).ok()
}

/// Open the device, set the mode, and pass every decoded frame to `sink`
/// until it returns false or `stop` is set. Frames the sink is not ready
/// for are skipped before decoding when `wants` says so.
pub fn run(config: &CaptureConfig, stop: &AtomicBool, mut wants: impl FnMut() -> bool, mut sink: impl FnMut(Frame) -> bool) -> Result<(), CameraError> {
    let dev = Device::with_path(&config.path).map_err(|source| CameraError::Open { path: config.path.clone(), source })?;
    let mut got = dev.set_format(&Format::new(config.width, config.height, FourCC::new(b"MJPG")))?;
    if got.fourcc != FourCC::new(b"MJPG") {
        got = dev.set_format(&Format::new(config.width, config.height, FourCC::new(b"YUYV")))?;
        if got.fourcc != FourCC::new(b"YUYV") {
            return Err(CameraError::Format(got.fourcc.to_string()));
        }
    }
    let mjpeg = got.fourcc == FourCC::new(b"MJPG");
    let (width, height) = (got.width as usize, got.height as usize);
    // The frame rate is advisory; a driver that ignores it still streams.
    let _ = dev.set_params(&Parameters::with_fps(config.fps));
    let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 4)?;
    let mut sequence = 0u64;
    let mut errors = 0u32;
    while !stop.load(Ordering::Relaxed) {
        let (buf, meta) = match stream.next() {
            Ok(v) => v,
            Err(_) => {
                errors += 1;
                if errors > 10 {
                    return Err(CameraError::Stalled(errors));
                }
                continue;
            }
        };
        errors = 0;
        if !wants() {
            continue;
        }
        let captured = Instant::now();
        let used = meta.bytesused as usize;
        let data = if used > 0 && used <= buf.len() { &buf[..used] } else { buf };
        let decoded = if mjpeg {
            // A torn frame is normal at high rates; drop it and carry on.
            match decode_mjpeg(data) {
                Ok(rgb) => rgb,
                Err(_) => continue,
            }
        } else {
            match yuyv_to_rgb(data, width, height) {
                Some(rgb) => rgb,
                None => continue,
            }
        };
        sequence += 1;
        let frame = Frame { rgb: decoded, sequence, captured, decode_ms: captured.elapsed().as_secs_f64() * 1000.0 };
        if !sink(frame) {
            break;
        }
    }
    Ok(())
}

/// A running capture on its own thread.
pub struct CaptureHandle {
    pub frames: Receiver<Frame>,
    pub stop: Arc<AtomicBool>,
    pub thread: JoinHandle<Result<(), CameraError>>,
}

impl CaptureHandle {
    /// The newest frame waiting, older ones dropped; `None` when nothing
    /// has arrived.
    pub fn latest(&self) -> Option<Frame> {
        let mut latest = None;
        while let Ok(f) = self.frames.try_recv() {
            latest = Some(f);
        }
        latest
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Start capturing on a thread. Frames queue one deep: when the consumer
/// lags, frames are skipped rather than delayed.
pub fn spawn(config: CaptureConfig) -> CaptureHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = sync_channel::<Frame>(1);
    let stop_thread = stop.clone();
    let thread = std::thread::Builder::new()
        .name("camera".into())
        .spawn(move || {
            let sender = tx.clone();
            run(
                &config,
                &stop_thread,
                || true,
                move |frame| match sender.try_send(frame) {
                    Ok(()) | Err(TrySendError::Full(_)) => true,
                    Err(TrySendError::Disconnected(_)) => false,
                },
            )
        })
        .expect("spawn camera thread");
    CaptureHandle { frames: rx, stop, thread }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuyv_conversion() {
        // Two gray pixels at Y 128, no chroma.
        let rgb = yuyv_to_rgb(&[128, 128, 128, 128], 2, 1).unwrap();
        assert_eq!(rgb.data, vec![128, 128, 128, 128, 128, 128]);
        assert!(yuyv_to_rgb(&[0; 3], 2, 1).is_none());
    }

    #[test]
    fn devices_list_without_panicking() {
        for d in list_devices() {
            assert!(!d.modes.is_empty());
            eprintln!("{} {}: {} modes", d.path, d.name, d.modes.len());
        }
    }
}
