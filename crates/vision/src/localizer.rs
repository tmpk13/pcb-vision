//! The frame to frame pipeline: spots in the picture, a search when the
//! board is lost, tracking while it is held, and the pose that results.

use std::time::Instant;

use pcbv_board::{Board, Finish, Point, Side};
use serde::{Deserialize, Serialize};

use crate::blobs::{detect_with, BlobParams, Polarity, Segmentation};
use crate::image::{Channel, Gray, Rgb};

use crate::matcher::{match_constellation_with, ImagePoint, MatchParams, ModelIndex};
use crate::model::{view_of, BoardModel, FeatureKind};
use crate::refine::{initial_from_pairs, kind_for_polarity, refine, Fit, RefineParams};
use crate::xform::Homography;

/// One way of seeing spots: which scalar, how it is split, and whether
/// the spots are the bright or the dark side of the split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Look {
    pub channel: Channel,
    pub segmentation: Segmentation,
    pub polarity: Polarity,
}

impl Look {
    /// Bright pads against dark mask or substrate.
    pub const BRIGHT_SPOTS: Look = Look {
        channel: Channel::Luma,
        segmentation: Segmentation::Local,
        polarity: Polarity::Bright,
    };
    /// Copper, gold or not, against everything else.
    pub const COPPER: Look = Look {
        channel: Channel::Chroma,
        segmentation: Segmentation::Global,
        polarity: Polarity::Bright,
    };
    /// Drilled holes and vias as dark spots, for a side that is mostly one
    /// pour.
    pub const HOLES: Look = Look {
        channel: Channel::Luma,
        segmentation: Segmentation::Local,
        polarity: Polarity::Dark,
    };

    /// The kind of model feature the look finds.
    pub fn kind(&self) -> FeatureKind {
        kind_for_polarity(self.polarity == Polarity::Bright)
    }

    /// "channel-segmentation[-polarity]", bright when the polarity is left
    /// out.
    pub fn parse(text: &str) -> Option<Look> {
        let mut parts = text.split('-');
        let channel = Channel::parse(parts.next()?)?;
        let segmentation = Segmentation::parse(parts.next()?)?;
        let polarity = match parts.next() {
            Some(p) => Polarity::parse(p)?,
            None => Polarity::Bright,
        };
        Some(Look {
            channel,
            segmentation,
            polarity,
        })
    }
}

impl std::fmt::Display for Look {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}-{}-{}",
            self.channel, self.segmentation, self.polarity
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalizerConfig {
    pub side: Side,
    pub finish: Finish,
    /// Ways of seeing, tried in turn after failed searches until one
    /// locks.
    pub looks: Vec<Look>,
    /// Sampling of the board model, px per mm.
    pub model_px_per_mm: f64,
    /// Frames are shrunk to at most this width before detection.
    pub work_width: usize,
    /// Local mean radius for the detector as a fraction of the work width.
    pub blob_radius: f64,
    pub threshold: u8,
    /// Least spot area, px at work size.
    pub min_area: usize,
    /// Largest spot area as a fraction of the work frame.
    pub max_area: f64,
    pub min_fill: f64,
    pub matching: MatchParams,
    pub refine: RefineParams,
    /// Pairing tolerance while tracking, mm.
    pub track_tolerance: f64,
    /// Least fraction of the model's features expected in the frame that
    /// must have been found for a pose to count.
    pub min_coverage: f64,
    /// Least fraction of the spots the pose puts on the board that must be
    /// explained by it.
    pub min_explained: f64,
    /// Frames without a verified pose before the search runs again.
    pub lost_after: u32,
}

impl Default for LocalizerConfig {
    fn default() -> LocalizerConfig {
        LocalizerConfig {
            side: Side::Front,
            finish: Finish::Bare,
            looks: vec![Look::BRIGHT_SPOTS, Look::COPPER, Look::HOLES],
            model_px_per_mm: 20.0,
            work_width: 1024,
            blob_radius: 0.05,
            threshold: 40,
            min_area: 6,
            max_area: 0.05,
            min_fill: 0.3,
            matching: MatchParams::default(),
            refine: RefineParams::default(),
            track_tolerance: 0.4,
            min_coverage: 0.25,
            min_explained: 0.2,
            lost_after: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum State {
    /// No pose; every frame runs the search.
    Lost,
    /// No pose this frame, but the last one is still tried first.
    Coasting,
    /// A verified pose for this frame.
    Locked,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    /// Full size image pixels to board frame millimeters.
    pub image_to_board: Homography,
    /// Full size image pixels to view frame millimeters.
    pub image_to_view: Homography,
    pub inliers: usize,
    pub spots: usize,
    /// Model features of the matched kind that the pose puts in the frame.
    pub expected: usize,
    pub rms_mm: f64,
    pub projective: bool,
    /// Board frame point under the center of the image.
    pub center: Point,
    /// Direction of the image's x axis in the view frame, degrees,
    /// counterclockwise positive on screen.
    pub angle_deg: f64,
    pub mm_per_px: f64,
    /// Board frame points under the image corners, clockwise from top left.
    pub corners: [Point; 4],
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Timing {
    pub prepare_ms: f64,
    pub detect_ms: f64,
    pub search_ms: f64,
    pub refine_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub state: State,
    pub pose: Option<Pose>,
    /// Spots in full size image pixels.
    pub spots: Vec<ImagePoint>,
    /// `(spot index, feature index)` for the spots on a feature.
    pub pairs: Vec<(usize, usize)>,
    /// Whether this frame ran the search rather than tracking.
    pub searched: bool,
    pub hypotheses: usize,
    /// How this frame was looked at.
    pub look: Look,
    pub timing: Timing,
    pub width: usize,
    pub height: usize,
}

pub struct Localizer {
    pub config: LocalizerConfig,
    pub model: BoardModel,
    pub index: ModelIndex,
    /// Index into the config's looks of the one in use; moves on after a
    /// failed search.
    pub look: usize,
    state: State,
    /// Work size pixels to view frame millimeters.
    last: Option<Homography>,
    misses: u32,
    /// Millimeters per work pixel seen at the last lock; narrows the search.
    scale_seen: Option<f64>,
    failed_searches: u32,
}

impl Localizer {
    pub fn new(board: &Board, config: LocalizerConfig) -> Localizer {
        let model = BoardModel::build(board, config.side, config.finish, config.model_px_per_mm);
        let index = ModelIndex::new(
            model.features.clone(),
            config.matching.neighbors,
            config.matching.tolerance.max(0.5) * 2.0,
        );
        Localizer {
            config,
            model,
            index,
            look: 0,
            state: State::Lost,
            last: None,
            misses: 0,
            scale_seen: None,
            failed_searches: 0,
        }
    }

    pub fn current_look(&self) -> Look {
        self.config
            .looks
            .get(self.look)
            .copied()
            .unwrap_or(Look::BRIGHT_SPOTS)
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn reset(&mut self) {
        self.look = 0;
        self.state = State::Lost;
        self.last = None;
        self.misses = 0;
        self.scale_seen = None;
        self.failed_searches = 0;
    }

    fn blob_params(&self, width: usize, height: usize) -> BlobParams {
        let look = self.current_look();
        BlobParams {
            polarity: look.polarity,
            radius: ((width as f64 * self.config.blob_radius) as usize).max(4),
            threshold: match look.segmentation {
                Segmentation::Local => self.config.threshold,
                Segmentation::Global => 1,
            },
            min_area: self.config.min_area,
            max_area: ((width * height) as f64 * self.config.max_area) as usize,
            min_fill: self.config.min_fill,
            keep_border: false,
            // Chroma below this is not copper or gold under any light.
            global_floor: if look.channel == Channel::Chroma {
                90
            } else {
                0
            },
        }
    }

    /// How many model features of the spots' kind a map puts inside a
    /// work size frame.
    fn expected_in_frame(&self, h: &Homography, width: usize, height: usize) -> usize {
        let Some(inv) = h.inverse() else {
            return usize::MAX;
        };
        let kind = self.current_look().kind();
        let (w, h) = (width as f64, height as f64);
        self.index
            .features
            .iter()
            .filter(|f| f.kind == kind)
            .filter(|f| {
                let (x, y) = inv.apply(f.x, f.y);
                x >= 0.0 && y >= 0.0 && x < w && y < h
            })
            .count()
    }

    /// Whether a fit explains enough of what the model says should be
    /// visible, and enough of what was seen where it says the board is.
    fn covered(
        &self,
        fit: &Fit,
        spots: &[ImagePoint],
        width: usize,
        height: usize,
    ) -> (bool, usize) {
        let expected = self.expected_in_frame(&fit.h, width, height);
        let bbox = self.model.bbox.expand(1.0);
        let on_board = spots
            .iter()
            .filter(|p| {
                let (x, y) = fit.h.apply(p.x, p.y);
                bbox.contains(Point::new(x, y))
            })
            .count();
        let pairs = fit.pairs.len() as f64;
        let ok = fit.pairs.len() >= self.config.matching.min_inliers
            && pairs >= self.config.min_coverage * expected as f64
            && pairs >= self.config.min_explained * on_board as f64;
        (ok, expected)
    }

    /// Spots in a frame already shrunk to work size, in work pixels.
    pub fn detect(&self, channel: &Gray) -> Vec<ImagePoint> {
        let params = self.blob_params(channel.width, channel.height);
        let kind = self.current_look().kind();
        detect_with(channel, &params, self.current_look().segmentation)
            .into_iter()
            .map(|b| ImagePoint {
                x: b.x,
                y: b.y,
                area: b.area,
                kind,
            })
            .collect()
    }

    pub fn process(&mut self, frame: &Rgb) -> Observation {
        let start = Instant::now();
        let factor = frame.factor_for(self.config.work_width);
        let small = if factor > 1 {
            frame.downscale(factor)
        } else {
            frame.clone()
        };
        let channel = small.channel(self.current_look().channel);
        let prepare_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut obs = self.process_channel(&channel, factor as f64, frame.width, frame.height);
        obs.timing.prepare_ms = prepare_ms;
        obs.timing.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        obs
    }

    /// The pipeline on a work size channel image; `factor` is how many
    /// full size pixels one work pixel spans.
    pub fn process_channel(
        &mut self,
        channel: &Gray,
        factor: f64,
        full_width: usize,
        full_height: usize,
    ) -> Observation {
        let start = Instant::now();
        let spots = self.detect(channel);
        let detect_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut timing = Timing {
            detect_ms,
            ..Timing::default()
        };
        let mut searched = false;
        let mut hypotheses = 0;
        let mut fit: Option<(Fit, usize)> = None;
        let (width, height) = (channel.width, channel.height);

        // Track first.
        if let Some(last) = self.last {
            let t = Instant::now();
            let params = RefineParams {
                tolerance: self.config.track_tolerance,
                perspective: 0.0,
                ..self.config.refine.clone()
            };
            let center = (width as f64 / 2.0, height as f64 / 2.0);
            if let Some(f) = refine(&spots, &self.index, &last, Some(center), &params) {
                let (ok, expected) = self.covered(&f, &spots, width, height);
                if ok {
                    fit = Some((f, expected));
                }
            }
            timing.refine_ms += t.elapsed().as_secs_f64() * 1000.0;
        }
        if fit.is_none() && (self.state == State::Lost || self.misses + 1 >= self.config.lost_after)
        {
            searched = true;
            let t = Instant::now();
            let mut params = self.config.matching.clone();
            if let Some(seen) = self.scale_seen.filter(|_| self.failed_searches < 3) {
                params.scale_range = (
                    (seen / 1.5).max(params.scale_range.0),
                    (seen * 1.5).min(params.scale_range.1),
                );
            }
            // A candidate is refined and accepted when it explains enough
            // of what should be in view; the first such one ends the search.
            let mut verified: Option<(Fit, usize)> = None;
            let found = match_constellation_with(&spots, &self.index, &params, |m| {
                let init = initial_from_pairs(&spots, &self.index, &m.pairs, &self.config.refine)
                    .unwrap_or_else(|| m.transform.to_homography());
                match refine(
                    &spots,
                    &self.index,
                    &init,
                    Some(m.anchor),
                    &self.config.refine,
                ) {
                    Some(f) => {
                        let (ok, expected) = self.covered(&f, &spots, width, height);
                        if ok {
                            verified = Some((f, expected));
                        }
                        ok
                    }
                    None => false,
                }
            });
            timing.search_ms = t.elapsed().as_secs_f64() * 1000.0;
            if let Some(m) = &found {
                hypotheses = m.hypotheses;
            }
            fit = verified;
            if fit.is_none() {
                if let Some(m) = found {
                    let init =
                        initial_from_pairs(&spots, &self.index, &m.pairs, &self.config.refine)
                            .unwrap_or_else(|| m.transform.to_homography());
                    if let Some(f) = refine(
                        &spots,
                        &self.index,
                        &init,
                        Some(m.anchor),
                        &self.config.refine,
                    ) {
                        let (ok, expected) = self.covered(&f, &spots, width, height);
                        if ok {
                            fit = Some((f, expected));
                        }
                    }
                }
            }
            timing.refine_ms = t.elapsed().as_secs_f64() * 1000.0 - timing.search_ms;
            if fit.is_none() {
                self.failed_searches += 1;
            }
        }

        let (pose, pairs) = match fit {
            Some((f, expected)) => {
                self.state = State::Locked;
                self.last = Some(f.h);
                self.misses = 0;
                self.failed_searches = 0;
                let cx = channel.width as f64 / 2.0;
                let cy = channel.height as f64 / 2.0;
                self.scale_seen = Some(f.h.local_scale(cx, cy));
                let pose = self.pose_from(
                    &f.h,
                    factor,
                    full_width,
                    full_height,
                    f.pairs.len(),
                    spots.len(),
                    expected,
                    f.rms,
                    f.projective,
                );
                (Some(pose), f.pairs)
            }
            None => {
                if self.last.is_some() {
                    self.misses += 1;
                    if self.misses >= self.config.lost_after {
                        self.state = State::Lost;
                        self.last = None;
                    } else {
                        self.state = State::Coasting;
                    }
                } else {
                    self.state = State::Lost;
                }
                (None, Vec::new())
            }
        };
        let look_used = self.current_look();
        if searched && pose.is_none() && self.config.looks.len() > 1 {
            self.look = (self.look + 1) % self.config.looks.len();
        }
        timing.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let spots_full = spots
            .iter()
            .map(|p| ImagePoint {
                x: p.x * factor,
                y: p.y * factor,
                area: p.area * factor * factor,
                kind: p.kind,
            })
            .collect();
        Observation {
            state: self.state,
            pose,
            spots: spots_full,
            pairs,
            searched,
            hypotheses,
            look: look_used,
            timing,
            width: full_width,
            height: full_height,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn pose_from(
        &self,
        work_to_view: &Homography,
        factor: f64,
        width: usize,
        height: usize,
        inliers: usize,
        spots: usize,
        expected: usize,
        rms_mm: f64,
        projective: bool,
    ) -> Pose {
        let image_to_view = work_to_view.pre_scale(1.0 / factor);
        let mirror = match self.config.side {
            Side::Front => Homography::identity(),
            Side::Back => Homography([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        };
        let image_to_board = image_to_view.then(&mirror);
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let (vx, vy) = image_to_view.apply(cx, cy);
        let center = view_of(self.config.side, Point::new(vx, vy));
        let w = width as f64;
        let h = height as f64;
        let corner = |x: f64, y: f64| {
            let (u, v) = image_to_board.apply(x, y);
            Point::new(u, v)
        };
        Pose {
            image_to_board,
            image_to_view,
            inliers,
            spots,
            expected,
            rms_mm,
            projective,
            center,
            angle_deg: -image_to_view.local_angle(cx, cy).to_degrees(),
            mm_per_px: image_to_view.local_scale(cx, cy),
            corners: [
                corner(0.0, 0.0),
                corner(w, 0.0),
                corner(w, h),
                corner(0.0, h),
            ],
        }
    }
}
