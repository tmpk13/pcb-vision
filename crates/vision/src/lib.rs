//! Image processing and localization against a board model.
//!
//! A frame goes through [`Localizer::process`]: spots are detected, paired
//! with the model's features (a search over the constellation when the
//! board is lost, tracking from the last pose otherwise), and a homography
//! from image pixels to board millimeters comes out as a [`Pose`].

pub mod blobs;
pub mod draw;
pub mod image;
pub mod localizer;
pub mod matcher;
pub mod model;
pub mod refine;
pub mod rng;
pub mod synth;
pub mod xform;

pub use blobs::{Blob, BlobParams, Polarity, Segmentation};
pub use image::{Channel, Gray, ImageError, Rgb};
pub use localizer::{Localizer, LocalizerConfig, Look, Observation, Pose, State, Timing};
pub use matcher::{match_constellation, ImagePoint, MatchParams, MatchResult, ModelIndex};
pub use model::{BoardModel, Feature, FeatureKind};
pub use refine::{refine, Fit, RefineParams};
pub use synth::{camera, synthesize, SynthParams};
pub use xform::{Homography, Similarity};
