//! KiCad board file parsing, geometry and rasterization.
//!
//! [`Board::load`] reads a `.kicad_pcb` (KiCad 5 through 10) into a model
//! whose coordinates are all absolute millimeters in KiCad's board frame,
//! x to the right and y down. [`render`] draws a side of it.

pub mod geom;
pub mod model;
pub mod parse;
pub mod render;
pub mod sexpr;

pub use geom::{Point, Rect};
pub use model::*;
pub use parse::{parse_board, BoardError};
pub use render::{outline_loops, render, Finish, Layers, Raster, RenderOptions, Side, Style};
