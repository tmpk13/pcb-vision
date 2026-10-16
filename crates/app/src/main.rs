//! `pcbv`: command line front end for the board model and the localizer.

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use pcbv_board::{render, Board, Finish, Point, RenderOptions, Side, Style};
use pcbv_vision::BoardModel;
use pcbv_vision::draw::{board_map, overlay};
use pcbv_vision::{camera, synthesize, Localizer, LocalizerConfig, Look, Polarity, Rgb, State, SynthParams};

#[derive(Parser)]
#[command(name = "pcbv", version, about = "Locate a camera view on a PCB from its KiCad design")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarize a board file: layers, counts, extent, holes
    Board {
        pcb: PathBuf,
        /// Print the summary as JSON
        #[arg(long)]
        json: bool,
    },
    /// Draw one side of a board to a PNG
    Render {
        pcb: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// front or back
        #[arg(long, default_value = "front")]
        side: String,
        #[arg(long, default_value_t = 20.0)]
        px_per_mm: f64,
        /// display, photo, masked, copper or holes
        #[arg(long, default_value = "display")]
        style: String,
        /// bare or masked: whether solder mask hides the copper
        #[arg(long, default_value = "bare")]
        finish: String,
    },
    /// Locate a picture of the board: where on it the camera looked
    Locate {
        pcb: PathBuf,
        image: PathBuf,
        #[command(flatten)]
        view: ViewArgs,
        /// Write the picture with the spots, pairs and projected board drawn on it
        #[arg(short, long)]
        overlay: Option<PathBuf>,
        /// Write the board picture with the camera's field of view drawn on it
        #[arg(long)]
        map: Option<PathBuf>,
        /// Print the result as JSON
        #[arg(long)]
        json: bool,
    },
    /// Make a camera picture of the board from the model, with its true pose
    Synth {
        pcb: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "front")]
        side: String,
        #[arg(long, default_value = "bare")]
        finish: String,
        /// Board frame point under the image center, "x,y" in mm; the board center by default
        #[arg(long)]
        center: Option<String>,
        /// Camera turn, degrees counterclockwise on screen
        #[arg(long, default_value_t = 0.0)]
        angle: f64,
        #[arg(long, default_value_t = 20.0)]
        px_per_mm: f64,
        /// Perspective, "tx,ty" per pixel, around 1e-4 for a visible tilt
        #[arg(long, default_value = "0,0")]
        tilt: String,
        #[arg(long, default_value_t = 1280)]
        width: usize,
        #[arg(long, default_value_t = 960)]
        height: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

/// How the board is looked at.
#[derive(clap::Args)]
struct ViewArgs {
    /// front or back
    #[arg(long, default_value = "front")]
    side: String,
    /// bare or masked: whether solder mask hides the copper
    #[arg(long, default_value = "bare")]
    finish: String,
    /// How to see spots, "channel-segmentation": luma-local (bright pads on
    /// dark mask) or chroma-global (copper against anything else); both are
    /// tried in turn when not given
    #[arg(long)]
    look: Option<String>,
    /// bright spots (pads) or dark spots (holes)
    #[arg(long, default_value = "bright")]
    polarity: String,
    /// Allowed millimeters per pixel, "low:high"
    #[arg(long)]
    scale: Option<String>,
    /// Width frames are shrunk to before detection
    #[arg(long, default_value_t = 1024)]
    work_width: usize,
    /// Detector contrast threshold, 0 to 255
    #[arg(long, default_value_t = 40)]
    threshold: u8,
}

impl ViewArgs {
    fn config(&self) -> Result<LocalizerConfig> {
        let mut config = LocalizerConfig {
            side: side_of(&self.side)?,
            finish: finish_of(&self.finish)?,
            polarity: Polarity::parse(&self.polarity).ok_or_else(|| anyhow!("polarity must be bright or dark"))?,
            work_width: self.work_width,
            threshold: self.threshold,
            ..LocalizerConfig::default()
        };
        if let Some(text) = &self.look {
            let look = Look::parse(text).ok_or_else(|| anyhow!("look must be channel-segmentation, such as luma-local or chroma-global"))?;
            config.looks = vec![look];
        }
        if let Some(range) = &self.scale {
            let (lo, hi) = range.split_once(':').ok_or_else(|| anyhow!("scale must be low:high"))?;
            let lo: f64 = lo.parse()?;
            let hi: f64 = hi.parse()?;
            config.matching.scale_range = (lo, hi);
            config.refine.scale_range = (lo, hi);
        }
        Ok(config)
    }
}

fn pair_of(text: &str) -> Result<(f64, f64)> {
    let (a, b) = text.split_once(',').ok_or_else(|| anyhow!("expected two numbers separated by a comma, not {text:?}"))?;
    Ok((a.trim().parse()?, b.trim().parse()?))
}

fn side_of(text: &str) -> Result<Side> {
    Side::parse(text).ok_or_else(|| anyhow!("side must be front or back, not {text:?}"))
}

fn finish_of(text: &str) -> Result<Finish> {
    Finish::parse(text).ok_or_else(|| anyhow!("finish must be bare or masked, not {text:?}"))
}

fn style_of(text: &str) -> Result<Style> {
    Ok(match text {
        "display" => Style::display(),
        "photo" => Style::photo(),
        "masked" => Style::masked_photo(),
        "copper" => Style::copper_mask(),
        "holes" => Style::hole_mask(),
        other => return Err(anyhow!("unknown style {other:?}")),
    })
}

fn load_board(path: &PathBuf) -> Result<Board> {
    Board::load(path).with_context(|| format!("loading {}", path.display()))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Board { pcb, json } => {
            let board = load_board(&pcb)?;
            let stats = board.stats();
            if json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!("{}: format {} by {}", pcb.display(), board.version, board.generator);
                println!("layers: {}", board.layers.iter().map(|l| l.name.as_str()).collect::<Vec<_>>().join(" "));
                println!(
                    "footprints {} pads {} holes {} tracks {} arcs {} vias {} zone outlines {} graphics {}",
                    stats.footprints, stats.pads, stats.holes, stats.tracks, stats.arcs, stats.vias, stats.zone_outlines, stats.graphics
                );
                match stats.bbox {
                    Some(b) => println!(
                        "extent x {:.3}..{:.3} y {:.3}..{:.3} ({:.2} x {:.2} mm)",
                        b.min.x, b.max.x, b.min.y, b.max.y, b.width(), b.height()
                    ),
                    None => println!("extent: none"),
                }
            }
        }
        Command::Render { pcb, output, side, px_per_mm, style, finish } => {
            let board = load_board(&pcb)?;
            let opts = RenderOptions::new(side_of(&side)?, px_per_mm, style_of(&style)?).with_finish(finish_of(&finish)?);
            let raster = render(&board, &opts);
            raster.save_png(&output).map_err(|e| anyhow!("writing {}: {e}", output.display()))?;
            println!("{}: {}x{} px, {:.3} px/mm, {} side", output.display(), raster.width(), raster.height(), raster.px_per_mm, raster.side);
        }
        Command::Locate { pcb, image, view, overlay: overlay_path, map, json } => {
            let board = load_board(&pcb)?;
            let config = view.config()?;
            let looks = config.looks.len();
            let mut localizer = Localizer::new(&board, config);
            let frame = Rgb::load(&image).with_context(|| format!("loading {}", image.display()))?;
            let mut obs = localizer.process(&frame);
            for _ in 1..looks {
                if obs.pose.is_some() {
                    break;
                }
                obs = localizer.process(&frame);
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&obs)?);
            } else {
                println!(
                    "{}x{} px, {} spots through {}, search {:.0} ms ({} hypotheses), total {:.0} ms",
                    obs.width, obs.height, obs.spots.len(), obs.look, obs.timing.search_ms, obs.hypotheses, obs.timing.total_ms
                );
                match &obs.pose {
                    Some(pose) => {
                        println!(
                            "locked: center x {:.3} y {:.3} mm, turned {:.2} deg, {:.4} mm/px ({:.2} px/mm), {} of {} spots paired, {} of {} expected features found, rms {:.3} mm, {}",
                            pose.center.x, pose.center.y, pose.angle_deg, pose.mm_per_px, 1.0 / pose.mm_per_px, pose.inliers, pose.spots, pose.inliers, pose.expected, pose.rms_mm,
                            if pose.projective { "projective" } else { "similarity" }
                        );
                        for (i, c) in pose.corners.iter().enumerate() {
                            println!("corner {i}: x {:.2} y {:.2}", c.x, c.y);
                        }
                    }
                    None => println!("lost: no pose"),
                }
            }
            if let Some(path) = overlay_path {
                overlay(&frame, &obs, &localizer.model).save_png(&path).with_context(|| format!("writing {}", path.display()))?;
            }
            if let Some(path) = map {
                board_map(&localizer.model, &obs).save_png(&path).with_context(|| format!("writing {}", path.display()))?;
            }
            if obs.state != State::Locked {
                std::process::exit(2);
            }
        }
        Command::Synth { pcb, output, side, finish, center, angle, px_per_mm, tilt, width, height, seed } => {
            let board = load_board(&pcb)?;
            let side = side_of(&side)?;
            let finish = finish_of(&finish)?;
            let source = BoardModel::build(&board, side, finish, (px_per_mm * 1.5).max(20.0));
            let extent = board.bbox().ok_or_else(|| anyhow!("empty board"))?;
            let center = match center {
                Some(text) => {
                    let (x, y) = pair_of(&text)?;
                    Point::new(x, y)
                }
                None => extent.center(),
            };
            let view_center = pcbv_vision::model::view_of(side, center);
            let tilt = pair_of(&tilt)?;
            let truth = camera(view_center, angle, px_per_mm, tilt, width, height);
            let params = SynthParams { width, height, seed, ..SynthParams::default() };
            let frame = synthesize(&source, &truth, &params);
            frame.save_png(&output).with_context(|| format!("writing {}", output.display()))?;
            println!("{}", serde_json::json!({
                "output": output,
                "side": side.to_string(),
                "center_board": [center.x, center.y],
                "angle_deg": angle,
                "px_per_mm": px_per_mm,
                "view_to_image": truth.0,
            }));
        }
    }
    Ok(())
}
