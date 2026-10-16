//! The live viewer: a camera thread, a localizer thread, and an egui
//! window showing the frame with the pose drawn over it and the board
//! with the field of view drawn on it.

use std::sync::mpsc::{sync_channel, Receiver, Sender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use pcbv_board::{Board, Finish, Point, Side};
use pcbv_camera::{spawn, CaptureConfig, CaptureHandle, Frame};
use pcbv_vision::draw::{board_map, overlay};
use pcbv_vision::{Feature, Localizer, LocalizerConfig, Look, Observation, Rgb, State};

/// What the window needs of the model, sent once per configuration.
pub struct ModelView {
    pub map: Rgb,
    pub features: Vec<Feature>,
    pub outline: Vec<Vec<Point>>,
    /// View frame mm to map pixels.
    pub map_px_per_mm: f64,
    pub map_origin: (f64, f64),
    pub side: Side,
}

pub struct FrameResult {
    pub frame: Arc<Rgb>,
    pub obs: Observation,
    pub fps: f64,
    pub decode_ms: f64,
}

pub enum Command {
    Pause(bool),
    Reset,
    Configure(Box<LocalizerConfig>),
    Snapshot(std::path::PathBuf),
}

pub struct Worker {
    pub results: Receiver<FrameResult>,
    pub models: Receiver<Arc<ModelView>>,
    pub commands: Sender<Command>,
}

fn model_view(localizer: &Localizer) -> ModelView {
    let model = &localizer.model;
    let raster = &model.raster;
    let (ox, oy) = model.view_to_raster(Point::new(0.0, 0.0));
    let (x1, _) = model.view_to_raster(Point::new(1.0, 0.0));
    ModelView {
        map: Rgb {
            width: raster.width(),
            height: raster.height(),
            data: raster.rgb(),
        },
        features: model.features.clone(),
        outline: model.outline.clone(),
        map_px_per_mm: x1 - ox,
        map_origin: (ox, oy),
        side: model.side,
    }
}

/// Where frames come from: a camera, or pictures replayed from disk.
pub enum Source {
    Camera(CaptureConfig),
    /// Image files shown in turn at the given rate, forever.
    Replay {
        paths: Vec<std::path::PathBuf>,
        fps: f64,
    },
}

/// Pictures from disk pushed through the same handle a camera uses.
fn spawn_replay(paths: Vec<std::path::PathBuf>, fps: f64) -> Result<CaptureHandle> {
    if paths.is_empty() {
        return Err(anyhow!("nothing to replay"));
    }
    let images: Vec<Rgb> = paths
        .iter()
        .map(|p| Rgb::load(p).map_err(|e| anyhow!("{e}")))
        .collect::<Result<_>>()?;
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (tx, rx) = sync_channel::<Frame>(1);
    let stop_thread = stop.clone();
    let period = Duration::from_secs_f64(1.0 / fps.max(0.1));
    let thread = std::thread::Builder::new()
        .name("replay".into())
        .spawn(move || {
            let mut sequence = 0u64;
            loop {
                for img in &images {
                    if stop_thread.load(std::sync::atomic::Ordering::Relaxed) {
                        return Ok(());
                    }
                    sequence += 1;
                    let frame = Frame {
                        rgb: img.clone(),
                        sequence,
                        captured: Instant::now(),
                        decode_ms: 0.0,
                    };
                    match tx.try_send(frame) {
                        Ok(()) | Err(TrySendError::Full(_)) => {}
                        Err(TrySendError::Disconnected(_)) => return Ok(()),
                    }
                    std::thread::sleep(period);
                }
            }
        })
        .map_err(|e| anyhow!("spawning replay thread: {e}"))?;
    Ok(CaptureHandle {
        frames: rx,
        stop,
        thread,
    })
}

/// Start the frame source and the localizer on their own threads.
pub fn start(
    board: Arc<Board>,
    config: LocalizerConfig,
    source: Source,
    json: bool,
) -> Result<Worker> {
    let (result_tx, result_rx) = sync_channel::<FrameResult>(1);
    let (model_tx, model_rx) = sync_channel::<Arc<ModelView>>(2);
    let (command_tx, command_rx) = std::sync::mpsc::channel::<Command>();
    let capture: CaptureHandle = match source {
        Source::Camera(camera) => spawn(camera),
        Source::Replay { paths, fps } => spawn_replay(paths, fps)?,
    };
    std::thread::Builder::new()
        .name("localizer".into())
        .spawn(move || {
            let mut localizer = Localizer::new(&board, config);
            let _ = model_tx.send(Arc::new(model_view(&localizer)));
            let mut paused = false;
            let mut last_frame: Option<Arc<Rgb>> = None;
            let mut last_obs: Option<Observation> = None;
            let mut fps = 0.0;
            let mut last_time = Instant::now();
            loop {
                while let Ok(cmd) = command_rx.try_recv() {
                    match cmd {
                        Command::Pause(p) => paused = p,
                        Command::Reset => localizer.reset(),
                        Command::Configure(c) => {
                            localizer = Localizer::new(&board, *c);
                            let _ = model_tx.try_send(Arc::new(model_view(&localizer)));
                        }
                        Command::Snapshot(dir) => {
                            if let (Some(f), Some(o)) = (&last_frame, &last_obs) {
                                let stamp = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| d.as_secs())
                                    .unwrap_or(0);
                                let _ = std::fs::create_dir_all(&dir);
                                let _ = f.save_png(dir.join(format!("frame-{stamp}.png")));
                                let _ = overlay(f, o, &localizer.model)
                                    .save_png(dir.join(format!("overlay-{stamp}.png")));
                                let _ = board_map(&localizer.model, o)
                                    .save_png(dir.join(format!("map-{stamp}.png")));
                                if let Ok(text) = serde_json::to_string_pretty(o) {
                                    let _ = std::fs::write(
                                        dir.join(format!("pose-{stamp}.json")),
                                        text,
                                    );
                                }
                                eprintln!("snapshot written to {}", dir.display());
                            }
                        }
                    }
                }
                if paused {
                    std::thread::sleep(Duration::from_millis(20));
                    continue;
                }
                let Some(frame) = capture.latest() else {
                    if capture.thread.is_finished() {
                        eprintln!("camera stopped");
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                    continue;
                };
                let rgb = Arc::new(frame.rgb);
                let obs = localizer.process(&rgb);
                let now = Instant::now();
                let dt = now.duration_since(last_time).as_secs_f64();
                last_time = now;
                if dt > 0.0 {
                    fps = 0.8 * fps + 0.2 / dt;
                }
                if json {
                    println!("{}", pose_line(&obs, fps));
                }
                last_frame = Some(rgb.clone());
                last_obs = Some(obs.clone());
                match result_tx.try_send(FrameResult {
                    frame: rgb,
                    obs,
                    fps,
                    decode_ms: frame.decode_ms,
                }) {
                    Ok(()) | Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => return,
                }
            }
        })
        .map_err(|e| anyhow!("spawning localizer thread: {e}"))?;
    Ok(Worker {
        results: result_rx,
        models: model_rx,
        commands: command_tx,
    })
}

/// One line of JSON per frame for another program to read.
pub fn pose_line(obs: &Observation, fps: f64) -> String {
    let mut v = serde_json::json!({
        "state": match obs.state {
            State::Locked => "locked",
            State::Coasting => "coasting",
            State::Lost => "lost",
        },
        "fps": (fps * 10.0).round() / 10.0,
        "spots": obs.spots.len(),
        "look": obs.look.to_string(),
        "ms": (obs.timing.total_ms * 10.0).round() / 10.0,
    });
    if let Some(p) = &obs.pose {
        v["center"] = serde_json::json!([p.center.x, p.center.y]);
        v["angle_deg"] = serde_json::json!(p.angle_deg);
        v["mm_per_px"] = serde_json::json!(p.mm_per_px);
        v["inliers"] = serde_json::json!(p.inliers);
        v["expected"] = serde_json::json!(p.expected);
        v["rms_mm"] = serde_json::json!(p.rms_mm);
        v["image_to_board"] = serde_json::json!(p.image_to_board.0);
        v["corners"] = serde_json::json!(p.corners.iter().map(|c| [c.x, c.y]).collect::<Vec<_>>());
    }
    v.to_string()
}

/// Run without a window, printing a line per frame; stops after `frames`
/// frames when given.
pub fn headless(worker: Worker, frames: Option<usize>) -> Result<()> {
    let mut n = 0;
    loop {
        match worker.results.recv_timeout(Duration::from_secs(10)) {
            Ok(_) => {
                n += 1;
                if frames.is_some_and(|f| n >= f) {
                    return Ok(());
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                return Err(anyhow!("no frames for 10 s"))
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
        }
    }
}

// Solarized.
mod sol {
    use egui::Color32;
    pub const BASE03: Color32 = Color32::from_rgb(0, 43, 54);
    pub const BASE02: Color32 = Color32::from_rgb(7, 54, 66);
    pub const BASE01: Color32 = Color32::from_rgb(88, 110, 117);
    pub const BASE00: Color32 = Color32::from_rgb(101, 123, 131);
    pub const BASE0: Color32 = Color32::from_rgb(131, 148, 150);
    pub const BASE1: Color32 = Color32::from_rgb(147, 161, 161);
    pub const BASE2: Color32 = Color32::from_rgb(238, 232, 213);
    pub const BASE3: Color32 = Color32::from_rgb(253, 246, 227);
    pub const YELLOW: Color32 = Color32::from_rgb(181, 137, 0);
    pub const RED: Color32 = Color32::from_rgb(220, 50, 47);
    pub const MAGENTA: Color32 = Color32::from_rgb(211, 54, 130);
    pub const BLUE: Color32 = Color32::from_rgb(38, 139, 210);
    pub const CYAN: Color32 = Color32::from_rgb(42, 161, 152);
    pub const GREEN: Color32 = Color32::from_rgb(133, 153, 0);
}

fn solarized_visuals(dark: bool) -> egui::Visuals {
    let mut v = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    let (bg, panel, text, faint, stroke) = if dark {
        (
            sol::BASE03,
            sol::BASE02,
            sol::BASE0,
            sol::BASE01,
            sol::BASE1,
        )
    } else {
        (sol::BASE3, sol::BASE2, sol::BASE00, sol::BASE1, sol::BASE01)
    };
    v.panel_fill = panel;
    v.window_fill = bg;
    v.extreme_bg_color = bg;
    v.faint_bg_color = faint.gamma_multiply(0.3);
    v.override_text_color = Some(text);
    v.hyperlink_color = sol::BLUE;
    v.selection.bg_fill = sol::BLUE.gamma_multiply(0.5);
    v.selection.stroke = egui::Stroke::new(1.0, sol::BLUE);
    v.widgets.noninteractive.bg_fill = panel;
    v.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, text);
    v.widgets.inactive.bg_fill = faint.gamma_multiply(0.5);
    v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, text);
    v.widgets.hovered.bg_fill = faint;
    v.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, stroke);
    v.widgets.active.bg_fill = sol::BLUE.gamma_multiply(0.6);
    v.widgets.active.fg_stroke = egui::Stroke::new(1.5, stroke);
    v.widgets.open.bg_fill = faint;
    v
}

pub struct App {
    worker: Worker,
    config: LocalizerConfig,
    model: Option<Arc<ModelView>>,
    latest: Option<FrameResult>,
    frame_tex: Option<egui::TextureHandle>,
    map_tex: Option<egui::TextureHandle>,
    paused: bool,
    dark: bool,
    show_spots: bool,
    show_features: bool,
    threshold: u8,
    look_choice: usize,
    snapshot_dir: std::path::PathBuf,
    status: String,
    /// Save a picture of the window here after a few frames, then close.
    screenshot: Option<std::path::PathBuf>,
    frames_shown: usize,
}

impl App {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        worker: Worker,
        config: LocalizerConfig,
        snapshot_dir: std::path::PathBuf,
        screenshot: Option<std::path::PathBuf>,
    ) -> App {
        let dark = cc.egui_ctx.global_style().visuals.dark_mode;
        cc.egui_ctx.set_visuals(solarized_visuals(dark));
        let threshold = config.threshold;
        App {
            worker,
            config,
            model: None,
            latest: None,
            frame_tex: None,
            map_tex: None,
            paused: false,
            dark,
            show_spots: true,
            show_features: true,
            threshold,
            look_choice: 0,
            snapshot_dir,
            status: String::from("starting camera"),
            screenshot,
            frames_shown: 0,
        }
    }

    /// Ask for a picture of the window once frames are showing, save the
    /// one that comes back, and close.
    fn screenshot_step(&mut self, ctx: &egui::Context) {
        let Some(path) = self.screenshot.clone() else {
            return;
        };
        if let Some(image) = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        }) {
            let [w, h] = image.size;
            let data: Vec<u8> = image
                .pixels
                .iter()
                .flat_map(|c| [c.r(), c.g(), c.b()])
                .collect();
            match Rgb::from_vec(w, h, data) {
                Ok(rgb) => match rgb.save_png(&path) {
                    Ok(()) => eprintln!("screenshot written to {}", path.display()),
                    Err(e) => eprintln!("screenshot: {e}"),
                },
                Err(e) => eprintln!("screenshot: {e}"),
            }
            self.screenshot = None;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if self.latest.is_some() {
            self.frames_shown += 1;
            if self.frames_shown == 30 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            }
        }
    }

    fn send(&mut self, cmd: Command) {
        if self.worker.commands.send(cmd).is_err() {
            self.status = "localizer stopped".into();
        }
    }

    fn reconfigure(&mut self) {
        let config = Box::new(self.config.clone());
        self.send(Command::Configure(config));
    }

    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(m) = self.worker.models.try_recv() {
            let image = egui::ColorImage::from_rgb([m.map.width, m.map.height], &m.map.data);
            self.map_tex = Some(ctx.load_texture("board-map", image, egui::TextureOptions::LINEAR));
            self.model = Some(m);
        }
        let mut newest = None;
        while let Ok(r) = self.worker.results.try_recv() {
            newest = Some(r);
        }
        if let Some(r) = newest {
            let image = egui::ColorImage::from_rgb([r.frame.width, r.frame.height], &r.frame.data);
            match &mut self.frame_tex {
                Some(t) => t.set(image, egui::TextureOptions::LINEAR),
                None => {
                    self.frame_tex =
                        Some(ctx.load_texture("frame", image, egui::TextureOptions::LINEAR))
                }
            }
            self.status = match r.obs.state {
                State::Locked => "locked".into(),
                State::Coasting => "tracking from the last pose".into(),
                State::Lost => format!("searching through {}", r.obs.look),
            };
            self.latest = Some(r);
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let (color, text) = match self.latest.as_ref().map(|r| r.obs.state) {
                Some(State::Locked) => (sol::GREEN, "LOCKED"),
                Some(State::Coasting) => (sol::YELLOW, "COASTING"),
                Some(State::Lost) => (sol::RED, "LOST"),
                None => (sol::YELLOW, "WAITING"),
            };
            ui.colored_label(color, egui::RichText::new(text).strong());
            ui.separator();
            if let Some(r) = &self.latest {
                ui.label(format!(
                    "{:.1} fps, {:.0} ms ({:.0} detect, {:.0} search), decode {:.0} ms, {} spots, {}x{}",
                    r.fps, r.obs.timing.total_ms, r.obs.timing.detect_ms, r.obs.timing.search_ms, r.decode_ms, r.obs.spots.len(), r.obs.width, r.obs.height
                ));
            } else {
                ui.label(&self.status);
            }
            ui.separator();
            if ui.button(if self.paused { "Resume" } else { "Pause" }).clicked() {
                self.paused = !self.paused;
                let p = self.paused;
                self.send(Command::Pause(p));
            }
            if ui.button("Search again").clicked() {
                self.send(Command::Reset);
            }
            if ui.button("Snapshot").clicked() {
                let dir = self.snapshot_dir.clone();
                self.send(Command::Snapshot(dir));
            }
            if ui.button(if self.dark { "Light" } else { "Dark" }).clicked() {
                self.dark = !self.dark;
                ui.ctx().set_visuals(solarized_visuals(self.dark));
            }
            ui.checkbox(&mut self.show_spots, "spots");
            ui.checkbox(&mut self.show_features, "features");
        });
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        ui.heading("Board");
        ui.horizontal(|ui| {
            ui.label("Side");
            changed |= ui
                .selectable_value(&mut self.config.side, Side::Front, "front")
                .changed();
            changed |= ui
                .selectable_value(&mut self.config.side, Side::Back, "back")
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Finish");
            changed |= ui
                .selectable_value(&mut self.config.finish, Finish::Bare, "bare")
                .changed();
            changed |= ui
                .selectable_value(&mut self.config.finish, Finish::Masked, "masked")
                .changed();
        });
        ui.heading("Detector");
        let looks = [
            ("auto", None),
            ("bright spots", Some(Look::BRIGHT_SPOTS)),
            ("copper", Some(Look::COPPER)),
            ("holes", Some(Look::HOLES)),
        ];
        ui.horizontal(|ui| {
            ui.label("Look");
            for (i, (name, _)) in looks.iter().enumerate() {
                if ui
                    .selectable_value(&mut self.look_choice, i, *name)
                    .changed()
                {
                    changed = true;
                }
            }
        });
        if ui
            .add(egui::Slider::new(&mut self.threshold, 5..=120).text("contrast"))
            .drag_stopped()
        {
            changed = true;
        }
        if changed {
            self.config.looks = match looks[self.look_choice].1 {
                None => vec![Look::BRIGHT_SPOTS, Look::COPPER, Look::HOLES],
                Some(l) => vec![l],
            };
            self.config.threshold = self.threshold;
            self.reconfigure();
        }
        ui.separator();
        ui.heading("Pose");
        if let Some(pose) = self.latest.as_ref().and_then(|r| r.obs.pose.as_ref()) {
            egui::Grid::new("pose").num_columns(2).show(ui, |ui| {
                ui.label("center x");
                ui.label(format!("{:.3} mm", pose.center.x));
                ui.end_row();
                ui.label("center y");
                ui.label(format!("{:.3} mm", pose.center.y));
                ui.end_row();
                ui.label("turned");
                ui.label(format!("{:.2} deg", pose.angle_deg));
                ui.end_row();
                ui.label("scale");
                ui.label(format!("{:.2} px/mm", 1.0 / pose.mm_per_px));
                ui.end_row();
                ui.label("found");
                ui.label(format!("{} of {} expected", pose.inliers, pose.expected));
                ui.end_row();
                ui.label("residual");
                ui.label(format!("{:.3} mm rms", pose.rms_mm));
                ui.end_row();
                ui.label("fit");
                ui.label(if pose.projective {
                    "projective"
                } else {
                    "similarity or affine"
                });
                ui.end_row();
            });
        } else {
            ui.label("no pose");
        }
        ui.separator();
        ui.heading("Map");
        let avail = ui.available_size();
        if let (Some(tex), Some(model)) = (&self.map_tex, &self.model) {
            let aspect = model.map.width as f64 / model.map.height as f64;
            let w = avail.x.max(10.0);
            let h = (w as f64 / aspect).min(avail.y.max(10.0) as f64) as f32;
            let w = (h as f64 * aspect) as f32;
            let (response, painter) = ui.allocate_painter(egui::vec2(w, h), egui::Sense::hover());
            let rect = response.rect;
            painter.image(
                tex.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
            let to_screen = |p: Point| {
                let px = model.map_origin.0 + p.x * model.map_px_per_mm;
                let py = model.map_origin.1 + p.y * model.map_px_per_mm;
                egui::pos2(
                    rect.min.x + (px / model.map.width as f64) as f32 * rect.width(),
                    rect.min.y + (py / model.map.height as f64) as f32 * rect.height(),
                )
            };
            if let Some(pose) = self.latest.as_ref().and_then(|r| r.obs.pose.as_ref()) {
                let view_of = |c: &Point| match model.side {
                    Side::Front => *c,
                    Side::Back => Point::new(-c.x, c.y),
                };
                let pts: Vec<egui::Pos2> =
                    pose.corners.iter().map(|c| to_screen(view_of(c))).collect();
                let line = rect.width() / 300.0;
                painter.add(egui::Shape::closed_line(
                    pts,
                    egui::Stroke::new(line.max(1.0), sol::GREEN),
                ));
                let c = to_screen(view_of(&pose.center));
                painter.circle_stroke(
                    c,
                    rect.width() / 60.0,
                    egui::Stroke::new(line.max(1.0), sol::BLUE),
                );
            }
        }
    }

    fn frame_view(&mut self, ui: &mut egui::Ui) {
        let Some(tex) = &self.frame_tex else {
            ui.centered_and_justified(|ui| ui.label(&self.status));
            return;
        };
        let Some(r) = &self.latest else {
            return;
        };
        let avail = ui.available_size();
        let aspect = r.frame.width as f32 / r.frame.height as f32;
        let mut w = avail.x;
        let mut h = w / aspect;
        if h > avail.y {
            h = avail.y;
            w = h * aspect;
        }
        let (response, painter) = ui.allocate_painter(egui::vec2(w, h), egui::Sense::hover());
        let rect = response.rect;
        painter.image(
            tex.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        let scale = rect.width() / r.frame.width as f32;
        let to_screen = |x: f64, y: f64| {
            egui::pos2(rect.min.x + x as f32 * scale, rect.min.y + y as f32 * scale)
        };
        let unit = (rect.width() / 400.0).max(1.0);
        let paired: std::collections::HashSet<usize> = r.obs.pairs.iter().map(|p| p.0).collect();
        if self.show_spots {
            for (i, s) in r.obs.spots.iter().enumerate() {
                let radius = ((s.area / std::f64::consts::PI).sqrt() as f32 * scale).max(unit);
                let color = if paired.contains(&i) {
                    sol::GREEN
                } else {
                    sol::YELLOW
                };
                painter.circle_stroke(
                    to_screen(s.x, s.y),
                    radius,
                    egui::Stroke::new(unit * 0.5, color),
                );
            }
        }
        if let (Some(pose), Some(model)) = (&r.obs.pose, &self.model) {
            if let Some(inv) = pose.image_to_view.inverse() {
                if self.show_features {
                    let kind = r.obs.look.kind();
                    for f in model.features.iter().filter(|f| f.kind == kind) {
                        let (x, y) = inv.apply(f.x, f.y);
                        if x < 0.0
                            || y < 0.0
                            || x > r.frame.width as f64
                            || y > r.frame.height as f64
                        {
                            continue;
                        }
                        painter.circle_stroke(
                            to_screen(x, y),
                            unit * 0.8,
                            egui::Stroke::new(unit * 0.4, sol::CYAN),
                        );
                    }
                }
                for l in &model.outline {
                    let pts: Vec<egui::Pos2> = l
                        .iter()
                        .map(|p| {
                            let (x, y) = inv.apply(p.x, p.y);
                            to_screen(x, y)
                        })
                        .collect();
                    painter.add(egui::Shape::closed_line(
                        pts,
                        egui::Stroke::new(unit * 0.7, sol::MAGENTA),
                    ));
                }
            }
            let c = to_screen(r.frame.width as f64 / 2.0, r.frame.height as f64 / 2.0);
            let arm = unit * 8.0;
            painter.line_segment(
                [c - egui::vec2(arm, 0.0), c + egui::vec2(arm, 0.0)],
                egui::Stroke::new(unit * 0.6, sol::BLUE),
            );
            painter.line_segment(
                [c - egui::vec2(0.0, arm), c + egui::vec2(0.0, arm)],
                egui::Stroke::new(unit * 0.6, sol::BLUE),
            );
            let label = format!(
                "x {:.2}  y {:.2} mm   {:.1} deg",
                pose.center.x, pose.center.y, pose.angle_deg
            );
            painter.text(
                rect.min + egui::vec2(unit * 4.0, unit * 4.0),
                egui::Align2::LEFT_TOP,
                label,
                egui::FontId::proportional((rect.width() / 40.0).clamp(10.0, 24.0)),
                sol::BASE3,
            );
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        self.screenshot_step(&ctx);
        egui::Panel::top("top").show(ui, |ui| self.top_bar(ui));
        // The side panel starts at a fraction of the window and can be
        // dragged.
        let width = ctx.content_rect().width();
        egui::Panel::right("side")
            .resizable(true)
            .default_size(width * 0.28)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.side_panel(ui));
            });
        egui::CentralPanel::default().show(ui, |ui| self.frame_view(ui));
        ctx.request_repaint_after(Duration::from_millis(8));
    }
}

pub fn run_window(
    worker: Worker,
    config: LocalizerConfig,
    snapshot_dir: std::path::PathBuf,
    screenshot: Option<std::path::PathBuf>,
) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("pcb-vision"),
        ..Default::default()
    };
    eframe::run_native(
        "pcb-vision",
        options,
        Box::new(move |cc| {
            Ok(Box::new(App::new(
                cc,
                worker,
                config,
                snapshot_dir,
                screenshot,
            )))
        }),
    )
    .map_err(|e| anyhow!("window: {e}"))
}
