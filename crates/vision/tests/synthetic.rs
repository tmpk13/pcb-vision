//! Pictures made from the fixture boards, located again.

use std::path::PathBuf;

use pcbv_board::{Board, Finish, Point, Side};
use pcbv_vision::rng::Rng;
use pcbv_vision::{camera, synthesize, BoardModel, Localizer, LocalizerConfig, Polarity, State, SynthParams};

fn fixture(name: &str) -> Board {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../fixtures/boards/{name}.kicad_pcb"));
    Board::load(path).unwrap()
}

struct Case {
    name: &'static str,
    side: Side,
    finish: Finish,
    px_per_mm: f64,
    angle: f64,
    tilt: (f64, f64),
    center_frac: (f64, f64),
}

fn run(case: &Case, seed: u64) {
    let board = fixture(case.name);
    // The picture's source, sampled finer than any camera in the cases.
    let source = BoardModel::build(&board, case.side, case.finish, 60.0);
    // The default looks start with bright spots, which a bare board
    // without mask does not offer: the localizer has to move on by itself.
    let config = LocalizerConfig { side: case.side, finish: case.finish, polarity: Polarity::Bright, ..LocalizerConfig::default() };
    let mut localizer = Localizer::new(&board, config);
    let bbox = localizer.model.bbox;
    let center = Point::new(
        bbox.min.x + bbox.width() * case.center_frac.0,
        bbox.min.y + bbox.height() * case.center_frac.1,
    );
    let params = SynthParams { width: 1280, height: 960, seed, ..SynthParams::default() };
    let truth = camera(center, case.angle, case.px_per_mm, case.tilt, params.width, params.height);
    let frame = synthesize(&source, &truth, &params);
    let mut obs = localizer.process(&frame);
    for _ in 0..2 {
        if obs.pose.is_some() {
            break;
        }
        obs = localizer.process(&frame);
    }
    let pose = obs.pose.as_ref().unwrap_or_else(|| {
        panic!("{}: no lock through {}; {} spots, {} hypotheses, {:?}", case.name, obs.look, obs.spots.len(), obs.hypotheses, obs.timing)
    });
    assert_eq!(obs.state, State::Locked);
    // Accuracy over the frame where the board is; outside it nothing
    // constrains the fit.
    let inv = truth.inverse().unwrap();
    let mut worst: f64 = 0.0;
    let mut rng = Rng::new(seed);
    let mut checked = 0;
    while checked < 50 {
        let x = rng.range(0.0, params.width as f64);
        let y = rng.range(0.0, params.height as f64);
        let (tx, ty) = inv.apply(x, y);
        if !bbox.contains(Point::new(tx, ty)) {
            continue;
        }
        checked += 1;
        let (ex, ey) = pose.image_to_view.apply(x, y);
        worst = worst.max((tx - ex).hypot(ty - ey));
    }
    assert!(worst < 0.3, "{}: worst error {worst:.3} mm with {} of {} spots, rms {:.3}, {}", case.name, pose.inliers, pose.spots, pose.rms_mm, if pose.projective { "projective" } else { "not projective" });
    assert!(pose.inliers >= 8, "{}: only {} inliers", case.name, pose.inliers);
    eprintln!(
        "{} {:?} {}: {} of {} spots, rms {:.3} mm, worst {worst:.3} mm, search {:.0} ms ({} hypotheses), total {:.0} ms",
        case.name, case.side, obs.look, pose.inliers, pose.spots, pose.rms_mm, obs.timing.search_ms, obs.hypotheses, obs.timing.total_ms
    );

    // A second, slightly moved frame tracks without a search.
    let moved = camera(Point::new(center.x + 0.4, center.y - 0.3), case.angle + 1.0, case.px_per_mm * 1.01, case.tilt, params.width, params.height);
    let frame2 = synthesize(&source, &moved, &SynthParams { seed: seed + 1, ..params.clone() });
    let obs2 = localizer.process(&frame2);
    assert!(!obs2.searched, "{}: second frame searched", case.name);
    let pose2 = obs2.pose.as_ref().expect("tracked");
    let inv2 = moved.inverse().unwrap();
    let mut worst2: f64 = 0.0;
    let mut checked = 0;
    while checked < 50 {
        let x = rng.range(0.0, params.width as f64);
        let y = rng.range(0.0, params.height as f64);
        let (tx, ty) = inv2.apply(x, y);
        if !bbox.contains(Point::new(tx, ty)) {
            continue;
        }
        checked += 1;
        let (ex, ey) = pose2.image_to_view.apply(x, y);
        worst2 = worst2.max((tx - ex).hypot(ty - ey));
    }
    assert!(worst2 < 0.3, "{}: tracking error {worst2:.3} mm with {} of {} spots", case.name, pose2.inliers, pose2.spots);
}

#[test]
fn pac1944_front_straight() {
    run(&Case { name: "pac1944", side: Side::Front, finish: Finish::Bare, px_per_mm: 20.0, angle: 0.0, tilt: (0.0, 0.0), center_frac: (0.5, 0.5) }, 1);
}

#[test]
fn pac1944_front_turned_tilted_window() {
    run(&Case { name: "pac1944", side: Side::Front, finish: Finish::Bare, px_per_mm: 28.0, angle: 37.0, tilt: (1.5e-4, -1e-4), center_frac: (0.65, 0.6) }, 2);
}

#[test]
fn tmf8829_back_turned() {
    run(&Case { name: "tmf8829", side: Side::Back, finish: Finish::Bare, px_per_mm: 40.0, angle: -120.0, tilt: (0.0, 1e-4), center_frac: (0.5, 0.5) }, 3);
}

#[test]
fn tmf8829_front_bare_close() {
    run(&Case { name: "tmf8829", side: Side::Front, finish: Finish::Bare, px_per_mm: 60.0, angle: 200.0, tilt: (0.0, 0.0), center_frac: (0.5, 0.6) }, 4);
}

#[test]
fn pac1944_front_masked() {
    run(&Case { name: "pac1944", side: Side::Front, finish: Finish::Masked, px_per_mm: 15.0, angle: 95.0, tilt: (-1e-4, 1e-4), center_frac: (0.4, 0.5) }, 5);
}

#[test]
fn nothing_in_view_stays_lost() {
    let board = fixture("pac1944");
    let mut localizer = Localizer::new(&board, LocalizerConfig::default());
    let mut rng = Rng::new(9);
    let mut frame = pcbv_vision::Rgb::new(640, 480);
    for px in frame.data.iter_mut() {
        *px = rng.range(60.0, 200.0) as u8;
    }
    let obs = localizer.process(&frame);
    assert_eq!(obs.state, State::Lost);
    assert!(obs.pose.is_none());
}
