//! The parser against boards dumped through KiCad's own pcbnew module.

use std::collections::BTreeSet;
use std::path::PathBuf;

use pcbv_board::{Board, PadKind, Point};
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/boards")
}

fn load(name: &str) -> (Board, Value) {
    let dir = fixture_dir();
    load_pair(
        &dir.join(format!("{name}.kicad_pcb")),
        &dir.join(format!("{name}.fixture.json")),
    )
}

fn load_pair(pcb: &std::path::Path, fixture: &std::path::Path) -> (Board, Value) {
    let board = Board::load(pcb).unwrap();
    let text = std::fs::read_to_string(fixture).unwrap();
    (board, serde_json::from_str(&text).unwrap())
}

fn point(v: &Value) -> Point {
    Point::new(v[0].as_f64().unwrap(), v[1].as_f64().unwrap())
}

fn angle_eq(a: f64, b: f64) -> bool {
    ((a - b).rem_euclid(360.0)).min((b - a).rem_euclid(360.0)) < 1e-3
}

fn key(p: Point) -> (i64, i64) {
    ((p.x * 1000.0).round() as i64, (p.y * 1000.0).round() as i64)
}

fn check(name: &str) {
    let (board, fixture) = load(name);
    check_against(name, &board, &fixture);
}

fn check_against(name: &str, board: &Board, fixture: &Value) {
    let footprints = fixture["footprints"].as_array().unwrap();
    assert_eq!(
        board.footprints.len(),
        footprints.len(),
        "{name}: footprint count"
    );
    let mut pads_checked = 0;
    for want in footprints {
        let reference = want["reference"].as_str().unwrap();
        let got = board
            .footprints
            .iter()
            .find(|f| f.reference == reference)
            .unwrap_or_else(|| panic!("{name}: footprint {reference} missing"));
        assert!(
            got.at.dist(point(&want["at"])) < 1e-4,
            "{name}: {reference} position {:?}",
            got.at
        );
        assert!(
            angle_eq(got.rotation, want["rotation"].as_f64().unwrap()),
            "{name}: {reference} rotation"
        );
        assert_eq!(
            got.layer,
            want["layer"].as_str().unwrap(),
            "{name}: {reference} layer"
        );
        let pads = want["pads"].as_array().unwrap();
        assert_eq!(got.pads.len(), pads.len(), "{name}: {reference} pad count");
        for wp in pads {
            let number = wp["number"].as_str().unwrap();
            let at = point(&wp["at"]);
            // pcbnew lists every copper layer a "*.Cu" pad could be on, the
            // parser only the ones the board has.
            let want_layers: BTreeSet<&str> = wp["layers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l.as_str().unwrap())
                .filter(|l| board.has_layer(l))
                .collect();
            // Pads on both sides of an edge connector share a number and a
            // position, so the layers pick the pad.
            let gp = got
                .pads
                .iter()
                .filter(|p| p.number == number && p.at.dist(at) < 1e-4)
                .find(|p| {
                    p.layers.iter().map(|s| s.as_str()).collect::<BTreeSet<_>>() == want_layers
                })
                .unwrap_or_else(|| {
                    panic!("{name}: {reference} pad {number} at {at:?} on {want_layers:?} missing")
                });
            assert!(
                angle_eq(gp.rotation, wp["rotation"].as_f64().unwrap()),
                "{name}: {reference} pad {number} rotation {} want {}",
                gp.rotation,
                wp["rotation"]
            );
            let size = point(&wp["size"]);
            assert!(
                (gp.size.0 - size.x).abs() < 1e-6 && (gp.size.1 - size.y).abs() < 1e-6,
                "{name}: {reference} pad {number} size"
            );
            let drill = point(&wp["drill"]);
            match gp.drill {
                Some(d) => assert!(
                    (d.width - drill.x).abs() < 1e-6 && (d.height - drill.y).abs() < 1e-6,
                    "{name}: {reference} pad {number} drill"
                ),
                None => assert!(
                    drill.x == 0.0,
                    "{name}: {reference} pad {number} has no drill, want {drill:?}"
                ),
            }
            let kind = match gp.kind {
                PadKind::Smd => "smd",
                PadKind::ThruHole => "thru_hole",
                PadKind::NpThruHole => "np_thru_hole",
                PadKind::Connect => "connect",
            };
            assert_eq!(
                kind,
                wp["kind"].as_str().unwrap(),
                "{name}: {reference} pad {number} kind"
            );
            pads_checked += 1;
        }
    }
    assert!(pads_checked > 0);

    let tracks = fixture["tracks"].as_array().unwrap();
    assert_eq!(board.tracks.len(), tracks.len(), "{name}: track count");
    let want: BTreeSet<_> = tracks
        .iter()
        .map(|t| {
            (
                key(point(&t["start"])),
                key(point(&t["end"])),
                (t["width"].as_f64().unwrap() * 1000.0).round() as i64,
                t["layer"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    let got: BTreeSet<_> = board
        .tracks
        .iter()
        .map(|t| {
            (
                key(t.start),
                key(t.end),
                (t.width * 1000.0).round() as i64,
                t.layer.clone(),
            )
        })
        .collect();
    assert_eq!(got, want, "{name}: tracks");

    let arcs = fixture["arcs"].as_array().unwrap();
    assert_eq!(board.arcs.len(), arcs.len(), "{name}: arc count");
    for a in arcs {
        let mid = point(&a["mid"]);
        assert!(
            board.arcs.iter().any(|g| g.mid.dist(mid) < 1e-4),
            "{name}: arc through {mid:?}"
        );
    }

    let vias = fixture["vias"].as_array().unwrap();
    assert_eq!(board.vias.len(), vias.len(), "{name}: via count");
    for v in vias {
        let at = point(&v["at"]);
        let g = board
            .vias
            .iter()
            .find(|g| g.at.dist(at) < 1e-4)
            .unwrap_or_else(|| panic!("{name}: via at {at:?}"));
        assert!((g.drill - v["drill"].as_f64().unwrap()).abs() < 1e-6);
        assert!((g.size - v["size"].as_f64().unwrap()).abs() < 1e-6);
    }

    let bbox = fixture["edge_bbox"].as_array().unwrap();
    let got = board.outline_bbox().unwrap();
    for (g, w) in [got.min.x, got.min.y, got.max.x, got.max.y]
        .iter()
        .zip(bbox)
    {
        assert!(
            (g - w.as_f64().unwrap()).abs() < 0.01,
            "{name}: edge bbox {got:?} want {bbox:?}"
        );
    }

    let zones = fixture["zones"].as_array().unwrap();
    if zones.is_empty() {
        // Dumped without zones.
        return;
    }
    let want_outlines: usize = zones
        .iter()
        .map(|z| z["outlines"].as_array().unwrap().len())
        .sum();
    let got_outlines: usize = board.zones.iter().map(|z| z.outlines.len()).sum();
    assert_eq!(got_outlines, want_outlines, "{name}: zone outline count");
    let want_points: BTreeSet<_> = zones
        .iter()
        .flat_map(|z| z["outlines"].as_array().unwrap().iter())
        .flat_map(|o| o.as_array().unwrap().iter())
        .map(|p| key(point(p)))
        .collect();
    let got_points: BTreeSet<_> = board
        .zones
        .iter()
        .flat_map(|z| z.outlines.iter())
        .flatten()
        .map(|p| key(*p))
        .collect();
    assert_eq!(got_points, want_points, "{name}: zone points");
}

#[test]
fn coupon() {
    check("coupon");
}

#[test]
fn tmf8829() {
    check("tmf8829");
}

#[test]
fn pac1944() {
    check("pac1944");
}

/// A large routed board with hundreds of vias, kept outside the repository:
/// runs when the design and its dump are present on this machine.
#[test]
fn forest_sensor_platform() {
    let pcb = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join("forest/Forest-Sensor-Platform/Forest-Sensor-Platform.kicad_pcb");
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ref/forest.fixture.json");
    if !pcb.exists() || !fixture.exists() {
        eprintln!(
            "skipped: {} or {} missing",
            pcb.display(),
            fixture.display()
        );
        return;
    }
    let (board, value) = load_pair(&pcb, &fixture);
    check_against("forest", &board, &value);
    let stats = board.stats();
    assert!(stats.vias > 700 && stats.zone_outlines > 500, "{stats:?}");
}
