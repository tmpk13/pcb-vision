//! From a rough transform to a tight one: spots are paired with the
//! feature they land on, a homography is fitted to the pairs, and the two
//! steps repeat with a shrinking tolerance.
//!
//! A similarity found from two neighboring spots is exact there and drifts
//! with distance when the view has perspective, so pairing starts in a
//! small circle around that anchor and the circle grows with each fit: a
//! similarity on a few pairs, an affine map once there are more, and a
//! homography once they are spread across the frame.
//!
//! A map carried over from the last frame can be off by more than the
//! pitch of a pad array, which would pair every pad with its neighbor, so
//! it is first re-anchored: the spot nearest the anchor is tried against
//! each feature near where the map puts it, and the shift that puts the
//! most of its neighbors onto features wins.

use serde::{Deserialize, Serialize};

use crate::matcher::{ImagePoint, ModelIndex};
use crate::model::FeatureKind;
use crate::xform::{fit_affine, rms_error, Homography, Pair, Similarity};

#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    /// Image pixels to view frame millimeters.
    pub h: Homography,
    pub pairs: Vec<(usize, usize)>,
    /// Root mean square distance of the paired spots from their features, mm.
    pub rms: f64,
    /// Whether the fit had enough spread to be projective rather than a
    /// similarity.
    pub projective: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefineParams {
    /// Final pairing tolerance, mm; the first pass starts at twice this.
    pub tolerance: f64,
    /// Extra tolerance per millimeter from the anchor on the first pass.
    pub perspective: f64,
    pub same_kind: bool,
    pub iterations: usize,
    /// Least spread of the paired spots, px, for a projective fit.
    pub min_extent: f64,
    /// Allowed millimeters per pixel anywhere in the frame.
    pub scale_range: (f64, f64),
    pub min_pairs: usize,
    /// How far, in mm, the re-anchoring looks for the feature of the spot
    /// nearest the anchor; zero turns it off.
    pub reanchor_radius: f64,
}

impl Default for RefineParams {
    fn default() -> RefineParams {
        RefineParams {
            tolerance: 0.3,
            perspective: 0.03,
            same_kind: true,
            iterations: 4,
            min_extent: 60.0,
            scale_range: (0.005, 0.5),
            min_pairs: 6,
            reanchor_radius: 2.0,
        }
    }
}

/// Spots paired with the feature nearest to where `h` puts them, when that
/// feature is within the tolerance and clearly nearer than any other. With
/// `growth`, the tolerance widens by that much per millimeter of view
/// distance from the anchor pixel.
pub fn associate(
    points: &[ImagePoint],
    model: &ModelIndex,
    h: &Homography,
    tolerance: f64,
    same_kind: bool,
    growth: Option<((f64, f64), f64)>,
) -> Vec<(usize, usize)> {
    let mut scratch = Vec::new();
    let mut taken = vec![false; model.len()];
    let mut out = Vec::new();
    let anchor = growth.map(|(a, per_mm)| (h.apply(a.0, a.1), per_mm));
    for (i, p) in points.iter().enumerate() {
        let (x, y) = h.apply(p.x, p.y);
        let tol = match anchor {
            Some(((ax, ay), per_mm)) => tolerance + per_mm * (x - ax).hypot(y - ay),
            None => tolerance,
        };
        let kind = if same_kind { Some(p.kind) } else { None };
        let (best, second) = model.nearest_two(x, y, tol * 2.0, kind, &mut scratch);
        let Some((f, d)) = best else {
            continue;
        };
        if d > tol || taken[f] {
            continue;
        }
        if second.is_some_and(|s| s < d * 1.5 + 1e-9) {
            continue;
        }
        taken[f] = true;
        out.push((i, f));
    }
    out
}

fn pairs_to_coords(
    points: &[ImagePoint],
    model: &ModelIndex,
    pairs: &[(usize, usize)],
) -> Vec<Pair> {
    pairs
        .iter()
        .map(|&(i, f)| {
            (
                (points[i].x, points[i].y),
                (model.features[f].x, model.features[f].y),
            )
        })
        .collect()
}

/// Whether the paired spots cover enough of the frame, in both directions,
/// for eight parameters to be determined.
fn spread_ok(coords: &[Pair], min_extent: f64) -> bool {
    if coords.len() < 8 {
        return false;
    }
    let n = coords.len() as f64;
    let mx = coords.iter().map(|p| p.0 .0).sum::<f64>() / n;
    let my = coords.iter().map(|p| p.0 .1).sum::<f64>() / n;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for ((x, y), _) in coords {
        sxx += (x - mx).powi(2);
        syy += (y - my).powi(2);
        sxy += (x - mx) * (y - my);
    }
    let tr = (sxx + syy) / n;
    let det = (sxx * syy - sxy * sxy) / (n * n);
    let small = (tr / 2.0 - ((tr * tr / 4.0) - det).max(0.0).sqrt())
        .max(0.0)
        .sqrt();
    small * 4.0 >= min_extent
}

/// A fitted map must keep its orientation and a sane scale everywhere a
/// spot is.
fn sane(h: &Homography, points: &[ImagePoint], scale_range: (f64, f64)) -> bool {
    if !h.0.iter().all(|v| v.is_finite()) {
        return false;
    }
    let check = |x: f64, y: f64| {
        let j = h.jacobian(x, y);
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        let w = h.0[6] * x + h.0[7] * y + h.0[8];
        det > 0.0 && w > 0.0 && (scale_range.0 * 0.5..=scale_range.1 * 2.0).contains(&det.sqrt())
    };
    points.iter().all(|p| check(p.x, p.y))
}

/// The richest map the pairs determine that fits no worse than a simpler
/// one: a similarity, an affine map, or a homography. Missing a real tilt
/// costs millimeters at the far side of the frame; projective terms fitted
/// to noise cost a fraction of that, and the weighted fit keeps them small.
fn fit_pairs(
    points: &[ImagePoint],
    model: &ModelIndex,
    pairs: &[(usize, usize)],
    params: &RefineParams,
) -> Option<(Homography, bool, f64)> {
    let coords = pairs_to_coords(points, model, pairs);
    let sane_here = |c: &Homography| sane(c, points, params.scale_range);
    let similarity = Similarity::fit(&coords)?.to_homography();
    let mut best = (similarity, false, rms_error(&similarity, &coords));
    if coords.len() >= 5 && spread_ok_n(&coords, params.min_extent / 3.0, 5) {
        if let Some(h) = fit_affine(&coords).filter(sane_here) {
            let rms = rms_error(&h, &coords);
            if rms <= best.2 {
                best = (h, false, rms);
            }
        }
    }
    if spread_ok(&coords, params.min_extent) {
        if let Some(h) = Homography::fit(&coords).filter(sane_here) {
            let rms = rms_error(&h, &coords);
            if rms <= best.2 {
                best = (h, true, rms);
            }
        }
    }
    Some(best)
}

fn spread_ok_n(coords: &[Pair], min_extent: f64, min_count: usize) -> bool {
    coords.len() >= min_count && {
        let n = coords.len() as f64;
        let mx = coords.iter().map(|p| p.0 .0).sum::<f64>() / n;
        let my = coords.iter().map(|p| p.0 .1).sum::<f64>() / n;
        let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
        for ((x, y), _) in coords {
            sxx += (x - mx).powi(2);
            syy += (y - my).powi(2);
            sxy += (x - mx) * (y - my);
        }
        let tr = (sxx + syy) / n;
        let det = (sxx * syy - sxy * sxy) / (n * n);
        let small = (tr / 2.0 - ((tr * tr / 4.0) - det).max(0.0).sqrt())
            .max(0.0)
            .sqrt();
        small * 4.0 >= min_extent
    }
}

/// A starting map from pairs a search produced, when they allow a
/// projective fit; otherwise `None` and the search's own transform serves.
pub fn initial_from_pairs(
    points: &[ImagePoint],
    model: &ModelIndex,
    pairs: &[(usize, usize)],
    params: &RefineParams,
) -> Option<Homography> {
    let (h, projective, _) = fit_pairs(points, model, pairs, params)?;
    projective.then_some(h)
}

/// The map shifted in its output so that the spot nearest the anchor lands
/// on the feature that makes the most of its neighbors land on features.
fn reanchor(
    points: &[ImagePoint],
    model: &ModelIndex,
    h: &Homography,
    anchor: (f64, f64),
    params: &RefineParams,
) -> Homography {
    if params.reanchor_radius <= 0.0 || points.len() < 4 {
        return *h;
    }
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&a, &b| {
        let da = (points[a].x - anchor.0).hypot(points[a].y - anchor.1);
        let db = (points[b].x - anchor.0).hypot(points[b].y - anchor.1);
        da.partial_cmp(&db).unwrap()
    });
    let first = &points[order[0]];
    let neighbors: Vec<&ImagePoint> = order.iter().skip(1).take(12).map(|&i| &points[i]).collect();
    let (px, py) = h.apply(first.x, first.y);
    let kind = if params.same_kind {
        Some(first.kind)
    } else {
        None
    };
    let mut scratch = Vec::new();
    let mut candidates = Vec::new();
    model.grid_candidates(px, py, params.reanchor_radius, &mut candidates);
    let mut best: Option<(usize, f64, (f64, f64))> = None;
    for &c in &candidates {
        let f = &model.features[c as usize];
        if kind.is_some_and(|k| f.kind != k) || (f.x - px).hypot(f.y - py) > params.reanchor_radius
        {
            continue;
        }
        let shift = (f.x - px, f.y - py);
        let mut hits = 0;
        let mut error = 0.0;
        for n in &neighbors {
            let (x, y) = h.apply(n.x, n.y);
            let nk = if params.same_kind { Some(n.kind) } else { None };
            if let Some((_, d)) =
                model.nearest(x + shift.0, y + shift.1, params.tolerance, nk, &mut scratch)
            {
                hits += 1;
                error += d;
            }
        }
        let better = match best {
            None => true,
            Some((bh, be, _)) => hits > bh || (hits == bh && error < be),
        };
        if better {
            best = Some((hits, error, shift));
        }
    }
    match best {
        Some((hits, _, shift)) if hits >= 2 && (shift.0.abs() > 1e-9 || shift.1.abs() > 1e-9) => {
            let t = Homography([1.0, 0.0, shift.0, 0.0, 1.0, shift.1, 0.0, 0.0, 1.0]);
            h.then(&t)
        }
        _ => *h,
    }
}

/// `anchor` is the image point where `init` is most trusted; the frame's
/// spot centroid when unknown.
pub fn refine(
    points: &[ImagePoint],
    model: &ModelIndex,
    init: &Homography,
    anchor: Option<(f64, f64)>,
    params: &RefineParams,
) -> Option<Fit> {
    let anchor = anchor.unwrap_or_else(|| {
        let n = points.len().max(1) as f64;
        (
            points.iter().map(|p| p.x).sum::<f64>() / n,
            points.iter().map(|p| p.y).sum::<f64>() / n,
        )
    });
    let mut h = reanchor(points, model, init, anchor, params);
    let mut fit: Option<Fit> = None;
    let mut distances: Vec<f64> = points
        .iter()
        .map(|p| (p.x - anchor.0).hypot(p.y - anchor.1))
        .collect();
    let frame_radius = distances.iter().cloned().fold(0.0, f64::max);
    // Start with the dozen spots nearest the anchor.
    let mut radius = if distances.len() > 12 {
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distances[11].max(1.0)
    } else {
        frame_radius
    };
    let mut within: Vec<ImagePoint> = Vec::new();
    let mut within_index: Vec<usize> = Vec::new();
    let mut grown = 0;
    loop {
        within.clear();
        within_index.clear();
        for (i, p) in points.iter().enumerate() {
            if (p.x - anchor.0).hypot(p.y - anchor.1) <= radius {
                within.push(p.clone());
                within_index.push(i);
            }
        }
        let growth = if params.perspective > 0.0 {
            Some((anchor, params.perspective))
        } else {
            None
        };
        let local = associate(
            &within,
            model,
            &h,
            params.tolerance * 1.5,
            params.same_kind,
            growth,
        );
        let pairs: Vec<(usize, usize)> = local.iter().map(|&(i, f)| (within_index[i], f)).collect();
        if pairs.len() < 3 {
            break;
        }
        let Some((candidate, projective, rms)) = fit_pairs(points, model, &pairs, params) else {
            break;
        };
        h = candidate;
        fit = Some(Fit {
            h,
            pairs,
            rms,
            projective,
        });
        if radius >= frame_radius {
            break;
        }
        radius = (radius * 1.7).min(frame_radius);
        grown += 1;
        if grown > 12 {
            break;
        }
    }
    fit.as_ref()?;
    // Flat passes over everything with the tolerance shrinking to its
    // final value.
    let steps = params.iterations.max(2);
    for step in 0..steps {
        let k = step as f64 / (steps - 1) as f64;
        let t = params.tolerance * (1.5 - 0.5 * k);
        let pairs = associate(points, model, &h, t, params.same_kind, None);
        if pairs.len() < params.min_pairs.max(3) {
            break;
        }
        let Some((candidate, projective, rms)) = fit_pairs(points, model, &pairs, params) else {
            break;
        };
        h = candidate;
        fit = Some(Fit {
            h,
            pairs,
            rms,
            projective,
        });
    }
    fit.filter(|f| f.pairs.len() >= params.min_pairs)
}

/// Spot kinds for a detector polarity: bright spots are copper, dark ones
/// are holes.
pub fn kind_for_polarity(bright: bool) -> FeatureKind {
    if bright {
        FeatureKind::Island
    } else {
        FeatureKind::Hole
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Feature;
    use crate::rng::Rng;

    #[test]
    fn tightens_a_rough_similarity_into_a_homography() {
        let mut rng = Rng::new(5);
        let features: Vec<Feature> = (0..200)
            .map(|_| Feature {
                x: rng.range(0.0, 50.0),
                y: rng.range(0.0, 50.0),
                kind: FeatureKind::Island,
                area: 1.0,
                major: 1.0,
                minor: 1.0,
            })
            .collect();
        let model = ModelIndex::new(features.clone(), 6, 2.0);
        // Truth: a similarity with a little perspective.
        let base = Similarity {
            scale: 0.04,
            angle: 0.3,
            tx: 5.0,
            ty: 5.0,
        }
        .to_homography();
        let truth = Homography([
            base.0[0], base.0[1], base.0[2], base.0[3], base.0[4], base.0[5], 2e-4, -1e-4, 1.0,
        ]);
        let inv = truth.inverse().unwrap();
        let mut points = Vec::new();
        for f in &features {
            let (x, y) = inv.apply(f.x, f.y);
            if (0.0..800.0).contains(&x) && (0.0..600.0).contains(&y) {
                points.push(ImagePoint {
                    x: x + rng.normal() * 0.3,
                    y: y + rng.normal() * 0.3,
                    area: 600.0,
                    kind: FeatureKind::Island,
                });
            }
        }
        // A rough start: the similarity part only, slightly off, trusted
        // near the frame's top left.
        let rough = Similarity {
            scale: 0.041,
            angle: 0.31,
            tx: 5.3,
            ty: 4.8,
        }
        .to_homography();
        let fit = refine(
            &points,
            &model,
            &rough,
            Some((100.0, 100.0)),
            &RefineParams::default(),
        )
        .expect("fit");
        assert!(fit.projective);
        assert!(
            fit.pairs.len() >= points.len() * 9 / 10,
            "{} of {}",
            fit.pairs.len(),
            points.len()
        );
        assert!(fit.rms < 0.05, "{}", fit.rms);
        for &(x, y) in &[(10.0, 10.0), (790.0, 590.0), (400.0, 300.0)] {
            let a = fit.h.apply(x, y);
            let b = truth.apply(x, y);
            assert!((a.0 - b.0).hypot(a.1 - b.1) < 0.05, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn gives_up_without_pairs() {
        let features: Vec<Feature> = (0..10)
            .map(|i| Feature {
                x: i as f64 * 3.0,
                y: 0.0,
                kind: FeatureKind::Island,
                area: 1.0,
                major: 1.0,
                minor: 1.0,
            })
            .collect();
        let model = ModelIndex::new(features, 4, 2.0);
        let points: Vec<ImagePoint> = (0..10)
            .map(|i| ImagePoint {
                x: i as f64 * 100.0,
                y: 500.0,
                area: 1.0,
                kind: FeatureKind::Island,
            })
            .collect();
        let h = Similarity {
            scale: 0.03,
            angle: 0.0,
            tx: 100.0,
            ty: 100.0,
        }
        .to_homography();
        assert!(refine(&points, &model, &h, None, &RefineParams::default()).is_none());
    }
}
