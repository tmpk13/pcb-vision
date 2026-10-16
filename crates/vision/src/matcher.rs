//! Finding where a set of detected spots lies on the model's constellation
//! when nothing is known yet.
//!
//! Two spots and two model features fix a similarity between the image and
//! the board. Candidate pairs come from each spot's nearest neighbors and
//! from model pairs of a compatible length. A hypothesis first has to put
//! the model's neighbors of its feature onto spots, then a spread of spots
//! onto features, and then it is scored by how many spots land on a
//! feature, weighted by closeness. Perspective is allowed for by widening
//! the tolerance with distance from the hypothesis.

use serde::{Deserialize, Serialize};

use crate::model::{Feature, FeatureKind};
use crate::xform::Similarity;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImagePoint {
    pub x: f64,
    pub y: f64,
    /// px^2.
    pub area: f64,
    pub kind: FeatureKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchParams {
    /// Allowed millimeters per pixel, low and high.
    pub scale_range: (f64, f64),
    /// A mapped spot within this distance of a model feature matches, mm.
    pub tolerance: f64,
    /// Extra tolerance per millimeter of distance from the hypothesis pair.
    pub perspective: f64,
    /// Nearest neighbors each spot pairs with.
    pub neighbors: usize,
    pub min_inliers: usize,
    /// Matched fraction of the spots that ends the search outright.
    pub early_exit: f64,
    /// Weighted score from which a hypothesis that beats every earlier one
    /// is handed to the verifier.
    pub candidate_score: f64,
    /// Verifications allowed per search.
    pub max_verifications: usize,
    pub max_hypotheses: usize,
    /// Allowed magnitude of the log of the area ratio between a spot, in
    /// mm^2 under the hypothesis, and its feature.
    pub area_tolerance: Option<f64>,
    /// Only pair spots with features of the same kind.
    pub same_kind: bool,
}

impl Default for MatchParams {
    fn default() -> MatchParams {
        MatchParams {
            scale_range: (0.005, 0.5),
            tolerance: 0.4,
            perspective: 0.03,
            neighbors: 8,
            min_inliers: 8,
            early_exit: 0.7,
            candidate_score: 16.0,
            max_verifications: 60,
            max_hypotheses: 400_000,
            area_tolerance: Some(1.1),
            same_kind: true,
        }
    }
}

/// Points binned into square cells for neighborhood lookups.
pub struct Grid {
    cell: f64,
    min_x: f64,
    min_y: f64,
    cols: usize,
    rows: usize,
    bins: Vec<Vec<u32>>,
}

impl Grid {
    pub fn new(points: &[(f64, f64)], cell: f64) -> Grid {
        let cell = cell.max(1e-6);
        if points.is_empty() {
            return Grid {
                cell,
                min_x: 0.0,
                min_y: 0.0,
                cols: 1,
                rows: 1,
                bins: vec![Vec::new()],
            };
        }
        let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let cols = (((max_x - min_x) / cell).floor() as usize + 1).clamp(1, 4096);
        let rows = (((max_y - min_y) / cell).floor() as usize + 1).clamp(1, 4096);
        let mut bins = vec![Vec::new(); cols * rows];
        for (i, p) in points.iter().enumerate() {
            let c = (((p.0 - min_x) / cell).floor() as usize).min(cols - 1);
            let r = (((p.1 - min_y) / cell).floor() as usize).min(rows - 1);
            bins[r * cols + c].push(i as u32);
        }
        Grid {
            cell,
            min_x,
            min_y,
            cols,
            rows,
            bins,
        }
    }

    /// Indices of the points in the cells that a circle of `radius` at
    /// `(x, y)` can reach.
    pub fn candidates(&self, x: f64, y: f64, radius: f64, out: &mut Vec<u32>) {
        out.clear();
        let c0 = ((x - radius - self.min_x) / self.cell).floor();
        let c1 = ((x + radius - self.min_x) / self.cell).floor();
        let r0 = ((y - radius - self.min_y) / self.cell).floor();
        let r1 = ((y + radius - self.min_y) / self.cell).floor();
        if c1 < 0.0 || r1 < 0.0 || c0 >= self.cols as f64 || r0 >= self.rows as f64 {
            return;
        }
        let c0 = c0.max(0.0) as usize;
        let r0 = r0.max(0.0) as usize;
        let c1 = (c1 as usize).min(self.cols - 1);
        let r1 = (r1 as usize).min(self.rows - 1);
        for r in r0..=r1 {
            for c in c0..=c1 {
                out.extend_from_slice(&self.bins[r * self.cols + c]);
            }
        }
    }

    /// The nearest of `points` within `radius`, with its distance.
    pub fn nearest(
        &self,
        points: &[(f64, f64)],
        x: f64,
        y: f64,
        radius: f64,
        scratch: &mut Vec<u32>,
    ) -> Option<(usize, f64)> {
        self.candidates(x, y, radius, scratch);
        let mut best: Option<(usize, f64)> = None;
        for &i in scratch.iter() {
            let p = points[i as usize];
            let d = (p.0 - x).hypot(p.1 - y);
            if d <= radius && best.is_none_or(|b| d < b.1) {
                best = Some((i as usize, d));
            }
        }
        best
    }
}

/// For each point, the indices of its `k` nearest others, nearest first.
pub fn nearest_neighbors(points: &[(f64, f64)], k: usize) -> Vec<Vec<usize>> {
    let n = points.len();
    let mut out = Vec::with_capacity(n);
    let mut dists: Vec<(f64, usize)> = Vec::with_capacity(n);
    for (i, p) in points.iter().enumerate() {
        dists.clear();
        for (j, q) in points.iter().enumerate() {
            if j != i {
                dists.push(((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2), j));
            }
        }
        let k = k.min(dists.len());
        if k > 0 {
            dists.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            dists.truncate(k);
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        } else {
            dists.clear();
        }
        out.push(dists.iter().map(|d| d.1).collect());
    }
    out
}

/// Model features arranged for fast lookups.
pub struct ModelIndex {
    pub features: Vec<Feature>,
    positions: Vec<(f64, f64)>,
    grid: Grid,
    /// Each feature's nearest neighbors.
    neighbors: Vec<Vec<usize>>,
    /// Sorted by length.
    pairs: Vec<ModelPair>,
}

#[derive(Clone, Copy)]
struct ModelPair {
    a: u32,
    b: u32,
    dist: f64,
}

impl ModelIndex {
    pub fn new(features: Vec<Feature>, neighbors: usize, cell: f64) -> ModelIndex {
        let positions: Vec<(f64, f64)> = features.iter().map(|f| (f.x, f.y)).collect();
        let grid = Grid::new(&positions, cell);
        let knn = nearest_neighbors(&positions, neighbors);
        let mut pairs = Vec::new();
        for (a, list) in knn.iter().enumerate() {
            for &b in list {
                if a < b || !knn[b].contains(&a) {
                    let dist =
                        (positions[a].0 - positions[b].0).hypot(positions[a].1 - positions[b].1);
                    pairs.push(ModelPair {
                        a: a as u32,
                        b: b as u32,
                        dist,
                    });
                }
            }
        }
        pairs.sort_by(|p, q| p.dist.partial_cmp(&q.dist).unwrap());
        ModelIndex {
            features,
            positions,
            grid,
            neighbors: knn,
            pairs,
        }
    }

    /// The nearest feature within `radius` of a view frame point, with its
    /// distance; `kind` restricts the search.
    pub fn nearest(
        &self,
        x: f64,
        y: f64,
        radius: f64,
        kind: Option<FeatureKind>,
        scratch: &mut Vec<u32>,
    ) -> Option<(usize, f64)> {
        self.grid.candidates(x, y, radius, scratch);
        let mut best: Option<(usize, f64)> = None;
        for &i in scratch.iter() {
            let f = &self.features[i as usize];
            if kind.is_some_and(|k| f.kind != k) {
                continue;
            }
            let d = (f.x - x).hypot(f.y - y);
            if d <= radius && best.is_none_or(|b| d < b.1) {
                best = Some((i as usize, d));
            }
        }
        best
    }

    /// The two nearest features within `radius`, nearest first.
    pub fn nearest_two(
        &self,
        x: f64,
        y: f64,
        radius: f64,
        kind: Option<FeatureKind>,
        scratch: &mut Vec<u32>,
    ) -> (Option<(usize, f64)>, Option<f64>) {
        self.grid.candidates(x, y, radius, scratch);
        let mut best: Option<(usize, f64)> = None;
        let mut second: Option<f64> = None;
        for &i in scratch.iter() {
            let f = &self.features[i as usize];
            if kind.is_some_and(|k| f.kind != k) {
                continue;
            }
            let d = (f.x - x).hypot(f.y - y);
            if d > radius {
                continue;
            }
            match best {
                Some(b) if d >= b.1 => {
                    if second.is_none_or(|s| d < s) {
                        second = Some(d);
                    }
                }
                Some(b) => {
                    second = Some(b.1);
                    best = Some((i as usize, d));
                }
                None => best = Some((i as usize, d)),
            }
        }
        (best, second)
    }

    /// Feature indices in the cells a circle can reach.
    pub fn grid_candidates(&self, x: f64, y: f64, radius: f64, out: &mut Vec<u32>) {
        self.grid.candidates(x, y, radius, out);
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchResult {
    /// Image pixels to view frame millimeters.
    pub transform: Similarity,
    /// `(spot index, feature index)` for the spots that landed on a feature.
    pub pairs: Vec<(usize, usize)>,
    /// Hits weighted by closeness: a spot right on its feature counts one,
    /// one at the edge of the tolerance nothing.
    pub score: f64,
    /// Image point between the two spots the hypothesis came from, where
    /// the similarity is exact.
    pub anchor: (f64, f64),
    pub hypotheses: usize,
}

/// Scoring state reused across hypotheses.
struct Scorer<'a> {
    points: &'a [ImagePoint],
    model: &'a ModelIndex,
    params: &'a MatchParams,
    scratch: Vec<u32>,
    /// Generation stamp per feature, so one feature counts once per
    /// hypothesis.
    used: Vec<u32>,
    generation: u32,
}

impl Scorer<'_> {
    fn kind(&self, p: &ImagePoint) -> Option<FeatureKind> {
        if self.params.same_kind {
            Some(p.kind)
        } else {
            None
        }
    }

    fn tolerance(&self, s: &Similarity, p: &ImagePoint, mid: (f64, f64)) -> f64 {
        let d_mm = (p.x - mid.0).hypot(p.y - mid.1) * s.scale;
        self.params.tolerance + self.params.perspective * d_mm
    }

    /// The unused feature the spot lands on under the hypothesis, and the
    /// weight of the hit: one when exactly on it, zero at the edge of the
    /// tolerance. Chance hits spread over the tolerance disc and average
    /// half, so the weighted score separates a real match from clutter
    /// better than a count does.
    fn hit(&mut self, s: &Similarity, p: &ImagePoint, mid: (f64, f64)) -> Option<(usize, f64)> {
        let (x, y) = s.apply(p.x, p.y);
        let tol = self.tolerance(s, p, mid);
        let kind = self.kind(p);
        let (idx, d) = self.model.nearest(x, y, tol, kind, &mut self.scratch)?;
        if self.used[idx] == self.generation {
            return None;
        }
        self.used[idx] = self.generation;
        Some((idx, 1.0 - (d / tol).powi(2)))
    }

    fn begin(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.used.iter_mut().for_each(|u| *u = 0);
            self.generation = 1;
        }
    }

    /// Hits among `subset` (all spots when `None`): the count and the
    /// weighted score.
    fn count(&mut self, s: &Similarity, mid: (f64, f64), subset: Option<&[usize]>) -> (usize, f64) {
        let mut n = 0;
        let mut score = 0.0;
        let total = self.points.len();
        let mut index = 0;
        loop {
            let i = match subset {
                Some(list) => match list.get(index) {
                    Some(&i) => i,
                    None => break,
                },
                None => {
                    if index >= total {
                        break;
                    }
                    index
                }
            };
            index += 1;
            if let Some((_, w)) = self.hit(s, &self.points[i], mid) {
                n += 1;
                score += w;
            }
        }
        (n, score)
    }

    fn pairs(&mut self, s: &Similarity, mid: (f64, f64)) -> Vec<(usize, usize)> {
        self.begin();
        let mut out = Vec::new();
        for i in 0..self.points.len() {
            if let Some((f, _)) = self.hit(s, &self.points[i], mid) {
                out.push((i, f));
            }
        }
        out
    }
}

fn area_compatible(params: &MatchParams, p: &ImagePoint, f: &Feature, scale: f64) -> bool {
    match params.area_tolerance {
        None => true,
        Some(tol) => {
            let mm2 = p.area * scale * scale;
            if mm2 <= 0.0 || f.area <= 0.0 {
                return false;
            }
            (mm2 / f.area).ln().abs() <= tol
        }
    }
}

/// The search without a verifier: the best hypothesis by score.
pub fn match_constellation(
    points: &[ImagePoint],
    model: &ModelIndex,
    params: &MatchParams,
) -> Option<MatchResult> {
    match_constellation_with(points, model, params, |_| false)
}

/// The search with a verifier: a hypothesis that beats every earlier one
/// and reaches the candidate score is handed to `accept`, and the search
/// ends on the first one it takes. Otherwise the best hypothesis by score
/// is returned at the end, verified or not.
pub fn match_constellation_with(
    points: &[ImagePoint],
    model: &ModelIndex,
    params: &MatchParams,
    mut accept: impl FnMut(&MatchResult) -> bool,
) -> Option<MatchResult> {
    if points.len() < 3 || model.len() < 3 {
        return None;
    }
    let positions: Vec<(f64, f64)> = points.iter().map(|p| (p.x, p.y)).collect();
    let knn = nearest_neighbors(&positions, params.neighbors);
    let mut image_pairs: Vec<(usize, usize)> = Vec::new();
    for (a, list) in knn.iter().enumerate() {
        for &b in list {
            if a < b || !knn[b].contains(&a) {
                image_pairs.push((a, b));
            }
        }
    }
    // Spots binned for the model-driven gate, at a cell near the spacing.
    let span = positions
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max)
        - positions.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let spot_grid = Grid::new(&positions, (span / 40.0).max(2.0));
    // A sample of spots for the middle stage of verification.
    let sample: Vec<usize> = {
        let step = (points.len() / 32).max(1);
        (0..points.len()).step_by(step).take(32).collect()
    };
    let mut scorer = Scorer {
        points,
        model,
        params,
        scratch: Vec::new(),
        used: vec![0; model.len()],
        generation: 0,
    };
    let mut gate_scratch = Vec::new();
    let target = (points.len() as f64 * params.early_exit).max(params.min_inliers as f64);
    let candidate = params.candidate_score.max(params.min_inliers as f64);
    let mut verifications = 0usize;
    let mut best: Option<(f64, Similarity, (f64, f64))> = None;
    let mut hypotheses = 0usize;
    'search: for &(i, j) in &image_pairs {
        let (pi, pj) = (&points[i], &points[j]);
        let d_img = (pi.x - pj.x).hypot(pi.y - pj.y);
        if d_img < 1e-6 {
            continue;
        }
        let lo = d_img * params.scale_range.0;
        let hi = d_img * params.scale_range.1;
        let start = model.pairs.partition_point(|p| p.dist < lo);
        let end = model.pairs.partition_point(|p| p.dist <= hi);
        let mid = ((pi.x + pj.x) / 2.0, (pi.y + pj.y) / 2.0);
        for mp in &model.pairs[start..end] {
            for (fa, fb) in [
                (mp.a as usize, mp.b as usize),
                (mp.b as usize, mp.a as usize),
            ] {
                let (fi, fj) = (&model.features[fa], &model.features[fb]);
                if params.same_kind && (fi.kind != pi.kind || fj.kind != pj.kind) {
                    continue;
                }
                let scale = mp.dist / d_img;
                if !area_compatible(params, pi, fi, scale)
                    || !area_compatible(params, pj, fj, scale)
                {
                    continue;
                }
                hypotheses += 1;
                if hypotheses > params.max_hypotheses {
                    break 'search;
                }
                let Some(s) = Similarity::from_two_pairs(
                    [(pi.x, pi.y), (pj.x, pj.y)],
                    [(fi.x, fi.y), (fj.x, fj.y)],
                ) else {
                    continue;
                };
                // Stage one: the model's neighbors of the first feature
                // have to land on spots. Clutter adds spots but does not
                // take pads away, so this gate holds up on busy pictures.
                let inv = s.inverse();
                let tol_px = params.tolerance / s.scale;
                let mut need = 0;
                let mut hits = 0;
                for &n in model.neighbors[fa].iter().filter(|&&n| n != fb).take(6) {
                    need += 1;
                    let (x, y) = inv.apply(model.positions[n].0, model.positions[n].1);
                    if spot_grid
                        .nearest(&positions, x, y, tol_px, &mut gate_scratch)
                        .is_some()
                    {
                        hits += 1;
                    }
                }
                if need >= 2 && hits * 2 < need {
                    continue;
                }
                // Stage two: a spread of spots across the frame.
                scorer.begin();
                scorer.used[fa] = scorer.generation;
                scorer.used[fb] = scorer.generation;
                if sample.len() >= 8 {
                    let (hits, _) = scorer.count(&s, mid, Some(&sample));
                    if hits * 5 < sample.len() {
                        continue;
                    }
                }
                // Stage three: everything.
                scorer.begin();
                let (_, score) = scorer.count(&s, mid, None);
                let improved = best.as_ref().is_none_or(|b| score > b.0);
                if improved {
                    best = Some((score, s, mid));
                    if score >= target {
                        break 'search;
                    }
                    if score >= candidate && verifications < params.max_verifications {
                        verifications += 1;
                        let pairs = scorer.pairs(&s, mid);
                        let result = MatchResult {
                            transform: s,
                            pairs,
                            score,
                            anchor: mid,
                            hypotheses,
                        };
                        if accept(&result) {
                            return Some(result);
                        }
                    }
                }
            }
        }
    }
    let (score, s, mid) = best?;
    if score < params.min_inliers as f64 {
        return None;
    }
    let pairs = scorer.pairs(&s, mid);
    Some(MatchResult {
        transform: s,
        pairs,
        score,
        anchor: mid,
        hypotheses,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn model_features(n: usize, rng: &mut Rng) -> Vec<Feature> {
        (0..n)
            .map(|_| Feature {
                x: rng.range(0.0, 60.0),
                y: rng.range(0.0, 40.0),
                kind: FeatureKind::Island,
                area: rng.range(0.3, 2.0),
                major: 1.0,
                minor: 1.0,
            })
            .collect()
    }

    #[test]
    fn recovers_a_similarity_with_clutter_and_missing_points() {
        let mut rng = Rng::new(3);
        let features = model_features(150, &mut rng);
        let index = ModelIndex::new(features.clone(), 8, 2.0);
        // Camera: 0.05 mm per px, turned 40 degrees, looking at a window.
        let truth = Similarity {
            scale: 0.05,
            angle: 0.7,
            tx: 20.0,
            ty: 10.0,
        };
        let inv = truth.inverse();
        let mut points = Vec::new();
        for f in features.iter() {
            let (x, y) = inv.apply(f.x, f.y);
            if x < 0.0 || y < 0.0 || x > 640.0 || y > 480.0 || rng.unit() < 0.2 {
                continue;
            }
            points.push(ImagePoint {
                x: x + rng.normal() * 0.5,
                y: y + rng.normal() * 0.5,
                area: f.area / (0.05 * 0.05) * rng.range(0.8, 1.25),
                kind: FeatureKind::Island,
            });
        }
        let seen = points.len();
        for _ in 0..seen {
            points.push(ImagePoint {
                x: rng.range(0.0, 640.0),
                y: rng.range(0.0, 480.0),
                area: 300.0,
                kind: FeatureKind::Island,
            });
        }
        let params = MatchParams {
            scale_range: (0.02, 0.1),
            early_exit: 0.4,
            ..MatchParams::default()
        };
        let result = match_constellation(&points, &index, &params).expect("a match");
        assert!(
            result.pairs.len() >= seen * 8 / 10,
            "{} of {seen}",
            result.pairs.len()
        );
        let s = result.transform;
        assert!((s.scale - 0.05).abs() < 0.002, "{s:?}");
        assert!((s.angle - 0.7).abs() < 0.02, "{s:?}");
        for &(i, f) in &result.pairs {
            let (x, y) = s.apply(points[i].x, points[i].y);
            assert!((x - features[f].x).hypot(y - features[f].y) < 0.6);
        }
    }

    #[test]
    fn verifier_ends_the_search() {
        let mut rng = Rng::new(4);
        let features = model_features(150, &mut rng);
        let index = ModelIndex::new(features.clone(), 8, 2.0);
        let truth = Similarity {
            scale: 0.05,
            angle: -0.4,
            tx: 10.0,
            ty: 5.0,
        };
        let inv = truth.inverse();
        let points: Vec<ImagePoint> = features
            .iter()
            .filter_map(|f| {
                let (x, y) = inv.apply(f.x, f.y);
                ((0.0..640.0).contains(&x) && (0.0..480.0).contains(&y)).then(|| ImagePoint {
                    x,
                    y,
                    area: f.area / 0.0025,
                    kind: FeatureKind::Island,
                })
            })
            .collect();
        let params = MatchParams {
            scale_range: (0.02, 0.1),
            early_exit: 2.0,
            ..MatchParams::default()
        };
        let mut offered = 0;
        let result = match_constellation_with(&points, &index, &params, |m| {
            offered += 1;
            m.pairs.len() > points.len() / 2
        })
        .unwrap();
        assert!(offered >= 1);
        assert!(result.pairs.len() > points.len() / 2);
        assert!(result.hypotheses < params.max_hypotheses);
    }

    #[test]
    fn no_match_in_noise() {
        let mut rng = Rng::new(11);
        let features = model_features(120, &mut rng);
        let index = ModelIndex::new(features, 8, 2.0);
        let points: Vec<ImagePoint> = (0..40)
            .map(|_| ImagePoint {
                x: rng.range(0.0, 640.0),
                y: rng.range(0.0, 480.0),
                area: 200.0,
                kind: FeatureKind::Island,
            })
            .collect();
        let params = MatchParams {
            scale_range: (0.02, 0.1),
            min_inliers: 12,
            max_hypotheses: 50_000,
            ..MatchParams::default()
        };
        let result = match_constellation(&points, &index, &params);
        assert!(
            result.is_none(),
            "{:?}",
            result.map(|r| (r.pairs.len(), r.score))
        );
    }

    #[test]
    fn grid_lookups() {
        let features = vec![
            Feature {
                x: 0.0,
                y: 0.0,
                kind: FeatureKind::Island,
                area: 1.0,
                major: 1.0,
                minor: 1.0,
            },
            Feature {
                x: 5.0,
                y: 0.0,
                kind: FeatureKind::Hole,
                area: 1.0,
                major: 1.0,
                minor: 1.0,
            },
            Feature {
                x: 5.5,
                y: 0.0,
                kind: FeatureKind::Island,
                area: 1.0,
                major: 1.0,
                minor: 1.0,
            },
        ];
        let index = ModelIndex::new(features, 2, 1.0);
        let mut scratch = Vec::new();
        assert_eq!(
            index
                .nearest(5.1, 0.1, 1.0, None, &mut scratch)
                .map(|r| r.0),
            Some(1)
        );
        assert_eq!(
            index
                .nearest(5.1, 0.1, 1.0, Some(FeatureKind::Island), &mut scratch)
                .map(|r| r.0),
            Some(2)
        );
        assert!(index.nearest(20.0, 20.0, 1.0, None, &mut scratch).is_none());
        let (best, second) = index.nearest_two(5.1, 0.0, 1.0, None, &mut scratch);
        assert_eq!(best.map(|b| b.0), Some(1));
        assert!((second.unwrap() - 0.4).abs() < 1e-9);
        let grid = Grid::new(&[(0.0, 0.0), (3.0, 4.0)], 1.0);
        assert_eq!(
            grid.nearest(&[(0.0, 0.0), (3.0, 4.0)], 2.5, 3.5, 1.0, &mut scratch)
                .map(|r| r.0),
            Some(1)
        );
    }
}
