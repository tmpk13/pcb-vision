//! Planar transforms between image pixels and board millimeters.

use nalgebra::{Matrix3, SMatrix, SymmetricEigen, Vector3};
use serde::{Deserialize, Serialize};

pub type Pair = ((f64, f64), (f64, f64));

/// Rotation, uniform scale and translation: `q = s R p + t`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Similarity {
    pub scale: f64,
    /// Radians.
    pub angle: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Similarity {
    pub fn identity() -> Similarity {
        Similarity { scale: 1.0, angle: 0.0, tx: 0.0, ty: 0.0 }
    }

    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let (s, c) = self.angle.sin_cos();
        (self.scale * (c * x - s * y) + self.tx, self.scale * (s * x + c * y) + self.ty)
    }

    pub fn inverse(&self) -> Similarity {
        let scale = 1.0 / self.scale;
        let angle = -self.angle;
        let (s, c) = angle.sin_cos();
        Similarity {
            scale,
            angle,
            tx: -scale * (c * self.tx - s * self.ty),
            ty: -scale * (s * self.tx + c * self.ty),
        }
    }

    /// The transform taking the two source points onto the two targets.
    pub fn from_two_pairs(src: [(f64, f64); 2], dst: [(f64, f64); 2]) -> Option<Similarity> {
        let dsx = src[1].0 - src[0].0;
        let dsy = src[1].1 - src[0].1;
        let ddx = dst[1].0 - dst[0].0;
        let ddy = dst[1].1 - dst[0].1;
        let ls = dsx.hypot(dsy);
        let ld = ddx.hypot(ddy);
        if ls < 1e-9 || ld < 1e-9 {
            return None;
        }
        let scale = ld / ls;
        let angle = ddy.atan2(ddx) - dsy.atan2(dsx);
        let (s, c) = angle.sin_cos();
        Some(Similarity {
            scale,
            angle,
            tx: dst[0].0 - scale * (c * src[0].0 - s * src[0].1),
            ty: dst[0].1 - scale * (s * src[0].0 + c * src[0].1),
        })
    }

    /// Least squares fit without reflection.
    pub fn fit(pairs: &[Pair]) -> Option<Similarity> {
        if pairs.len() < 2 {
            return None;
        }
        let n = pairs.len() as f64;
        let (mut px, mut py, mut qx, mut qy) = (0.0, 0.0, 0.0, 0.0);
        for ((x, y), (u, v)) in pairs {
            px += x;
            py += y;
            qx += u;
            qy += v;
        }
        px /= n;
        py /= n;
        qx /= n;
        qy /= n;
        let (mut dot, mut cross, mut norm) = (0.0, 0.0, 0.0);
        for ((x, y), (u, v)) in pairs {
            let (a, b) = (x - px, y - py);
            let (c, d) = (u - qx, v - qy);
            dot += a * c + b * d;
            cross += a * d - b * c;
            norm += a * a + b * b;
        }
        if norm < 1e-12 {
            return None;
        }
        let scale = dot.hypot(cross) / norm;
        let angle = cross.atan2(dot);
        let (s, c) = angle.sin_cos();
        Some(Similarity {
            scale,
            angle,
            tx: qx - scale * (c * px - s * py),
            ty: qy - scale * (s * px + c * py),
        })
    }

    pub fn to_homography(&self) -> Homography {
        let (s, c) = self.angle.sin_cos();
        Homography([
            self.scale * c,
            -self.scale * s,
            self.tx,
            self.scale * s,
            self.scale * c,
            self.ty,
            0.0,
            0.0,
            1.0,
        ])
    }
}

/// A projective map, row major, normalized so the last entry is one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Homography(pub [f64; 9]);

impl Homography {
    pub fn identity() -> Homography {
        Homography([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    fn matrix(&self) -> Matrix3<f64> {
        Matrix3::from_row_slice(&self.0)
    }

    fn from_matrix(m: &Matrix3<f64>) -> Homography {
        let k = m[(2, 2)];
        let k = if k.abs() < 1e-12 { 1.0 } else { k };
        let mut out = [0.0; 9];
        for r in 0..3 {
            for c in 0..3 {
                out[r * 3 + c] = m[(r, c)] / k;
            }
        }
        Homography(out)
    }

    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let h = &self.0;
        let w = h[6] * x + h[7] * y + h[8];
        let w = if w.abs() < 1e-12 { 1e-12 } else { w };
        ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)
    }

    pub fn inverse(&self) -> Option<Homography> {
        self.matrix().try_inverse().map(|m| Homography::from_matrix(&m))
    }

    /// The map that applies `self` first, then `next`.
    pub fn then(&self, next: &Homography) -> Homography {
        Homography::from_matrix(&(next.matrix() * self.matrix()))
    }

    /// The map that first scales its input by `k`, then applies `self`.
    pub fn pre_scale(&self, k: f64) -> Homography {
        let s = Matrix3::new(k, 0.0, 0.0, 0.0, k, 0.0, 0.0, 0.0, 1.0);
        Homography::from_matrix(&(self.matrix() * s))
    }

    /// The map that first shifts its input by `(dx, dy)`, then applies `self`.
    pub fn pre_translate(&self, dx: f64, dy: f64) -> Homography {
        let t = Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0);
        Homography::from_matrix(&(self.matrix() * t))
    }

    /// Direct linear transform on normalized coordinates; needs four pairs
    /// in general position. The algebraic error is then reweighted twice
    /// by the projective denominator, which brings the answer close to the
    /// least squares distance in the output frame.
    pub fn fit(pairs: &[Pair]) -> Option<Homography> {
        let mut h = Homography::fit_weighted(pairs, None)?;
        for _ in 0..2 {
            let weights: Vec<f64> = pairs
                .iter()
                .map(|((x, y), _)| {
                    let w = h.0[6] * x + h.0[7] * y + h.0[8];
                    if w.abs() < 1e-9 {
                        1.0
                    } else {
                        1.0 / w.abs()
                    }
                })
                .collect();
            match Homography::fit_weighted(pairs, Some(&weights)) {
                Some(next) => h = next,
                None => break,
            }
        }
        Some(h)
    }

    fn fit_weighted(pairs: &[Pair], weights: Option<&[f64]>) -> Option<Homography> {
        if pairs.len() < 4 {
            return None;
        }
        let (t_src, src) = normalize(pairs.iter().map(|p| p.0));
        let (t_dst, dst) = normalize(pairs.iter().map(|p| p.1));
        let mut ata = SMatrix::<f64, 9, 9>::zeros();
        for (k, (&(x, y), &(u, v))) in src.iter().zip(&dst).enumerate() {
            let w = weights.map_or(1.0, |ws| ws[k]);
            let rows = [
                [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u],
                [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v],
            ];
            for row in &rows {
                for i in 0..9 {
                    for j in 0..9 {
                        ata[(i, j)] += w * w * row[i] * row[j];
                    }
                }
            }
        }
        let eig = SymmetricEigen::new(ata);
        let mut best = 0;
        for i in 1..9 {
            if eig.eigenvalues[i] < eig.eigenvalues[best] {
                best = i;
            }
        }
        let h = eig.eigenvectors.column(best);
        let hn = Matrix3::from_row_slice(h.as_slice());
        let t_dst_inv = t_dst.try_inverse()?;
        let m = t_dst_inv * hn * t_src;
        if m[(2, 2)].abs() < 1e-12 || !m.iter().all(|v| v.is_finite()) {
            return None;
        }
        Some(Homography::from_matrix(&m))
    }

    /// First derivative of the map at a point: `[[du/dx, du/dy], [dv/dx, dv/dy]]`.
    pub fn jacobian(&self, x: f64, y: f64) -> [[f64; 2]; 2] {
        let h = &self.0;
        let w = h[6] * x + h[7] * y + h[8];
        let u = h[0] * x + h[1] * y + h[2];
        let v = h[3] * x + h[4] * y + h[5];
        let w2 = w * w;
        [
            [(h[0] * w - u * h[6]) / w2, (h[1] * w - u * h[7]) / w2],
            [(h[3] * w - v * h[6]) / w2, (h[4] * w - v * h[7]) / w2],
        ]
    }

    /// Output units per input unit at a point.
    pub fn local_scale(&self, x: f64, y: f64) -> f64 {
        let j = self.jacobian(x, y);
        (j[0][0] * j[1][1] - j[0][1] * j[1][0]).abs().sqrt()
    }

    /// Direction, in the output, of the input's x axis at a point; radians.
    pub fn local_angle(&self, x: f64, y: f64) -> f64 {
        let j = self.jacobian(x, y);
        j[1][0].atan2(j[0][0])
    }
}

fn normalize(points: impl Iterator<Item = (f64, f64)>) -> (Matrix3<f64>, Vec<(f64, f64)>) {
    let pts: Vec<(f64, f64)> = points.collect();
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let mean_dist = pts.iter().map(|p| (p.0 - cx).hypot(p.1 - cy)).sum::<f64>() / n;
    let s = if mean_dist < 1e-12 { 1.0 } else { std::f64::consts::SQRT_2 / mean_dist };
    let t = Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0);
    let out = pts
        .iter()
        .map(|p| {
            let v = t * Vector3::new(p.0, p.1, 1.0);
            (v[0], v[1])
        })
        .collect();
    (t, out)
}

/// Least squares affine map as a homography with no projective part; needs
/// three pairs not on a line.
pub fn fit_affine(pairs: &[Pair]) -> Option<Homography> {
    if pairs.len() < 3 {
        return None;
    }
    // Normal equations for u = a x + b y + c and v = d x + e y + f.
    let mut m = Matrix3::zeros();
    let mut ru = Vector3::zeros();
    let mut rv = Vector3::zeros();
    for ((x, y), (u, v)) in pairs {
        let row = Vector3::new(*x, *y, 1.0);
        m += row * row.transpose();
        ru += row * *u;
        rv += row * *v;
    }
    let inv = m.try_inverse()?;
    let a = inv * ru;
    let d = inv * rv;
    let h = Homography([a[0], a[1], a[2], d[0], d[1], d[2], 0.0, 0.0, 1.0]);
    h.0.iter().all(|v| v.is_finite()).then_some(h)
}

/// Root mean square distance between mapped sources and their targets.
pub fn rms_error(h: &Homography, pairs: &[Pair]) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let sum: f64 = pairs
        .iter()
        .map(|((x, y), (u, v))| {
            let (px, py) = h.apply(*x, *y);
            (px - u).powi(2) + (py - v).powi(2)
        })
        .sum();
    (sum / pairs.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: (f64, f64), b: (f64, f64)) -> bool {
        (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6
    }

    #[test]
    fn similarity_round_trip_and_fit() {
        let s = Similarity { scale: 0.05, angle: 0.7, tx: 12.0, ty: -3.0 };
        let p = (100.0, 200.0);
        let q = s.apply(p.0, p.1);
        assert!(close(s.inverse().apply(q.0, q.1), p));
        let pts = [(10.0, 20.0), (300.0, 40.0), (150.0, 400.0), (0.0, 0.0)];
        let pairs: Vec<Pair> = pts.iter().map(|&p| (p, s.apply(p.0, p.1))).collect();
        let f = Similarity::fit(&pairs).unwrap();
        assert!((f.scale - s.scale).abs() < 1e-9 && (f.angle - s.angle).abs() < 1e-9);
        assert!(close((f.tx, f.ty), (s.tx, s.ty)));
        let two = Similarity::from_two_pairs([pts[0], pts[1]], [pairs[0].1, pairs[1].1]).unwrap();
        assert!(close(two.apply(pts[2].0, pts[2].1), pairs[2].1));
        assert!(close(s.to_homography().apply(p.0, p.1), q));
    }

    #[test]
    fn homography_fit_recovers_a_projective_map() {
        let truth = Homography([0.02, 0.003, 5.0, -0.002, 0.021, 7.0, 1e-4, -5e-5, 1.0]);
        let pts: Vec<(f64, f64)> =
            (0..12).map(|i| (((i * 37) % 11) as f64 * 60.0, ((i * 53) % 7) as f64 * 70.0)).collect();
        let pairs: Vec<Pair> = pts.iter().map(|&p| (p, truth.apply(p.0, p.1))).collect();
        let h = Homography::fit(&pairs).unwrap();
        assert!(rms_error(&h, &pairs) < 1e-6);
        for p in &pts {
            assert!(close(h.apply(p.0, p.1), truth.apply(p.0, p.1)));
        }
        let inv = h.inverse().unwrap();
        let q = h.apply(100.0, 100.0);
        assert!(close(inv.apply(q.0, q.1), (100.0, 100.0)));
        assert!(close(h.then(&inv).apply(50.0, 60.0), (50.0, 60.0)));
        let scaled = h.pre_scale(0.5);
        assert!(close(scaled.apply(200.0, 200.0), h.apply(100.0, 100.0)));
        let shifted = h.pre_translate(3.0, 4.0);
        assert!(close(shifted.apply(1.0, 1.0), h.apply(4.0, 5.0)));
    }

    #[test]
    fn affine_fit() {
        let truth = Homography([0.03, 0.01, 2.0, -0.005, 0.04, 3.0, 0.0, 0.0, 1.0]);
        let pts = [(0.0, 0.0), (100.0, 10.0), (20.0, 300.0), (400.0, 400.0)];
        let pairs: Vec<Pair> = pts.iter().map(|&p| (p, truth.apply(p.0, p.1))).collect();
        let h = fit_affine(&pairs).unwrap();
        assert!(rms_error(&h, &pairs) < 1e-9);
        assert!(fit_affine(&[((0.0, 0.0), (0.0, 0.0)), ((1.0, 1.0), (1.0, 1.0)), ((2.0, 2.0), (2.0, 2.0))]).is_none());
    }

    #[test]
    fn local_scale_and_angle() {
        let s = Similarity { scale: 0.1, angle: 0.5, tx: 1.0, ty: 2.0 }.to_homography();
        assert!((s.local_scale(10.0, 10.0) - 0.1).abs() < 1e-9);
        assert!((s.local_angle(10.0, 10.0) - 0.5).abs() < 1e-9);
        assert!(Homography::fit(&[((0.0, 0.0), (0.0, 0.0)); 3]).is_none());
    }
}
