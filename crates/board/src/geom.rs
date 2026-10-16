//! Points, rectangles and arcs in the board frame: millimeters, x to the
//! right and y down, as KiCad stores them.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const ZERO: Point = Point { x: 0.0, y: 0.0 };

    pub fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(self, o: Point) -> Point {
        Point::new(self.x + o.x, self.y + o.y)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, o: Point) -> Point {
        Point::new(self.x - o.x, self.y - o.y)
    }

    pub fn scale(self, k: f64) -> Point {
        Point::new(self.x * k, self.y * k)
    }

    pub fn dist(self, o: Point) -> f64 {
        self.sub(o).norm()
    }

    pub fn norm(self) -> f64 {
        self.x.hypot(self.y)
    }

    pub fn dot(self, o: Point) -> f64 {
        self.x * o.x + self.y * o.y
    }

    pub fn cross(self, o: Point) -> f64 {
        self.x * o.y - self.y * o.x
    }

    /// Rotate about the origin by `deg` the way KiCad does: a positive angle
    /// turns counterclockwise as seen on screen, where y points down.
    pub fn rotate_deg(self, deg: f64) -> Point {
        if deg == 0.0 {
            return self;
        }
        let (s, c) = deg.to_radians().sin_cos();
        Point::new(self.x * c + self.y * s, -self.x * s + self.y * c)
    }

    pub fn rotate_about(self, center: Point, deg: f64) -> Point {
        self.sub(center).rotate_deg(deg).add(center)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub min: Point,
    pub max: Point,
}

impl Rect {
    pub fn from_points<'a>(points: impl IntoIterator<Item = &'a Point>) -> Option<Rect> {
        let mut rect: Option<Rect> = None;
        for p in points {
            rect = Some(match rect {
                None => Rect { min: *p, max: *p },
                Some(r) => r.include(*p),
            });
        }
        rect
    }

    pub fn include(self, p: Point) -> Rect {
        Rect {
            min: Point::new(self.min.x.min(p.x), self.min.y.min(p.y)),
            max: Point::new(self.max.x.max(p.x), self.max.y.max(p.y)),
        }
    }

    pub fn union(self, o: Rect) -> Rect {
        self.include(o.min).include(o.max)
    }

    pub fn expand(self, by: f64) -> Rect {
        Rect {
            min: Point::new(self.min.x - by, self.min.y - by),
            max: Point::new(self.max.x + by, self.max.y + by),
        }
    }

    pub fn width(self) -> f64 {
        self.max.x - self.min.x
    }

    pub fn height(self) -> f64 {
        self.max.y - self.min.y
    }

    pub fn center(self) -> Point {
        Point::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
        )
    }

    pub fn contains(self, p: Point) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }
}

/// A circular arc through three points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcGeom {
    pub center: Point,
    pub radius: f64,
    /// Angle of the start point about the center, radians, in the y down
    /// frame (atan2 of the raw offsets).
    pub start_angle: f64,
    /// Signed sweep in radians, positive for increasing atan2 angle.
    pub sweep: f64,
}

/// The circle through `start`, `mid` and `end`, with the sweep that passes
/// through `mid`. `None` when the points are collinear.
pub fn arc_from_points(start: Point, mid: Point, end: Point) -> Option<ArcGeom> {
    let ax = start.x;
    let ay = start.y;
    let bx = mid.x;
    let by = mid.y;
    let cx = end.x;
    let cy = end.y;
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-12 {
        return None;
    }
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    let center = Point::new(ux, uy);
    let radius = center.dist(start);
    let start_angle = (ay - uy).atan2(ax - ux);
    let mid_angle = (by - uy).atan2(bx - ux);
    let end_angle = (cy - uy).atan2(cx - ux);
    let two_pi = std::f64::consts::TAU;
    // Walk counterclockwise in atan2 terms from start to end and see whether
    // mid lies on that way; otherwise the arc goes the other way round.
    let ccw_to_end = (end_angle - start_angle).rem_euclid(two_pi);
    let ccw_to_mid = (mid_angle - start_angle).rem_euclid(two_pi);
    let sweep = if ccw_to_mid <= ccw_to_end {
        ccw_to_end
    } else {
        ccw_to_end - two_pi
    };
    Some(ArcGeom {
        center,
        radius,
        start_angle,
        sweep,
    })
}

/// Points along the arc from start to end, the ends included, spaced so no
/// chord deviates from the circle by more than `tolerance`.
pub fn arc_points(start: Point, mid: Point, end: Point, tolerance: f64) -> Vec<Point> {
    let Some(arc) = arc_from_points(start, mid, end) else {
        return vec![start, mid, end];
    };
    let tolerance = tolerance.max(1e-6);
    let step = if arc.radius <= tolerance {
        std::f64::consts::PI
    } else {
        2.0 * (1.0 - tolerance / arc.radius).acos()
    };
    let n = ((arc.sweep.abs() / step).ceil() as usize).clamp(1, 4096);
    let mut out = Vec::with_capacity(n + 1);
    out.push(start);
    for i in 1..n {
        let a = arc.start_angle + arc.sweep * (i as f64 / n as f64);
        out.push(Point::new(
            arc.center.x + arc.radius * a.cos(),
            arc.center.y + arc.radius * a.sin(),
        ));
    }
    out.push(end);
    out
}

/// Points along a cubic Bezier, the ends included.
pub fn bezier_points(p0: Point, p1: Point, p2: Point, p3: Point, segments: usize) -> Vec<Point> {
    let n = segments.max(1);
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let u = 1.0 - t;
            let w0 = u * u * u;
            let w1 = 3.0 * u * u * t;
            let w2 = 3.0 * u * t * t;
            let w3 = t * t * t;
            Point::new(
                w0 * p0.x + w1 * p1.x + w2 * p2.x + w3 * p3.x,
                w0 * p0.y + w1 * p1.y + w2 * p2.y + w3 * p3.y,
            )
        })
        .collect()
}

/// Signed area of a closed polygon (shoelace); positive when the vertices
/// run clockwise on screen in the y down frame.
pub fn polygon_area(points: &[Point]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        sum += a.cross(b);
    }
    sum / 2.0
}

pub fn polygon_centroid(points: &[Point]) -> Option<Point> {
    let area = polygon_area(points);
    if area.abs() < 1e-12 {
        return Rect::from_points(points).map(|r| r.center());
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        let w = a.cross(b);
        cx += (a.x + b.x) * w;
        cy += (a.y + b.y) * w;
    }
    Some(Point::new(cx / (6.0 * area), cy / (6.0 * area)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Point, b: Point) -> bool {
        a.dist(b) < 1e-9
    }

    #[test]
    fn rotation_is_counterclockwise_on_screen() {
        // In a y down frame, turning (1, 0) counterclockwise on screen gives
        // (0, -1): the point moves up.
        let p = Point::new(1.0, 0.0).rotate_deg(90.0);
        assert!(close(p, Point::new(0.0, -1.0)), "{p:?}");
        let q = Point::new(1.0, 0.0).rotate_deg(-90.0);
        assert!(close(q, Point::new(0.0, 1.0)), "{q:?}");
    }

    #[test]
    fn arc_through_three_points() {
        let arc = arc_from_points(
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(-1.0, 0.0),
        )
        .unwrap();
        assert!(close(arc.center, Point::ZERO));
        assert!((arc.radius - 1.0).abs() < 1e-9);
        assert!((arc.sweep - std::f64::consts::PI).abs() < 1e-9);
        let other = arc_from_points(
            Point::new(1.0, 0.0),
            Point::new(0.0, -1.0),
            Point::new(-1.0, 0.0),
        )
        .unwrap();
        assert!((other.sweep + std::f64::consts::PI).abs() < 1e-9);
        assert!(arc_from_points(Point::ZERO, Point::new(1.0, 1.0), Point::new(2.0, 2.0)).is_none());
    }

    #[test]
    fn arc_points_stay_on_circle() {
        let pts = arc_points(
            Point::new(2.0, 0.0),
            Point::new(0.0, 2.0),
            Point::new(-2.0, 0.0),
            0.01,
        );
        assert!(pts.len() > 10);
        for p in &pts {
            assert!((p.norm() - 2.0).abs() < 0.011, "{p:?}");
        }
        assert!(close(pts[0], Point::new(2.0, 0.0)));
        assert!(close(*pts.last().unwrap(), Point::new(-2.0, 0.0)));
    }

    #[test]
    fn polygon_area_and_centroid() {
        let square = [
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ];
        assert!((polygon_area(&square).abs() - 4.0).abs() < 1e-12);
        assert!(close(
            polygon_centroid(&square).unwrap(),
            Point::new(1.0, 1.0)
        ));
    }
}
