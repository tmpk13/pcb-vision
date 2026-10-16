//! A small deterministic random source for tests and synthetic images.

#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.unit()
    }

    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Standard normal, Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_and_spread() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        let xs: Vec<f64> = (0..1000).map(|_| a.unit()).collect();
        assert!(xs.iter().all(|x| (0.0..1.0).contains(x)));
        assert!(xs.iter().zip((0..1000).map(|_| b.unit())).all(|(x, y)| *x == y));
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        assert!((mean - 0.5).abs() < 0.05, "{mean}");
        let n: Vec<f64> = (0..2000).map(|_| a.normal()).collect();
        let var = n.iter().map(|v| v * v).sum::<f64>() / n.len() as f64;
        assert!((var - 1.0).abs() < 0.15, "{var}");
    }
}
