use super::*;
pub(super) fn product(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    ta: bool,
    tb: bool,
) -> Scratch {
    let mut out = Scratch::zeros(m * n);
    paisho_platform::matrix_product(a, b, &mut out, m, k, n, ta, tb);
    out
}
pub(crate) struct Cache {
    pub x: Scratch,
    pub q: Scratch,
    pub norms: Vec<f64>,
    pub u: Scratch,
    pub h: Scratch,
    pub erfc: Scratch,
    pub normalized: Scratch,
    pub inverse: Vec<f64>,
    pub r: Scratch,
    pub output: Scratch,
    pub n: usize,
}
impl Cache {
    pub(super) fn new(
        w: &[f64],
        state: &[f64],
        actions: &[[f64; 32]],
        memory: &[f64; 32],
        value: f64,
        logits: &[f64],
    ) -> Self {
        Self::build(w, state, actions, memory, value, logits, true)
    }
    pub(super) fn inference(
        w: &[f64], state: &[f64], actions: &[[f64; 32]], memory: &[f64; 32],
        value: f64, logits: &[f64],
    ) -> Self {
        Self::build(w, state, actions, memory, value, logits, false)
    }
    fn build(
        w: &[f64], state: &[f64], actions: &[[f64; 32]], memory: &[f64; 32],
        value: f64, logits: &[f64], retain_input: bool,
    ) -> Self {
        let n = actions.len();
        assert_eq!(logits.len(), n);
        let compact = if retain_input { None } else {
            super::query::root(state, actions, memory, value, logits, &w[W0..B0])
        };
        let (x, mut q) = if let Some(q) = compact { (Scratch::zeros(0), q) } else {
        let mut x = Scratch::zeros(n * INPUT);
        for (i, a) in actions.iter().enumerate() {
            let row = &mut x[i * INPUT..(i + 1) * INPUT];
            row[..state.len()].copy_from_slice(state);
            row[417..449].copy_from_slice(a);
            row[449..481].copy_from_slice(memory);
            row[481] = value;
            row[482] = logits[i];
        }
        let q = super::query::forward(&x, &w[W0..B0], n);
        (x, q)
        };
        let mut norms = Vec::with_capacity(n);
        for row in q.chunks_exact_mut(WIDTH) {
            for (j, v) in row.iter_mut().enumerate() {
                *v += w[B0 + j];
            }
            let norm = (row.iter().map(|v| v * v).sum::<f64>() + 1e-12).sqrt();
            norms.push(norm);
            for v in row {
                *v /= norm;
            }
        }
        let mut u = product(&q, &w[W1..B1], n, WIDTH, EXPAND, false, false);
        for (i, v) in u.iter_mut().enumerate() {
            *v += w[B1 + i % EXPAND];
        }
        let erfc = super::gelu::erfc_inputs(&u);
        let mut h=Scratch::zeros(u.len());
        for (h,(v,c)) in h.iter_mut().zip(u.iter().zip(&erfc)) {*h=0.5*v*c;}
        let mut normalized = product(&h, &w[W2..B2], n, EXPAND, WIDTH, false, false);
        let mut inverse = Vec::with_capacity(n);
        for row in normalized.chunks_exact_mut(WIDTH) {
            for (j, v) in row.iter_mut().enumerate() {
                *v += w[B2 + j];
            }
            let mean = row.iter().sum::<f64>() / WIDTH as f64;
            for v in row.iter_mut() {
                *v -= mean;
            }
            let inv = 1. / (row.iter().map(|v| v * v).sum::<f64>() / WIDTH as f64 + 1e-5).sqrt();
            inverse.push(inv);
            for v in row {
                *v *= inv;
            }
        }
        let mut r=Scratch::zeros(q.len());
        for (r,(i,(q,v))) in r.iter_mut().zip(q
            .iter()
            .zip(&normalized)
            .enumerate()) {*r=q + v * w[GAMMA + i % WIDTH] + w[BETA + i % WIDTH];}
        let mut output = product(&r, &w[OUT..BOUT], n, WIDTH, 2, false, false);
        for (i, v) in output.iter_mut().enumerate() {
            *v += w[BOUT + i % 2];
        }
        Self {
            x,
            q,
            norms,
            u,
            h,
            erfc,
            normalized,
            inverse,
            r,
            output,
            n,
        }
    }
}
