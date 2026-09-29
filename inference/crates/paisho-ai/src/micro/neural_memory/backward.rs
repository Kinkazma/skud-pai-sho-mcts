use super::*;
use forward::product;
fn sum_rows(x: &[f64], width: usize, out: &mut [f64]) {
    for row in x.chunks_exact(width) {
        for (o, v) in out.iter_mut().zip(row) {
            *o += v;
        }
    }
}
impl Cache {
    pub(super) fn backward(&self, w: &[f64], d: &[f64], g: &mut [f64], views:Option<&BackwardWeights>) -> SideGradient {
        assert_eq!(d.len(), self.n * 2);
        let views=views.filter(|_|(32..=1024).contains(&self.n));
        paisho_platform::matrix_product(&self.r,d,&mut g[OUT..BOUT],WIDTH,self.n,2,true,false);
        sum_rows(d, 2, &mut g[BOUT..BOUT + 2]);
        let mut dq = product(d, &w[OUT..BOUT], self.n, 2, WIDTH, false, true);
        let mut db = Scratch::zeros(self.n * WIDTH);
        for a in 0..self.n {
            let mut mean = 0.;
            let mut projection = 0.;
            for j in 0..WIDTH {
                let i = a * WIDTH + j;
                g[GAMMA + j] += dq[i] * self.normalized[i];
                g[BETA + j] += dq[i];
                db[i] = dq[i] * w[GAMMA + j];
                mean += db[i];
                projection += db[i] * self.normalized[i];
            }
            mean /= WIDTH as f64;
            projection /= WIDTH as f64;
            for j in 0..WIDTH {
                let i = a * WIDTH + j;
                db[i] = self.inverse[a] * (db[i] - mean - self.normalized[i] * projection);
            }
        }
        paisho_platform::matrix_product(&self.h,&db,&mut g[W2..B2],EXPAND,self.n,WIDTH,true,false);
        sum_rows(&db, WIDTH, &mut g[B2..GAMMA]);
        let mut du = match views {Some(v)=>product(&db,&v.second,self.n,WIDTH,EXPAND,false,false),None=>product(&db, &w[W2..B2], self.n, WIDTH, EXPAND, false, true)};
        for ((d, u), c) in du.iter_mut().zip(&self.u).zip(&self.erfc) {
            *d *= 0.5 * c
                + u * (-u * u / 2.).exp() / (2. * std::f64::consts::PI).sqrt();
        }
        paisho_platform::matrix_product(&self.q,&du,&mut g[W1..B1],WIDTH,self.n,EXPAND,true,false);
        sum_rows(&du, EXPAND, &mut g[B1..W2]);
        let inner = match views {Some(v)=>product(&du,&v.first,self.n,EXPAND,WIDTH,false,false),None=>product(&du, &w[W1..B1], self.n, EXPAND, WIDTH, false, true)};
        for (q, v) in dq.iter_mut().zip(inner.iter()) {
            *q += v;
        }
        for a in 0..self.n {
            let range = a * WIDTH..(a + 1) * WIDTH;
            let dot = dq[range.clone()]
                .iter()
                .zip(&self.q[range.clone()])
                .map(|(d, q)| d * q)
                .sum::<f64>();
            for i in range {
                dq[i] = (dq[i] - self.q[i] * dot) / self.norms[a];
            }
        }
        super::query::backward(&self.x,&dq,self.n,&mut g[W0..B0]);
        sum_rows(&dq, WIDTH, &mut g[B0..W1]);
        // Static state/action features need no input gradient. Differentiate the
        // current reader, value and action logit, so there is no stale side input.
        let side = match views {Some(v)=>product(&dq,&v.side,self.n,WIDTH,34,false,false),None=>product(&dq, &w[449 * WIDTH..B0], self.n, WIDTH, 34, false, true)};
        let mut out = SideGradient {
            value: 0.,
            memory: [0.; 32],
            logits: Vec::with_capacity(self.n),
        };
        for row in side.chunks_exact(34) {
            for (g, v) in out.memory.iter_mut().zip(row) {
                *g += v;
            }
            out.value += row[32];
            out.logits.push(row[33]);
        }
        out
    }
}
