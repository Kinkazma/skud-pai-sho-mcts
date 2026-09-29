use super::*;

/// Per-head class balancing, with a separate fixed budget for counts/events.
pub(super) fn auxiliary(
    c: &Cache,
    targets: &[Option<MicroStructuredTarget>],
    scale: f64,
    balance: Option<&MicroStructuredBatchBalance>,
) -> (f64, Vec<[f64; OUTPUT]>) {
    let mut d = vec![[0.; OUTPUT]; c.output.len()];
    let mut loss = 0.;
    let n = targets.iter().flatten().count();
    if n == 0 || scale == 0. {
        return (loss, d);
    }
    let count_scale = scale * 0.2 / (n * 10) as f64;
    for (i, target) in targets.iter().enumerate() {
        if let Some(target) = target {
            for j in 0..10 {
                let e = c.output[i][1 + j] - target.counts[j];
                loss += count_scale
                    * if e.abs() <= 1. {
                        0.5 * e * e
                    } else {
                        e.abs() - 0.5
                    };
                d[i][1 + j] = count_scale * e.clamp(-1., 1.);
            }
        }
    }
    let active = (0..10)
        .filter(|&j| targets.iter().flatten().any(|t| t.events[j].is_some()))
        .count();
    if active == 0 {
        return (loss, d);
    }
    for j in 0..10 {
        let counts = [false, true].map(|v| {
            targets
                .iter()
                .flatten()
                .filter(|t| t.events[j] == Some(v))
                .count()
        });
        let classes = counts.iter().filter(|n| **n > 0).count();
        if classes == 0 {
            continue;
        }
        for (i, t) in targets.iter().enumerate() {
            if let Some(v) = t.as_ref().and_then(|t| t.events[j]) {
                let k = 11 + j;
                let y = c.output[i][k];
                let weight = match balance {
                    Some(b) => scale * 0.2 * b.weight(j, v),
                    None => scale * 0.2 / (active * classes * counts[usize::from(v)]) as f64,
                };
                loss += weight * (y.max(0.) + (-y.abs()).exp().ln_1p() - f64::from(v) * y);
                d[i][k] = weight * (1. / (1. + (-y).exp()) - f64::from(v));
            }
        }
    }
    (loss, d)
}

impl Weights {
    pub(in crate::micro) fn backward(
        &self,
        c: &Cache,
        state: &[f64],
        actions: &[[f64; 32]],
        d: &[[f64; OUTPUT]],
        dv: f64,
        g: &mut [f64],
    ) {
        let w = &self.w;
        let mut total = [0.; H];
        let mut dp = [0.; 32];
        let mut dn = vec![[0.; 16]; c.position.nodes.len()];
        for (i, a) in actions.iter().enumerate() {
            let h = &c.hidden[i];
            let mut dh = [0.; H];
            for k in 0..OUTPUT {
                g[OB + k] += d[i][k];
                for j in 0..H {
                    g[OW + j * OUTPUT + k] += h[j] * d[i][k];
                    dh[j] += w[OW + j * OUTPUT + k] * d[i][k];
                }
            }
            for j in 0..H {
                dh[j] *= 1. - h[j] * h[j];
                total[j] += dh[j];
            }
            for k in 0..32 {
                for j in 0..H {
                    g[HW + (417 + k) * H + j] += a[k] * dh[j];
                }
            }
            let dc: [f64; 16] = std::array::from_fn(|k| {
                dh.iter()
                    .enumerate()
                    .map(|(j, d)| w[HW + (481 + k) * H + j] * d)
                    .sum()
            });
            for k in 0..16 {
                for j in 0..H {
                    g[HW + (481 + k) * H + j] += c.context[i][k] * dh[j];
                }
            }
            let da: Vec<f64> = c
                .position
                .nodes
                .iter()
                .map(|node| node.iter().zip(&dc).map(|(a, b)| a * b).sum())
                .collect();
            let mean: f64 = da.iter().zip(&c.attention[i]).map(|(a, b)| a * b).sum();
            let mut dq = [0.; 16];
            for (n, node) in c.position.nodes.iter().enumerate() {
                let ds = c.attention[i][n] * (da[n] - mean);
                for k in 0..16 {
                    dn[n][k] += c.attention[i][n] * dc[k] + ds * c.query[i][k] / 4.;
                    dq[k] += ds * node[k] / 4.;
                }
            }
            for k in 0..32 {
                for j in 0..16 {
                    g[QUERY + k * 16 + j] += a[k] * dq[j];
                }
            }
        }
        for j in 0..H {
            g[HB + j] += total[j];
            for k in 0..417 {
                g[HW + k * H + j] += state[k] * total[j];
            }
            for k in 0..32 {
                g[HW + (449 + k) * H + j] += c.position.pool[k] * total[j];
                dp[k] += w[HW + (449 + k) * H + j] * total[j];
            }
        }
        for k in 0..32 {
            g[VALUE + k] += dv * c.position.pool[k];
            dp[k] += dv * w[VALUE + k];
        }
        g[VALUE + 32] += dv;
        let enc = self
            .encoder
            .gradient_with_nodes(
                &c.position.graph,
                MicroGraphPropagation::TwoHarmonyRounds,
                &dp,
                &dn,
            )
            .expect("matching cached nodes");
        for (to, from) in g[..HW].iter_mut().zip(enc) {
            *to += from;
        }
    }
}

impl MicroModel {
    pub(in crate::micro) fn relational_auxiliary(
        &self,
        c: &Cache,
        targets: &[Option<MicroStructuredTarget>],
        scale: f64,
        balance: Option<&MicroStructuredBatchBalance>,
    ) -> (f64, Vec<[f64; OUTPUT]>) {
        auxiliary(c, targets, scale, balance)
    }
}
