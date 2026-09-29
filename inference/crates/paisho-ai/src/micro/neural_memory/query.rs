//! Reuse the common board prefix without regrouping its floating-point sum.
//! These kernels target the native Apple ARM f64 BLAS reduction; other backends
//! retain the original full matrix products. Small/exceptional inputs fall back.
use super::*;

pub(super) fn forward(x: &[f64], w: &[f64], rows: usize) -> Scratch {
    #[cfg(all(target_os="macos",target_arch="aarch64"))]
    if (32..=1024).contains(&rows) && x.iter().all(|v|v.is_finite()) {
        const PREFIX:usize=417;
        const REST:usize=INPUT-PREFIX+1;
        // BLAS beta=1 adds the old C after its dot and changes rounding. Instead
        // put the exact prefix sum in the first product (1 * prefix + 0), then
        // continue the original fused input order for every remaining feature.
        let mut prefix=[0f64;WIDTH];
        for (i,&v) in x[..PREFIX].iter().enumerate() {
            for j in 0..WIDTH {prefix[j]=v.mul_add(w[i*WIDTH+j],prefix[j]);}
        }
        if prefix.iter().all(|v|v.is_finite()) {
            let mut suffix=Vec::with_capacity(REST*WIDTH);
            suffix.extend_from_slice(&prefix);
            suffix.extend_from_slice(&w[PREFIX*WIDTH..]);
            let mut dynamic=Vec::with_capacity(rows*REST);
            for row in x.chunks_exact(INPUT) {
                dynamic.push(1.);
                dynamic.extend_from_slice(&row[PREFIX..]);
            }
            return forward::product(&dynamic,&suffix,rows,REST,WIDTH,false,false);
        }
    }
    forward::product(x,w,rows,INPUT,WIDTH,false,false)
}

pub(super) fn backward(x: &[f64], d: &[f64], rows: usize, g: &mut [f64]) {
    #[cfg(all(target_os="macos",target_arch="aarch64"))]
    if (32..=1024).contains(&rows) && d.iter().all(|v|v.is_finite()) {
        // Constant columns with identical bits have identical per-unit gradients.
        // Keep one BLAS representative, including distinct +0 and -0. Actual
        // action columns may vary; inspect all their rows before sharing a result.
        let mut constants=std::collections::HashMap::<u64,usize>::new();
        let mut active=Vec::new();
        let mut omitted=Vec::new();
        for j in 0..INPUT {
            let bits=x[j].to_bits();
            let fixed=(!(417..449).contains(&j) && j!=482)
                || x.chunks_exact(INPUT).all(|row|row[j].to_bits()==bits);
            if fixed && x[j].is_finite() {
                if let Some(&source)=constants.get(&bits) {omitted.push((j,source));}
                else {constants.insert(bits,active.len());active.push(j);}
            } else {active.push(j);}
        }
        if active.len()>=8 && active.len()<INPUT*3/4 {
            let mut compact=Vec::with_capacity(rows*active.len());
            for row in x.chunks_exact(INPUT) {for &j in &active {compact.push(row[j]);}}
            let product=forward::product(&compact,d,active.len(),rows,WIDTH,true,false);
            for (&j,row) in active.iter().zip(product.chunks_exact(WIDTH)) {
                g[j*WIDTH..(j+1)*WIDTH].copy_from_slice(row);
            }
            for (j,source) in omitted {
                g[j*WIDTH..(j+1)*WIDTH].copy_from_slice(&product[source*WIDTH..(source+1)*WIDTH]);
            }
            return;
        }
    }
    paisho_platform::matrix_product(x,d,g,INPUT,rows,WIDTH,true,false);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_constant_columns_keep_full_blas_derivatives_including_sign_bits() {
        let mut rng=StableRng::new(92931);
        for rows in [32,33,65,130,185,257,524,800,1024] {
            let mut x=vec![0.;rows*INPUT];
            for (i,row) in x.chunks_exact_mut(INPUT).enumerate() {
                for (j,v) in row.iter_mut().enumerate() {*v=[0.,-0.,0.5,-0.75,1.][j%5];}
                for j in 417..449 {if j%3==0 {row[j]=i as f64/(rows+1) as f64;}}
                row[482]=rng.next_f64();
            }
            for d in [(0..rows*WIDTH).map(|_|rng.next_f64()*2.-1.).collect::<Vec<_>>(),vec![-0.;rows*WIDTH]] {
                let reference=forward::product(&x,&d,INPUT,rows,WIDTH,true,false);
                let mut candidate=vec![f64::NAN;INPUT*WIDTH];backward(&x,&d,rows,&mut candidate);
                assert!(reference.iter().zip(candidate).all(|(a,b)|a.to_bits()==b.to_bits()),"constant derivative rows {rows}");
            }
        }
    }
    #[test]
    fn common_prefix_and_sparse_derivatives_match_full_blas_bits() {
        let mut rng=StableRng::new(95131);
        for n in [1,2,8,31,32,33,65,127,130,185,257,524,800,1025] {
            let w=(0..INPUT*WIDTH).map(|_|rng.next_f64()*2.-1.).collect::<Vec<_>>();
            let state=(0..417).map(|j|if j%5==0 {rng.next_f64()*2.-1.} else {-0.}).collect::<Vec<_>>();
            let mut x=vec![0.;n*INPUT];
            for row in x.chunks_exact_mut(INPUT) {
                row[..417].copy_from_slice(&state);
                for v in &mut row[417..449] {*v=rng.next_f64()*2.-1.;}
                row[481]=0.31;row[482]=rng.next_f64()*2.-1.;
            }
            let full=forward::product(&x,&w,n,INPUT,WIDTH,false,false);
            let candidate=forward(&x,&w,n);
            assert!(full.iter().zip(candidate.iter()).all(|(a,b)|a.to_bits()==b.to_bits()),"query rows {n}");
            for d in [(0..n*WIDTH).map(|_|rng.next_f64()*2.-1.).collect::<Vec<_>>(),vec![-0.031;n*WIDTH],vec![-0.;n*WIDTH]] {
                let full=forward::product(&x,&d,INPUT,n,WIDTH,true,false);
                let mut candidate=vec![0.;INPUT*WIDTH];backward(&x,&d,n,&mut candidate);
                assert!(full.iter().zip(candidate).all(|(a,b)|a.to_bits()==b.to_bits()),"gradient rows {n}");
            }
        }
    }
}
