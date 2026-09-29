//! Batch the identical libm 0.2.15 small-argument expression, preserving every
//! operation and falling back to libm for all other ranges. No GELU approximation.
/* origin: FreeBSD /usr/src/lib/msun/src/s_erf.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
const PP0: f64 = 1.28379167095512558561e-01; /* 0x3FC06EBA, 0x8214DB68 */
const PP1: f64 = -3.25042107247001499370e-01; /* 0xBFD4CD7D, 0x691CB913 */
const PP2: f64 = -2.84817495755985104766e-02; /* 0xBF9D2A51, 0xDBD7194F */
const PP3: f64 = -5.77027029648944159157e-03; /* 0xBF77A291, 0x236668E4 */
const PP4: f64 = -2.37630166566501626084e-05; /* 0xBEF8EAD6, 0x120016AC */
const QQ1: f64 = 3.97917223959155352819e-01; /* 0x3FD97779, 0xCDDADC09 */
const QQ2: f64 = 6.50222499887672944485e-02; /* 0x3FB0A54C, 0x5536CEBA */
const QQ3: f64 = 5.08130628187576562776e-03; /* 0x3F74D022, 0xC4D36B0F */
const QQ4: f64 = 1.32494738004321644526e-04; /* 0x3F215DC9, 0x221C1A10 */
const QQ5: f64 = -3.96022827877536812320e-06; /* 0xBED09C43, 0x42A26120 */

#[inline]
fn polynomial(x:f64)->f64 {
    let z=x*x;
    let r=PP0+z*(PP1+z*(PP2+z*(PP3+z*PP4)));
    let s=1.+z*(QQ1+z*(QQ2+z*(QQ3+z*(QQ4+z*QQ5))));
    let y=r/s;
    1.-(x+x*y)
}
pub(super) fn erfc_inputs(u:&[f64])->super::Scratch {
    let mut out=super::Scratch::zeros(u.len());
    let mut a=u.chunks_exact(4);let mut b=out.chunks_exact_mut(4);
    for (src,dst) in a.by_ref().zip(b.by_ref()) {
        let x:[f64;4]=std::array::from_fn(|i|-src[i]/std::f64::consts::SQRT_2);
        if x.iter().all(|v| {let h=((v.to_bits()>>32) as u32)&0x7fffffff; h>=0x3c700000 && h<0x3fd00000}) {
            for i in 0..4 {dst[i]=polynomial(x[i]);}
        } else {
            for i in 0..4 {dst[i]=libm::erfc(x[i]);}
        }
    }
    for (src,dst) in a.remainder().iter().zip(b.into_remainder()) {*dst=libm::erfc(-src/std::f64::consts::SQRT_2);}
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn libm_bits_match_dense_ranges_and_exponent_extremes() {
        let mut u:Vec<f64>=(-200000..=200000).map(|i|i as f64/100000.).collect();
        for bits in [0u64,1,0x3c70000000000000,0x3fd0000000000000,0x3feb_0000_0000_0000,0x7fefffffffffffff,0x7ff0000000000000] {
            for d in [bits.saturating_sub(1),bits,bits.saturating_add(1)] {
                for sign in [0,1u64<<63] {u.push(f64::from_bits(d|sign));}
            }
        }
        for (x,y) in u.iter().zip(erfc_inputs(&u).iter()) {
            let reference=libm::erfc(-x/std::f64::consts::SQRT_2);
            assert!(reference.to_bits()==y.to_bits() || reference.is_nan() && y.is_nan());
        }
    }
}
