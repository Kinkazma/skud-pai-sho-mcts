//! Reuse allocation capacity only. Each borrower gets freshly initialized +0.0.
//! Per-thread storage is bounded and contains no model, examples or cache keys.
use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
};
const MAX_BYTES: usize = 8 * 1024 * 1024;
const MAX_BUFFERS: usize = 16;
thread_local! {static POOL:RefCell<Vec<Vec<f64>>>=const {RefCell::new(Vec::new())};}
pub(crate) struct Scratch(Vec<f64>);
impl Scratch {
    pub(super) fn zeros(n: usize) -> Self {
        let mut v = POOL.with(|p| {
            let mut p = p.borrow_mut();
            let i = p
                .iter()
                .enumerate()
                .filter(|(_, v)| v.capacity() >= n)
                .min_by_key(|(_, v)| v.capacity())
                .map(|(i, _)| i);
            i.map(|i| p.swap_remove(i)).unwrap_or_default()
        });
        let old = v.len();
        v.resize(n, 0.);
        v[..old.min(n)].fill(0.);
        Self(v)
    }
}
impl Deref for Scratch {
    type Target = Vec<f64>;
    fn deref(&self) -> &Vec<f64> {
        &self.0
    }
}
impl DerefMut for Scratch {
    fn deref_mut(&mut self) -> &mut Vec<f64> {
        &mut self.0
    }
}
impl<'a> IntoIterator for &'a Scratch {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = POOL.try_with(|p| {
            let mut p = p.borrow_mut();
            let bytes = p.iter().map(|v| v.capacity() * 8).sum::<usize>();
            if self.0.capacity() > 0
                && p.len() < MAX_BUFFERS
                && self.0.capacity() * 8 <= MAX_BYTES - bytes
            {
                p.push(std::mem::take(&mut self.0));
            }
        });
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stale_values_are_cleared_and_resident_capacity_is_bounded() {
        std::thread::spawn(|| {
            let mut a = Scratch::zeros(1024);
            let ptr = a.as_ptr();
            a.fill(f64::NAN);
            drop(a);
            let b = Scratch::zeros(511);
            assert_eq!(ptr, b.as_ptr());
            assert!(b.iter().all(|v| v.to_bits() == 0));
            drop(b);
            let large = (0..24).map(|_| Scratch::zeros(131072)).collect::<Vec<_>>();
            drop(large);
            POOL.with(|p| {
                let p = p.borrow();
                assert!(p.len() <= MAX_BUFFERS);
                assert!(p.iter().map(|v| v.capacity() * 8).sum::<usize>() <= MAX_BYTES);
            });
            let mut c = Scratch::zeros(300);
            c.fill(-0.);
            drop(c);
            assert!(Scratch::zeros(300).iter().all(|v| v.to_bits() == 0));
        })
        .join()
        .unwrap();
    }
}
