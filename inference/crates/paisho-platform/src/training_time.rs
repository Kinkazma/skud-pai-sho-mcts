//! Optional process-local training clock. Only a verified, suspended process may
//! have its mapped pause offset changed by the desktop controller. Search loops
//! read RAM, never files. Processes without this clock use the ordinary Instant.
use std::{io, path::{Path, PathBuf}, sync::OnceLock, time::{Duration, Instant}};
struct Mapping { address: usize, path: PathBuf }
static CLOCK: OnceLock<Mapping> = OnceLock::new();

/// Enable once, before any workers start. The mapping stays alive until exit.
/// The file is never truncated/replaced: the controller updates its eight bytes
/// only after SIGSTOP is confirmed and before SIGCONT.
pub fn enable(path: &Path) -> io::Result<bool> {
    #[cfg(unix)] {
        use std::{fs::OpenOptions, io::Write, os::fd::AsRawFd};
        use std::ffi::c_void;
        if CLOCK.get().is_some() { return Err(io::Error::new(io::ErrorKind::AlreadyExists,"resident clock already enabled")); }
        let mut file=OpenOptions::new().read(true).write(true).create_new(true).open(path)?;
        file.write_all(&0u64.to_le_bytes())?;
        file.sync_all()?;
        extern "C" { fn mmap(address:*mut c_void,length:usize,protection:i32,flags:i32,fd:i32,offset:i64)->*mut c_void; }
        // SAFETY: the initialized eight-byte regular file remains mapped for the
        // process lifetime. PROT_READ=1, MAP_SHARED=1 on supported Unix systems.
        // Nothing in this process writes it. External updates require suspension
        // of every reader, enforced by the controller's verified SIGSTOP protocol.
        let address=unsafe { mmap(std::ptr::null_mut(),8,1,1,file.as_raw_fd(),0) };
        if address as isize == -1 { return Err(io::Error::last_os_error()); }
        CLOCK.set(Mapping {address:address as usize,path:path.to_owned()})
            .map_err(|_|io::Error::new(io::ErrorKind::AlreadyExists,"resident clock raced"))?;
        Ok(true)
    }
    #[cfg(not(unix))] { let _=path; Ok(false) }
}
pub fn path() -> Option<&'static Path> { CLOCK.get().map(|c|c.path.as_path()) }
pub fn paused() -> Duration {
    CLOCK.get().map_or(Duration::ZERO, |m| {
        // SAFETY: mmap returns page-aligned initialized readable memory, retained
        // for the entire process lifetime. Volatile reads observe controller
        // writes after resume; no concurrent writer is permitted by the protocol.
        let nanos=u64::from_le(unsafe { std::ptr::read_volatile(m.address as *const u64) });
        Duration::from_nanos(nanos)
    })
}
#[inline]
pub fn now() -> Instant {
    loop {
        let before=paused();
        let raw=Instant::now();
        let after=paused();
        // SIGSTOP may land between the raw timestamp and mapped-offset read.
        if before==after { return raw.checked_sub(after).unwrap_or(raw); }
    }
}
#[inline]
pub fn elapsed(start: Instant) -> Duration { now().saturating_duration_since(start) }
