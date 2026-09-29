//! Small safe interface to per-thread macOS QoS; it does not pin physical cores.
//! Apple: https://developer.apple.com/news/?id=vk3m204o
use std::io;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadQos {
    UserInitiated,
    Utility,
    Background,
}
impl ThreadQos {
    pub fn code(self) -> u32 {
        match self {
            Self::UserInitiated => 0x19,
            Self::Utility => 0x11,
            Self::Background => 0x09,
        }
    }
}
/// Applies a scheduling hint to the calling thread only, checking the requested
/// class back through the native API. Unsupported platforms return None.
pub fn set_current_thread_qos(qos: ThreadQos) -> io::Result<Option<u32>> {
    #[cfg(target_os = "macos")]
    {
        use std::os::raw::{c_int, c_uint};
        extern "C" {
            fn pthread_set_qos_class_self_np(class: c_uint, priority: c_int) -> c_int;
            fn qos_class_self() -> c_uint;
        }
        // SAFETY: these macOS 10.10+ functions accept only primitive values,
        // access the current thread, and retain no Rust memory or callbacks.
        // The enum supplies a valid qos_class_t and relative priority zero.
        let code = unsafe { pthread_set_qos_class_self_np(qos.code(), 0) };
        if code != 0 {
            return Err(io::Error::from_raw_os_error(code));
        }
        // SAFETY: this parameterless query only returns the calling thread's QoS.
        let actual = unsafe { qos_class_self() };
        if actual != qos.code() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "QoS readback mismatch",
            ));
        }
        Ok(Some(actual))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = qos;
        Ok(None)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn isolated_threads_read_back_both_native_qos_classes() {
        for qos in [
            ThreadQos::Background,
            ThreadQos::Utility,
            ThreadQos::UserInitiated,
        ] {
            let result = std::thread::spawn(move || set_current_thread_qos(qos))
                .join()
                .unwrap()
                .unwrap();
            #[cfg(target_os = "macos")]
            assert_eq!(result, Some(qos.code()));
            #[cfg(not(target_os = "macos"))]
            assert_eq!(result, None);
        }
    }
}

/// Flush this file's OS buffers before a later, single full storage barrier for
/// the whole batch. This is NOT a replacement for the caller's final sync_all().
/// Apple fsync(2): https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html
pub fn sync_before_batch_commit(file: &std::fs::File) -> io::Result<()> {
    #[cfg(target_os="macos")]
    {
        use std::os::fd::AsRawFd;
        extern "C" { fn fsync(fd:std::os::raw::c_int)->std::os::raw::c_int; }
        loop {
            // SAFETY: the borrowed File keeps this descriptor valid throughout
            // the synchronous POSIX call. No pointer or ownership is transferred.
            if unsafe { fsync(file.as_raw_fd()) } == 0 { return Ok(()); }
            let error=io::Error::last_os_error();
            if error.kind()!=io::ErrorKind::Interrupted { return Err(error); }
        }
    }
    #[cfg(not(target_os="macos"))]
    file.sync_all()
}
