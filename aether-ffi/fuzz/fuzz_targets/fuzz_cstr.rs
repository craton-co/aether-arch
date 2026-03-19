//! Fuzz target for C string handling.
//!
//! Exercises the bounded CStr conversion with arbitrary byte sequences
//! to ensure no panics or memory safety issues.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::ffi::CString;
use std::ptr;

fuzz_target!(|data: &[u8]| {
    // Create a null-terminated version of the fuzz data
    let mut buf = data.to_vec();
    buf.push(0); // ensure null termination

    let ptr = buf.as_ptr() as *const std::ffi::c_char;

    // Exercise compressor creation with the fuzzed string as a
    // hypothetical path (will fail at file I/O, but exercises cstr parsing)
    let c = aether_ffi::aether_compressor_new(0);
    if !c.is_null() {
        let file_ptrs = [ptr];
        let out_path = CString::new("/dev/null").unwrap();

        let _code = aether_ffi::aether_compress(
            c,
            ptr,
            file_ptrs.as_ptr(),
            1,
            out_path.as_ptr(),
        );

        let err = aether_ffi::aether_last_error();
        if !err.is_null() {
            aether_ffi::aether_error_free(err);
        }

        let mut c = c;
        aether_ffi::aether_compressor_free(&mut c);
    }
});
