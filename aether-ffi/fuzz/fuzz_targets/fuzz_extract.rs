//! Fuzz target for archive extraction.
//!
//! Feeds arbitrary bytes as an archive to the decompression path,
//! exercising header parsing, path validation, and extraction logic.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::ffi::CString;
use std::io::Write;
use std::ptr;

fuzz_target!(|data: &[u8]| {
    // Write fuzz input to a temp file
    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(data).ok();
    let path = tmp.path().to_str().unwrap();
    let archive_cstr = CString::new(path).unwrap();

    // Try auto-detect (exercises header parsing)
    let d = aether_ffi::aether_decompressor_auto(archive_cstr.as_ptr());
    if d.is_null() {
        // Invalid archive — clean up error
        let err = aether_ffi::aether_last_error();
        if !err.is_null() {
            aether_ffi::aether_error_free(err);
        }
        return;
    }

    // Try listing files
    let mut list: *mut aether_ffi::AetherFileList = ptr::null_mut();
    let _code = aether_ffi::aether_list(d, archive_cstr.as_ptr(), &mut list);
    if !list.is_null() {
        let mut list = list;
        aether_ffi::aether_file_list_free(&mut list);
    }

    // Try extracting to a temp dir
    let outdir = tempfile::tempdir().unwrap();
    let outdir_cstr = CString::new(outdir.path().to_str().unwrap()).unwrap();
    let _code = aether_ffi::aether_extract_all(d, archive_cstr.as_ptr(), outdir_cstr.as_ptr());

    // Clean up
    let err = aether_ffi::aether_last_error();
    if !err.is_null() {
        aether_ffi::aether_error_free(err);
    }
    let mut d = d;
    aether_ffi::aether_decompressor_free(&mut d);
});
