//! Fuzz target for archive listing.
//!
//! Focuses on exercising the file listing and metadata parsing paths
//! with malformed archives.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::ffi::CString;
use std::io::Write;
use std::ptr;

fuzz_target!(|data: &[u8]| {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(data).ok();
    let path = tmp.path().to_str().unwrap();
    let archive_cstr = CString::new(path).unwrap();

    // Use Order0 predictor (cheapest)
    let d = aether_ffi::aether_decompressor_new(0);
    if d.is_null() {
        return;
    }

    // Try listing
    let mut list: *mut aether_ffi::AetherFileList = ptr::null_mut();
    let _code = aether_ffi::aether_list(d, archive_cstr.as_ptr(), &mut list);
    if !list.is_null() {
        // Exercise accessors
        let count = aether_ffi::aether_file_list_count(list);
        for i in 0..count {
            let _info = aether_ffi::aether_file_list_get(list, i);
        }
        // Out-of-bounds access should return null
        let _null = aether_ffi::aether_file_list_get(list, count);

        let mut list = list;
        aether_ffi::aether_file_list_free(&mut list);
    }

    // Try verify
    let mut result = aether_ffi::AetherVerifyResult {
        total_blocks: 0,
        verified_blocks: 0,
        corrupted_count: 0,
        is_ok: 0,
    };
    let _code = aether_ffi::aether_verify(d, archive_cstr.as_ptr(), &mut result);

    let err = aether_ffi::aether_last_error();
    if !err.is_null() {
        aether_ffi::aether_error_free(err);
    }
    let mut d = d;
    aether_ffi::aether_decompressor_free(&mut d);
});
