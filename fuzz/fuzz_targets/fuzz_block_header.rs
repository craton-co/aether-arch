//! Fuzz target for block header and trailer parsing.
//!
//! Feeds arbitrary bytes to `BlockHeader::read_from` and `BlockTrailer::read_from`
//! to catch panics, integer overflows, or unbounded allocations from crafted input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Try parsing as a block header (28 bytes)
    let mut cursor = Cursor::new(data);
    let _ = aether_core::block::BlockHeader::read_from(&mut cursor);

    // Try parsing as a block trailer (36 bytes)
    let mut cursor = Cursor::new(data);
    let _ = aether_core::block::BlockTrailer::read_from(&mut cursor);

    // Try parsing as a block index entry (24 bytes)
    let mut cursor = Cursor::new(data);
    let _ = aether_core::block::BlockIndexEntry::read_from(&mut cursor);
});
