//! Fuzz target for path validation and header parsing.
//!
//! Exercises `FileEntry::read_from` and `ArchiveHeader::read_from` with
//! arbitrary bytes to catch path traversal bypasses, panics from crafted
//! path lengths, and integer overflows in header field parsing.
//!
//! `FileEntry::read_from` is the primary gatekeeper that rejects `..`
//! traversal, absolute paths, NUL bytes, Windows reserved names, and
//! excessively long paths — making it the #1 attack surface for archive
//! extraction.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

use aether_core::header::{ArchiveHeader, FileEntry, SolidGroupEntry};

fuzz_target!(|data: &[u8]| {
    // Phase 1: Fuzz FileEntry::read_from — the primary path validation entry point.
    // This exercises: path length limits, UTF-8 validation, NUL byte rejection,
    // absolute path rejection, `..` traversal rejection, Windows reserved name
    // rejection, trailing dot/space rejection, and permission sanitization.
    {
        let mut cursor = Cursor::new(data);
        if let Ok(entry) = FileEntry::read_from(&mut cursor) {
            // If parsing succeeded, verify the safety invariants hold:
            let path = &entry.path;

            // Must not contain NUL bytes
            assert!(
                !path.as_bytes().contains(&0),
                "FileEntry accepted path with NUL byte: {path:?}"
            );

            // Must not be absolute
            assert!(
                !path.starts_with('/') && !path.starts_with('\\'),
                "FileEntry accepted absolute path: {path:?}"
            );

            // Must not contain `..` traversal
            for component in path.split(&['/', '\\']) {
                assert_ne!(
                    component, "..",
                    "FileEntry accepted path with `..` component: {path:?}"
                );
            }

            // Must not have Windows drive letter (C:\, D:\, etc.)
            if path.len() >= 2 {
                assert_ne!(
                    path.as_bytes()[1], b':',
                    "FileEntry accepted Windows absolute path: {path:?}"
                );
            }

            // Permissions must have setuid/setgid/sticky stripped
            assert_eq!(
                entry.permissions & !0o777,
                0,
                "FileEntry preserved dangerous permission bits: {:o}",
                entry.permissions,
            );
        }
    }

    // Phase 2: Fuzz ArchiveHeader::read_from — validates magic, counts, and integrity tag.
    {
        let mut cursor = Cursor::new(data);
        let _ = ArchiveHeader::read_from(&mut cursor);
    }

    // Phase 3: Fuzz SolidGroupEntry::read_from — fixed 24-byte structure.
    {
        let mut cursor = Cursor::new(data);
        let _ = SolidGroupEntry::read_from(&mut cursor);
    }
});
