//! Built-in dictionaries embedded in the binary for per-type auto-selection.
//!
//! The compressor can auto-select one of these by detected content type
//! (see [`for_content_type`]); the decompressor resolves it transparently by
//! the dictionary's BLAKE3 state hash recorded in the archive header (see
//! [`by_hash`]) so no `--dictionary` file is needed to extract.
//!
//! Currently a single text dictionary covers `ContentType::Text` (prose,
//! source code, JSON/XML/logs — `detect_content_type` maps them all to
//! `Text`). It is a NeuralSSM dictionary trained on the BWT+MTF+RLE stream
//! (see [`Dictionary::train_transformed`]) so it can seed the BWT coding
//! path's per-block reset baseline (Stage A).

use std::io::Cursor;

use crate::dictionary::Dictionary;
use crate::format::ContentType;

/// Text dictionary: trained on the repo's diverse text fixtures
/// (English prose + Rust source + JSON). Regenerate with:
/// `aet train tests/fixtures/large/{english.txt,source.rs,mixed.json}
///  -o aether-core/src/dictionaries/text.aed --predictor ssm --force`
static TEXT_DICT: &[u8] = include_bytes!("dictionaries/text.aed");

/// All embedded dictionaries, for hash-based resolution.
const ALL: &[&[u8]] = &[TEXT_DICT];

/// Return the built-in dictionary suited to `ct`, if one exists.
///
/// Only `ContentType::Text` has a dictionary today; other types return
/// `None` (the BWT/NeuralSSM path that dictionaries seed mainly benefits
/// text-like data).
pub fn for_content_type(ct: ContentType) -> Option<Dictionary> {
    let bytes = match ct {
        ContentType::Text => TEXT_DICT,
        _ => return None,
    };
    Dictionary::read_from(&mut Cursor::new(bytes)).ok()
}

/// Resolve a built-in dictionary by its state hash, for transparent extract.
pub fn by_hash(hash: &[u8; 32]) -> Option<Dictionary> {
    for bytes in ALL {
        if let Ok(d) = Dictionary::read_from(&mut Cursor::new(*bytes)) {
            if &d.hash == hash {
                return Some(d);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_text_dict_is_valid_and_resolvable() {
        let d = for_content_type(ContentType::Text).expect("text dict present");
        // Round-trips by hash (the mechanism extract relies on).
        let resolved = by_hash(&d.hash).expect("resolvable by hash");
        assert_eq!(resolved.hash, d.hash);
        assert_eq!(resolved.predictor_id, crate::format::PredictorId::NeuralSsm);
    }

    #[test]
    fn non_text_types_have_no_builtin_dict() {
        assert!(for_content_type(ContentType::Image).is_none());
        assert!(for_content_type(ContentType::NumericData).is_none());
    }
}
