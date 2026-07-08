//! Dictionary pretraining for domain-specific compression.
//!
//! A dictionary captures the learned state of a predictor after processing a
//! training corpus. When compressing similar data, initializing the predictor
//! from a dictionary instead of scratch can significantly improve compression
//! ratio — especially for small files where the predictor has little data
//! to learn from.
//!
//! # File Format (.aed)
//!
//! ```text
//! [magic: 4 bytes "AEDX"]
//! [version: u8]
//! [predictor_id: u16 LE]
//! [state_len: u32 LE]
//! [predictor_state: state_len bytes]
//! [blake3_hash: 32 bytes]  (hash of predictor_state)
//! ```

use std::io::{Read, Write};
use std::path::Path;

use crate::entropy::ProbabilityPredictor;
use crate::error::{AetherError, Result};
use crate::format::PredictorId;

/// Magic bytes identifying an AetherArch dictionary file.
const DICT_MAGIC: [u8; 4] = [0x41, 0x45, 0x44, 0x58]; // "AEDX"

/// Current dictionary format version.
const DICT_VERSION: u8 = 1;

/// Maximum dictionary state size (64 MiB) — enforced on both save and load.
const MAX_DICT_STATE_SIZE: usize = 64 * 1024 * 1024;

/// A trained dictionary that can initialize a predictor with pretrained state.
#[derive(Clone)]
pub struct Dictionary {
    /// Which predictor this dictionary was trained for.
    pub predictor_id: PredictorId,
    /// Serialized predictor state.
    pub state: Vec<u8>,
    /// BLAKE3 hash of the state bytes (for integrity and archive matching).
    pub hash: [u8; 32],
}

impl Dictionary {
    /// Train a dictionary by feeding all files through a predictor.
    ///
    /// The predictor is fed each file's bytes via predict+update (simulating
    /// compression without actually encoding). After processing the entire
    /// corpus, the learned state is saved.
    pub fn train(
        predictor: &mut dyn ProbabilityPredictor,
        training_files: &[impl AsRef<Path>],
    ) -> Result<Self> {
        for path in training_files {
            let data = std::fs::read(path.as_ref())?;
            for &byte in &data {
                predictor.predict();
                predictor.update(byte);
            }
        }

        let state = predictor.save_state().ok_or_else(|| {
            AetherError::Compression(format!(
                "Predictor '{}' does not support dictionary pretraining",
                predictor.name(),
            ))
        })?;

        let hash = *blake3::hash(&state).as_bytes();
        let predictor_id = predictor.predictor_id();

        Ok(Dictionary {
            predictor_id,
            state,
            hash,
        })
    }

    /// Train a dictionary on the BWT+MTF+RLE-transformed stream (Stage A).
    ///
    /// The high-ratio coding path (BwtPredictorRans) feeds its NeuralSSM the
    /// *transformed* stream — MTF ranks + RUNA/RUNB — not raw bytes. A
    /// dictionary used as that path's per-block reset baseline must therefore
    /// be trained on the same representation, or it seeds the wrong
    /// distribution. This mirrors the router's chunk → BWT+MTF → RLE pipeline
    /// (including the high-entropy BWT skip) and accumulates state across the
    /// whole corpus without resetting, producing a "warmed" baseline.
    pub fn train_transformed(
        predictor: &mut dyn ProbabilityPredictor,
        training_files: &[impl AsRef<Path>],
    ) -> Result<Self> {
        use crate::chunker;
        use crate::coding::bwt_preprocess;
        use crate::format::BWT_ENTROPY_SKIP;
        for path in training_files {
            let data = std::fs::read(path.as_ref())?;
            let chunks = if data.len() < chunker::MIN_CHUNK_SIZE as usize {
                chunker::chunk_fixed_refs(&data, chunker::AVG_CHUNK_SIZE as usize)
            } else {
                chunker::chunk_data_refs(&data)
            };
            for chunk in &chunks {
                if chunk.data.len() < 8 || chunk.entropy >= BWT_ENTROPY_SKIP {
                    continue;
                }
                let Ok((_primary_index, mtf_data)) =
                    bwt_preprocess::bwt_mtf_encode_parts(chunk.data)
                else {
                    continue;
                };
                let stream = bwt_preprocess::rle_encode(&mtf_data).unwrap_or(mtf_data);
                for &byte in &stream {
                    predictor.predict();
                    predictor.update(byte);
                }
            }
        }

        let state = predictor.save_state().ok_or_else(|| {
            AetherError::Compression(format!(
                "Predictor '{}' does not support dictionary pretraining",
                predictor.name(),
            ))
        })?;
        let hash = *blake3::hash(&state).as_bytes();
        let predictor_id = predictor.predictor_id();
        Ok(Dictionary {
            predictor_id,
            state,
            hash,
        })
    }

    /// Save the dictionary to a file (.aed format).
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut f = std::fs::File::create(path)?;
        self.write_to(&mut f)
    }

    /// Write the dictionary to a writer.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        if self.state.len() > MAX_DICT_STATE_SIZE {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Dictionary state size {} exceeds {} byte limit",
                self.state.len(),
                MAX_DICT_STATE_SIZE,
            )));
        }
        w.write_all(&DICT_MAGIC)?;
        w.write_all(&[DICT_VERSION])?;
        w.write_all(&(self.predictor_id as u16).to_le_bytes())?;
        w.write_all(&(self.state.len() as u32).to_le_bytes())?;
        w.write_all(&self.state)?;
        w.write_all(&self.hash)?;
        Ok(())
    }

    /// Load a dictionary from a file.
    pub fn load(path: &Path) -> Result<Self> {
        let mut f = std::fs::File::open(path)?;
        Self::read_from(&mut f)
    }

    /// Read a dictionary from a reader.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != DICT_MAGIC {
            return Err(AetherError::InvalidMagic);
        }

        let mut version = [0u8; 1];
        r.read_exact(&mut version)?;
        if version[0] != DICT_VERSION {
            return Err(AetherError::Decompression(format!(
                "Unsupported dictionary version {} (expected {})",
                version[0], DICT_VERSION,
            )));
        }

        let mut pid_bytes = [0u8; 2];
        r.read_exact(&mut pid_bytes)?;
        let predictor_id = PredictorId::from_u16(u16::from_le_bytes(pid_bytes))?;

        let mut len_bytes = [0u8; 4];
        r.read_exact(&mut len_bytes)?;
        let state_len = u32::from_le_bytes(len_bytes) as usize;

        if state_len > MAX_DICT_STATE_SIZE {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "Dictionary state size {} exceeds {} byte limit",
                state_len, MAX_DICT_STATE_SIZE,
            )));
        }

        let mut state = vec![0u8; state_len];
        r.read_exact(&mut state)?;

        let mut stored_hash = [0u8; 32];
        r.read_exact(&mut stored_hash)?;

        // Verify integrity
        let computed_hash = *blake3::hash(&state).as_bytes();
        if computed_hash != stored_hash {
            return Err(AetherError::ChecksumMismatch {
                block_id: 0,
                expected: format!("{:x?}", &stored_hash[..4]),
                actual: format!("{:x?}", &computed_hash[..4]),
            });
        }

        Ok(Dictionary {
            predictor_id,
            state,
            hash: stored_hash,
        })
    }

    /// Apply the dictionary state to a predictor.
    ///
    /// Returns `Err` if the predictor ID doesn't match or the predictor
    /// cannot load this dictionary's state.
    pub fn apply(&self, predictor: &mut dyn ProbabilityPredictor) -> Result<()> {
        if predictor.predictor_id() != self.predictor_id {
            return Err(AetherError::Decompression(format!(
                "Dictionary was trained for predictor {:?}, but applied to '{:?}'",
                self.predictor_id,
                predictor.predictor_id(),
            )));
        }
        if !predictor.load_state(&self.state) {
            return Err(AetherError::Decompression(format!(
                "Failed to load dictionary state into predictor '{}'",
                predictor.name(),
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::Order0Model;
    use std::io::Cursor;

    #[test]
    fn dictionary_roundtrip() {
        let mut pred = Order0Model::new();
        // Train on some data
        let data = b"hello world hello world hello";
        for &b in data {
            pred.predict();
            pred.update(b);
        }

        let state = pred.save_state().unwrap();
        let hash = *blake3::hash(&state).as_bytes();
        let dict = Dictionary {
            predictor_id: PredictorId::Order0,
            state,
            hash,
        };

        // Write to buffer
        let mut buf = Vec::new();
        dict.write_to(&mut buf).unwrap();

        // Read back
        let loaded = Dictionary::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.predictor_id, PredictorId::Order0);
        assert_eq!(loaded.hash, dict.hash);
        assert_eq!(loaded.state, dict.state);

        // Apply to fresh predictor
        let mut pred2 = Order0Model::new();
        loaded.apply(&mut pred2).unwrap();

        // Predictions should match
        let p1 = pred.predict();
        let p2 = pred2.predict();
        for i in 0..256 {
            assert!((p1[i] - p2[i]).abs() < 1e-6, "Mismatch at byte {i}");
        }
    }

    #[test]
    fn dictionary_train_api() {
        let tmp = tempfile::tempdir().unwrap();
        let file1 = tmp.path().join("train1.txt");
        std::fs::write(&file1, "hello world ").unwrap();
        let file2 = tmp.path().join("train2.txt");
        std::fs::write(&file2, "hello again ").unwrap();

        let mut pred = Order0Model::new();
        let dict = Dictionary::train(&mut pred, &[file1, file2]).unwrap();

        assert_eq!(dict.predictor_id, PredictorId::Order0);
        assert!(!dict.state.is_empty());

        // Save and reload
        let dict_path = tmp.path().join("test.aed");
        dict.save(&dict_path).unwrap();
        let loaded = Dictionary::load(&dict_path).unwrap();
        assert_eq!(loaded.hash, dict.hash);
    }

    #[test]
    fn invalid_magic_rejected() {
        let data = b"XXXX\x01\x00\x00\x04\x00\x00\x00";
        let result = Dictionary::read_from(&mut Cursor::new(&data[..]));
        assert!(result.is_err());
    }

    #[test]
    fn corrupted_hash_rejected() {
        let mut pred = Order0Model::new();
        for &b in b"test" {
            pred.predict();
            pred.update(b);
        }
        let state = pred.save_state().unwrap();
        let hash = *blake3::hash(&state).as_bytes();
        let dict = Dictionary {
            predictor_id: PredictorId::Order0,
            state,
            hash,
        };

        let mut buf = Vec::new();
        dict.write_to(&mut buf).unwrap();

        // Corrupt a state byte
        if buf.len() > 10 {
            buf[10] ^= 0xFF;
        }

        let result = Dictionary::read_from(&mut Cursor::new(&buf));
        assert!(result.is_err());
    }
}
