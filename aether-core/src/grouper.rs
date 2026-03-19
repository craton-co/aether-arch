//! Semantic solid grouping.
//!
//! Groups files by content type so that similar files are compressed together
//! in the same solid block, maximizing cross-file redundancy.

use crate::analyzer::{self, RecommendedMethod};
use crate::format::ContentType;

/// Maximum total size of files in a single solid group (256 MiB).
pub const MAX_SOLID_GROUP_SIZE: u64 = 256 * 1024 * 1024;

/// A solid group: files of similar content type to be compressed together.
///
/// All files in a group share a predictor instance during compression,
/// allowing the predictor to learn cross-file patterns.
#[derive(Debug)]
pub struct SolidGroup {
    /// Unique group identifier.
    pub group_id: u32,
    /// Content type shared by all files in this group.
    pub content_type: ContentType,
    /// Recommended compression method for this group's blocks.
    pub recommended_method: RecommendedMethod,
    /// Indices into the original file list.
    pub file_indices: Vec<usize>,
    /// Total uncompressed size of all files in this group.
    pub total_size: u64,
}

/// Input for the grouper: metadata about one file to be grouped.
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// File path (used for extension-based sorting within groups).
    pub path: String,
    /// File size in bytes.
    pub size: u64,
    /// Detected content type.
    pub content_type: ContentType,
    /// Mean Shannon entropy of the file's chunks (bits per byte).
    pub mean_entropy: f64,
}

/// Group files by content type for solid archiving.
///
/// Within each type bucket, files are sorted by extension then size to
/// maximize locality of similar content. Large buckets are split into
/// sub-groups of at most [`MAX_SOLID_GROUP_SIZE`].
pub fn group_files(files: &[FileInfo]) -> Vec<SolidGroup> {
    if files.is_empty() {
        return Vec::new();
    }

    // Bucket files by content type
    let mut buckets: std::collections::BTreeMap<u16, Vec<(usize, &FileInfo)>> =
        std::collections::BTreeMap::new();

    for (idx, file) in files.iter().enumerate() {
        buckets
            .entry(file.content_type as u16)
            .or_default()
            .push((idx, file));
    }

    let mut groups = Vec::new();
    let mut group_id = 0u32;

    for (_type_key, mut bucket) in buckets {
        // Sort within bucket: by file extension, then by size (ascending)
        bucket.sort_by(|a, b| {
            let ext_a = a.1.path.rsplit('.').next().unwrap_or("");
            let ext_b = b.1.path.rsplit('.').next().unwrap_or("");
            ext_a.cmp(ext_b).then(a.1.size.cmp(&b.1.size))
        });

        // Split into sub-groups if exceeding MAX_SOLID_GROUP_SIZE
        let mut current_indices = Vec::new();
        let mut current_size = 0u64;
        let content_type = bucket[0].1.content_type;

        // Compute the dominant recommended method for this content type
        let mean_entropy: f64 = if bucket.is_empty() {
            0.0
        } else {
            bucket.iter().map(|(_, f)| f.mean_entropy).sum::<f64>() / bucket.len() as f64
        };

        for (idx, file) in &bucket {
            if current_size + file.size > MAX_SOLID_GROUP_SIZE && !current_indices.is_empty() {
                // Flush current group
                groups.push(SolidGroup {
                    group_id,
                    content_type,
                    recommended_method: analyzer::recommend_method_for(mean_entropy, content_type),
                    file_indices: std::mem::take(&mut current_indices),
                    total_size: current_size,
                });
                group_id += 1;
                current_size = 0;
            }
            current_indices.push(*idx);
            current_size += file.size;
        }

        // Flush remaining
        if !current_indices.is_empty() {
            groups.push(SolidGroup {
                group_id,
                content_type,
                recommended_method: analyzer::recommend_method_for(mean_entropy, content_type),
                file_indices: current_indices,
                total_size: current_size,
            });
            group_id += 1;
        }
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_file(path: &str, size: u64, ct: ContentType, entropy: f64) -> FileInfo {
        FileInfo {
            path: path.into(),
            size,
            content_type: ct,
            mean_entropy: entropy,
        }
    }

    #[test]
    fn empty_input() {
        assert!(group_files(&[]).is_empty());
    }

    #[test]
    fn groups_by_type() {
        let files = vec![
            make_file("a.txt", 100, ContentType::Text, 4.0),
            make_file("b.jpg", 5000, ContentType::Image, 7.8),
            make_file("c.rs", 200, ContentType::Text, 4.2),
            make_file("d.png", 3000, ContentType::Image, 7.9),
        ];

        let groups = group_files(&files);
        // Should have 2 groups: Text and Image
        assert_eq!(groups.len(), 2);

        let text_group = groups
            .iter()
            .find(|g| g.content_type == ContentType::Text)
            .unwrap();
        assert_eq!(text_group.file_indices.len(), 2);

        let image_group = groups
            .iter()
            .find(|g| g.content_type == ContentType::Image)
            .unwrap();
        assert_eq!(image_group.file_indices.len(), 2);
    }

    #[test]
    fn splits_large_groups() {
        // Create files that exceed MAX_SOLID_GROUP_SIZE
        let big = MAX_SOLID_GROUP_SIZE / 2 + 1;
        let files = vec![
            make_file("a.txt", big, ContentType::Text, 4.0),
            make_file("b.txt", big, ContentType::Text, 4.0),
            make_file("c.txt", big, ContentType::Text, 4.0),
        ];

        let groups = group_files(&files);
        // Should be split into 2 groups (a+b wouldn't fit, so a alone, then b alone, then c alone — or a, then b+c if b+c fits)
        assert!(
            groups.len() >= 2,
            "Expected at least 2 groups, got {}",
            groups.len()
        );
    }

    #[test]
    fn unique_group_ids() {
        let files = vec![
            make_file("a.txt", 100, ContentType::Text, 4.0),
            make_file("b.exe", 200, ContentType::Executable, 6.0),
            make_file("c.jpg", 300, ContentType::Image, 7.8),
        ];

        let groups = group_files(&files);
        let ids: Vec<u32> = groups.iter().map(|g| g.group_id).collect();
        let unique: std::collections::HashSet<u32> = ids.iter().cloned().collect();
        assert_eq!(ids.len(), unique.len(), "Group IDs must be unique");
    }
}
