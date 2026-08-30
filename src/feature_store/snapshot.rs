use std::time::SystemTime;

use sha2::{Digest, Sha256};

use super::FeatureStore;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SnapshotId(String);

pub struct Snapshot {
    id: SnapshotId,
    ordered_features: Vec<(String, usize)>,
    timestamp: SystemTime,
    label: Option<String>,
}

fn compute_snapshot_id(ordered_features: &[(String, usize)]) -> SnapshotId {
    let mut hasher = Sha256::new();
    for (name, version) in ordered_features.iter() {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&(*version as u64).to_le_bytes());
    }
    let hash256 = hasher.finalize();
    let hex_string: String = hash256.iter().map(|byte| format!("{:02x}", byte)).collect();
    return SnapshotId(hex_string);
}
