use std::{collections::HashMap, time::SystemTime};

use crate::models::table::ColumnData;

pub enum FeatureStoreError {
    MaxVersionsTooLow { min: usize, actual: usize },
    RawColumnPresent,
}

pub enum Transformation {
    FillWith,
    FillMean,
    ForwardFill,
    BackwardFill,
    MinMaxScale,
    ZScoreStandardization,
    Custom(String),
}

pub struct ColumnVersion {
    data: ColumnData,
    version_number: usize,
    name: String,
    transformation: Transformation,
    timestamp: SystemTime,
}

pub struct FeatureStore {
    versions: HashMap<String, Vec<ColumnVersion>>,
    max_versions: usize,
}

impl FeatureStore {
    pub fn new(max_versions: usize) -> Result<Self, FeatureStoreError> {
        if max_versions == 0 {
            return Err(FeatureStoreError::MaxVersionsTooLow {
                min: 1,
                actual: max_versions,
            });
        }

        Ok(Self {
            versions: HashMap::new(),
            max_versions,
        })
    }

    pub fn commit(
        &mut self,
        data: ColumnData,
        transformation: Transformation,
    ) -> Result<usize, FeatureStoreError> {
        let name: String;
        match data.extract_name() {
            Some(n) => {
                name = n.to_string();
            }
            None => return Err(FeatureStoreError::RawColumnPresent),
        }
        let versions = self.versions.entry(name.clone()).or_insert_with(Vec::new);

        let max_version = versions.iter().map(|v| v.version_number).max().unwrap_or(0);

        versions.push(ColumnVersion {
            data,
            version_number: max_version + 1,
            name,
            transformation,
            timestamp: SystemTime::now(),
        });

        Ok(max_version + 1)
    }
}
