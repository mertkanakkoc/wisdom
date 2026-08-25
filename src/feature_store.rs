use std::{collections::HashMap, time::SystemTime};

use crate::models::table::ColumnData;

pub enum FeatureStoreError {
    MaxVersionsTooLow { min: usize, actual: usize },
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
}
