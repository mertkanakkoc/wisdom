use crate::{feature_store::FeatureStoreError, models::table::ColumnData};

use super::FeatureStore;

fn to_f64_vec(data: &ColumnData, name: &str) -> Result<Vec<f64>, FeatureStoreError> {
    match data {
        ColumnData::Float(c) => Ok(c.present_values().copied().collect()),
        ColumnData::Int(c) => Ok(c.present_values().map(|x| *x as f64).collect()),
        ColumnData::Bool(c) => Ok(c
            .present_values()
            .map(|b| if *b { 1.0 } else { 0.0 })
            .collect()),
        ColumnData::Text(_) | ColumnData::Raw(_) => Err(FeatureStoreError::NonNumericFeature {
            name: name.to_string(),
        }),
    }
}
