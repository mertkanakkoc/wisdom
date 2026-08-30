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

impl FeatureStore {
    pub fn detect_drift(
        &self,
        name: &str,
        baseline_version: usize,
        current_version: usize,
        threshold: f64,
    ) -> Result<bool, FeatureStoreError> {
        let baseline = self.get_version(name, baseline_version)?;
        let current = self.get_version(name, current_version)?;

        let baseline_values = to_f64_vec(&baseline.data, name)?;
        if baseline_values.is_empty() {
            return Err(FeatureStoreError::AllValuesMissing {
                name: name.to_string(),
                version: baseline_version,
            });
        }
        let current_values = to_f64_vec(&current.data, name)?;
        if current_values.is_empty() {
            return Err(FeatureStoreError::AllValuesMissing {
                name: name.to_string(),
                version: current_version,
            });
        }

        let baseline_mean = baseline_values.iter().sum::<f64>() / baseline_values.len() as f64;
        let baseline_variance = baseline_values
            .iter()
            .map(|x| (x - baseline_mean).powi(2))
            .sum::<f64>()
            / baseline_values.len() as f64;
        let baseline_std = baseline_variance.sqrt();

        let current_mean = current_values.iter().sum::<f64>() / current_values.len() as f64;

        let diff = (current_mean - baseline_mean).abs();
        let score = if baseline_std == 0.0 {
            diff
        } else {
            diff / baseline_std
        };

        Ok(score > threshold)
    }
}
