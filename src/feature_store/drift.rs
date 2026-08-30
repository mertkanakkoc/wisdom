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
    /// Compares two committed versions of the same feature (`baseline_version` vs
    /// `current_version`) and reports whether they've drifted apart by more than `threshold`.
    ///
    /// The drift score is the shift in mean between the two versions' present (non-missing)
    /// values, normalized by the baseline's (population) standard deviation:
    /// `|current_mean - baseline_mean| / baseline_std`. If `baseline_std` is `0` (every present
    /// baseline value is identical), the plain absolute mean difference is used instead, to
    /// avoid dividing by zero. Returns `true` if this score is greater than `threshold`.
    ///
    /// Only the same numeric types [`Table::to_tensor`] accepts (`Int`/`Float`/`Bool`) are
    /// supported, matching what's actually trainable; `Int`/`Bool` values are cast to `f64`.
    /// Missing values are skipped when computing the mean/standard deviation, not treated as
    /// `0`.
    ///
    /// # Errors
    /// Returns [`FeatureStoreError::VersionNotFound`] if either version doesn't exist,
    /// [`FeatureStoreError::NonNumericFeature`] if the feature is [`ColumnData::Text`] or
    /// [`ColumnData::Raw`], or [`FeatureStoreError::AllValuesMissing`] if either version's
    /// values are entirely missing.
    ///
    /// [`Table::to_tensor`]: crate::models::table::Table::to_tensor
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

#[cfg(test)]
mod tests {
    use crate::{feature_store::Transformation, models::column::Column};

    use super::*;

    #[test]
    fn detect_drift_returns_false_when_values_are_similar() {
        let mut store = FeatureStore::new(5).unwrap();
        let baseline =
            Column::new_from_parsed(vec![Some(1.0), Some(2.0), Some(3.0)], "age".to_string())
                .unwrap();
        let current =
            Column::new_from_parsed(vec![Some(1.1), Some(2.1), Some(3.1)], "age".to_string())
                .unwrap();
        store
            .commit(
                ColumnData::Float(baseline),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        store
            .commit(
                ColumnData::Float(current),
                Transformation::Custom("v2".to_string()),
            )
            .unwrap();

        let result = store.detect_drift("age", 1, 2, 1.0);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false);
    }

    #[test]
    fn detect_drift_returns_true_when_values_shift_significantly() {
        let mut store = FeatureStore::new(5).unwrap();
        let baseline =
            Column::new_from_parsed(vec![Some(1.0), Some(2.0), Some(3.0)], "age".to_string())
                .unwrap();
        let current = Column::new_from_parsed(
            vec![Some(100.0), Some(200.0), Some(300.0)],
            "age".to_string(),
        )
        .unwrap();
        store
            .commit(
                ColumnData::Float(baseline),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        store
            .commit(
                ColumnData::Float(current),
                Transformation::Custom("v2".to_string()),
            )
            .unwrap();

        let result = store.detect_drift("age", 1, 2, 1.0);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), true);
    }

    #[test]
    fn detect_drift_uses_absolute_difference_when_baseline_std_is_zero() {
        let mut store = FeatureStore::new(5).unwrap();
        let baseline =
            Column::new_from_parsed(vec![Some(5.0), Some(5.0), Some(5.0)], "age".to_string())
                .unwrap();
        let current =
            Column::new_from_parsed(vec![Some(5.5), Some(5.5), Some(5.5)], "age".to_string())
                .unwrap();
        store
            .commit(
                ColumnData::Float(baseline),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        store
            .commit(
                ColumnData::Float(current),
                Transformation::Custom("v2".to_string()),
            )
            .unwrap();

        let result = store.detect_drift("age", 1, 2, 1.0);

        // baseline_std == 0, diff = |5.5 - 5.0| = 0.5, threshold = 1.0
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false);
    }

    #[test]
    fn detect_drift_fails_for_non_numeric_feature() {
        let mut store = FeatureStore::new(5).unwrap();
        let baseline =
            Column::new_from_parsed(vec![Some("a".to_string())], "city".to_string()).unwrap();
        let current =
            Column::new_from_parsed(vec![Some("b".to_string())], "city".to_string()).unwrap();
        store
            .commit(
                ColumnData::Text(baseline),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        store
            .commit(
                ColumnData::Text(current),
                Transformation::Custom("v2".to_string()),
            )
            .unwrap();

        let result = store.detect_drift("city", 1, 2, 1.0);

        match result {
            Err(FeatureStoreError::NonNumericFeature { name }) => {
                assert_eq!(name, "city".to_string());
            }
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn detect_drift_fails_when_baseline_is_entirely_missing() {
        let mut store = FeatureStore::new(5).unwrap();
        let baseline: Column<f64> =
            Column::new_from_parsed(vec![None, None], "age".to_string()).unwrap();
        let current = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(baseline),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        store
            .commit(
                ColumnData::Float(current),
                Transformation::Custom("v2".to_string()),
            )
            .unwrap();

        let result = store.detect_drift("age", 1, 2, 1.0);

        match result {
            Err(FeatureStoreError::AllValuesMissing { name, version }) => {
                assert_eq!(name, "age");
                assert_eq!(version, 1);
            }
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn detect_drift_fails_for_nonexistent_version() {
        let mut store = FeatureStore::new(5).unwrap();
        let baseline = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(baseline),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();

        let result = store.detect_drift("age", 1, 2, 1.0);

        match result {
            Err(FeatureStoreError::VersionNotFound { name, version }) => {
                assert_eq!(name, "age".to_string());
                assert_eq!(version, 2);
            }
            _ => panic!("Unexpected result."),
        }
    }
}
