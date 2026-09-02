use std::collections::HashMap;

use candle_core::{Device, Tensor};
use candle_nn::Module;

use crate::{
    feature_store::{FeatureStore, FeatureStoreError, SnapshotId},
    ml::training::LoadedModel,
    models::table::TableError,
};

/// Errors that can occur while running inference.
#[derive(Debug)]
pub enum InferenceError {
    /// [`run_inference_from_snapshot`] failed while rebuilding a [`Table`](crate::models::table::Table)
    /// from the given snapshot (via [`FeatureStore::reconstruct_table`]).
    TableReconstructionFailed(FeatureStoreError),
    /// [`run_inference_from_snapshot`] failed while looking up the snapshot itself (via
    /// [`FeatureStore::get_snapshot`]) to recover its feature order.
    SnapshotReconstructionFailed(FeatureStoreError),
    /// [`run_inference_from_snapshot`] failed while converting the reconstructed table's
    /// columns into a tensor (via [`Table::to_feature_tensor`](crate::models::table::Table::to_feature_tensor)).
    FeatureTensorFailed(TableError),
    /// The loaded model's own `.forward()` call failed.
    ForwardFailed(candle_core::Error),
    /// [`run_inference_from_values`] was missing a value for a feature the model requires.
    MissingFeature { name: String },
    /// [`run_inference_from_values`] was given a value for a feature the model doesn't expect.
    UnexpectedFeature,
    /// [`run_inference_from_values`] failed while building the input tensor from the given
    /// values.
    TensorCreationFailed(candle_core::Error),
}

/// Runs inference (scenario "A" — batch, over already-known data) against every row of the
/// [`Snapshot`](crate::feature_store::Snapshot) named by `snapshot_id`: reconstructs a `Table`
/// from it (via [`FeatureStore::reconstruct_table`]), converts it to a tensor in the snapshot's
/// own recorded feature order (via `to_feature_tensor`, not `to_tensor` — there's no known
/// target to supply here), and runs the model on it.
///
/// Suited to scoring/evaluating a model against data the [`FeatureStore`] already holds, as
/// opposed to [`run_inference_from_values`], which predicts from brand-new values the caller
/// supplies directly.
///
/// # Errors
/// Returns [`InferenceError::TableReconstructionFailed`] or
/// [`InferenceError::SnapshotReconstructionFailed`] if reading the snapshot fails,
/// [`InferenceError::FeatureTensorFailed`] if building the tensor fails, or
/// [`InferenceError::ForwardFailed`] if the model itself fails.
pub fn run_inference_from_snapshot(
    loaded_model: &LoadedModel,
    feature_store: &FeatureStore,
    snapshot_id: &SnapshotId,
    device: &Device,
) -> Result<Tensor, InferenceError> {
    let table = feature_store
        .reconstruct_table(snapshot_id)
        .map_err(InferenceError::TableReconstructionFailed)?;
    let snapshot = feature_store
        .get_snapshot(snapshot_id)
        .map_err(InferenceError::SnapshotReconstructionFailed)?;
    let feature_columns: Vec<String> = snapshot
        .ordered_features()
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    let x = table
        .to_feature_tensor(feature_columns, device)
        .map_err(InferenceError::FeatureTensorFailed)?;
    run(loaded_model, &x)
}

pub fn run_inference_from_values(
    loaded_model: &LoadedModel,
    feature_names: &[String],
    values: &HashMap<String, f64>,
    device: &Device,
) -> Result<Tensor, InferenceError> {
    let mut row: Vec<f32> = Vec::with_capacity(feature_names.len());
    for name in feature_names {
        let value = values
            .get(name)
            .ok_or_else(|| InferenceError::MissingFeature { name: name.clone() })?;
        row.push(*value as f32);
    }

    if values.len() != feature_names.len() {
        return Err(InferenceError::UnexpectedFeature);
    }

    let x = Tensor::from_vec(row, (1, feature_names.len()), device)
        .map_err(InferenceError::TensorCreationFailed)?;
    run(loaded_model, &x)
}

fn run(loaded_model: &LoadedModel, x: &Tensor) -> Result<Tensor, InferenceError> {
    match loaded_model {
        LoadedModel::Linear(linear) => linear.forward(x).map_err(InferenceError::ForwardFailed),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        feature_store::Transformation,
        ml::training::train_linear_regression,
        models::{column::Column, table::ColumnData},
    };

    use super::*;

    #[test]
    fn run_inference_from_snapshot_produces_predictions() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1f32], [2.], [3.], [4.]], &device).unwrap();
        let y = Tensor::new(&[3f32, 5., 7., 9.], &device).unwrap();
        let training_output = train_linear_regression(&x, &y, 5, 0.05, &device).unwrap();
        let loaded_model = LoadedModel::Linear(training_output.model);

        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
            "age".to_string(),
        )
        .unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        let snapshot_id = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();

        let result = run_inference_from_snapshot(&loaded_model, &store, &snapshot_id, &device);

        assert!(result.is_ok());
        let predictions = result.unwrap();
        assert_eq!(predictions.dims(), &[4, 1]);
    }

    #[test]
    fn run_inference_from_values_produces_predictions_regardless_of_map_order() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1f32], [2.], [3.], [4.]], &device).unwrap();
        let y = Tensor::new(&[3f32, 5., 7., 9.], &device).unwrap();
        let training_output = train_linear_regression(&x, &y, 5, 0.05, &device).unwrap();
        let loaded_model = LoadedModel::Linear(training_output.model);

        let feature_names = vec!["age".to_string()];
        let mut values = HashMap::new();
        values.insert("age".to_string(), 2.0);

        let result = run_inference_from_values(&loaded_model, &feature_names, &values, &device);

        assert!(result.is_ok());
        let predictions = result.unwrap();
        assert_eq!(predictions.dims(), &[1, 1]);
    }

    #[test]
    fn run_inference_from_values_fails_for_missing_feature() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1f32], [2.], [3.], [4.]], &device).unwrap();
        let y = Tensor::new(&[3f32, 5., 7., 9.], &device).unwrap();
        let training_output = train_linear_regression(&x, &y, 5, 0.05, &device).unwrap();
        let loaded_model = LoadedModel::Linear(training_output.model);

        let feature_names = vec!["age".to_string(), "income".to_string()];
        let mut values = HashMap::new();
        values.insert("age".to_string(), 2.0);

        let result = run_inference_from_values(&loaded_model, &feature_names, &values, &device);

        match result {
            Err(InferenceError::MissingFeature { name }) => assert_eq!(name, "income".to_string()),
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn run_inference_from_values_fails_for_unexpected_feature() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1f32], [2.], [3.], [4.]], &device).unwrap();
        let y = Tensor::new(&[3f32, 5., 7., 9.], &device).unwrap();
        let training_output = train_linear_regression(&x, &y, 5, 0.05, &device).unwrap();
        let loaded_model = LoadedModel::Linear(training_output.model);

        let feature_names = vec!["age".to_string()];
        let mut values = HashMap::new();
        values.insert("age".to_string(), 2.0);
        values.insert("extra".to_string(), 9.0);

        let result = run_inference_from_values(&loaded_model, &feature_names, &values, &device);

        match result {
            Err(InferenceError::UnexpectedFeature) => {}
            _ => panic!("Unexpected result."),
        }
    }
}
