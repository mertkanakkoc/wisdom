use candle_core::{Device, Tensor};

use crate::models::table::{ColumnData, TableError};

use super::Table;

impl Table {
    pub fn to_tensor(
        &self,
        feature_columns: Vec<String>,
        target_column: String,
        device: &Device,
    ) -> Result<(Tensor, Tensor), TableError> {
        if feature_columns.contains(&target_column) {
            return Err(TableError::TargetColumnInFeatures {
                name: target_column.clone(),
            });
        }
        let mut combined = feature_columns.clone();
        combined.push(target_column.clone());

        if let Err(e) = self.check_missing_and_type(&combined) {
            return Err(e);
        }

        let mut combined_for_x: Vec<f32> = vec![];
        let mut for_y: Vec<f32> = vec![];

        let row_count = self.row_count();
        for i in 0..row_count {
            for name in combined.iter() {
                let from_table = self
                    .data
                    .get(name)
                    .ok_or_else(|| TableError::ColumnNotFound { name: name.clone() })?;
                match from_table {
                    ColumnData::Raw(_) | ColumnData::Text(_) => {
                        return Err(TableError::NonNumericColumn { name: name.clone() });
                    }
                    ColumnData::Bool(column) => {
                        if *name != target_column {
                            combined_for_x.push(match column.get(i) {
                                Some(Some(x)) => {
                                    if *x {
                                        1.0_f32
                                    } else {
                                        0.0_f32
                                    }
                                }
                                _ => {
                                    return Err(TableError::MissingValuesPresent {
                                        name: name.clone(),
                                    });
                                }
                            });
                        } else {
                            for_y.push(match column.get(i) {
                                Some(Some(x)) => {
                                    if *x {
                                        1.0_f32
                                    } else {
                                        0.0_f32
                                    }
                                }
                                _ => {
                                    return Err(TableError::MissingValuesPresent {
                                        name: name.clone(),
                                    });
                                }
                            });
                        }
                    }
                    ColumnData::Int(column) => {
                        if *name != target_column {
                            combined_for_x.push(match column.get(i) {
                                Some(Some(x)) => *x as f32,
                                _ => {
                                    return Err(TableError::MissingValuesPresent {
                                        name: name.clone(),
                                    });
                                }
                            });
                        } else {
                            for_y.push(match column.get(i) {
                                Some(Some(x)) => *x as f32,
                                _ => {
                                    return Err(TableError::MissingValuesPresent {
                                        name: name.clone(),
                                    });
                                }
                            });
                        }
                    }
                    ColumnData::Float(column) => {
                        if *name != target_column {
                            combined_for_x.push(match column.get(i) {
                                Some(Some(x)) => *x as f32,
                                _ => {
                                    return Err(TableError::MissingValuesPresent {
                                        name: name.clone(),
                                    });
                                }
                            });
                        } else {
                            for_y.push(match column.get(i) {
                                Some(Some(x)) => *x as f32,
                                _ => {
                                    return Err(TableError::MissingValuesPresent {
                                        name: name.clone(),
                                    });
                                }
                            });
                        }
                    }
                }
            }
        }

        let tensor_x = Tensor::from_vec(combined_for_x, (row_count, feature_columns.len()), device)
            .map_err(TableError::TensorCreationFailed)?;
        let tensor_y = Tensor::from_vec(for_y, (row_count,), device)
            .map_err(TableError::TensorCreationFailed)?;

        Ok((tensor_x, tensor_y))
    }

    fn check_missing_and_type(&self, columns: &[String]) -> Result<(), TableError> {
        for name in columns.iter() {
            let from_table = self
                .data
                .get(name)
                .ok_or_else(|| TableError::ColumnNotFound { name: name.clone() })?;

            match from_table {
                ColumnData::Raw(_) | ColumnData::Text(_) => {
                    return Err(TableError::NonNumericColumn { name: name.clone() });
                }
                ColumnData::Bool(column) => {
                    if column.missing_count() > 0 {
                        return Err(TableError::MissingValuesPresent { name: name.clone() });
                    }
                }
                ColumnData::Float(column) => {
                    if column.missing_count() > 0 {
                        return Err(TableError::MissingValuesPresent { name: name.clone() });
                    }
                }
                ColumnData::Int(column) => {
                    if column.missing_count() > 0 {
                        return Err(TableError::MissingValuesPresent { name: name.clone() });
                    }
                }
            }
        }
        Ok(())
    }
}
