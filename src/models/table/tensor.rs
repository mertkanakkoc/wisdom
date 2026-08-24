use candle_core::{Device, Tensor};

use crate::models::table::{ColumnData, TableError};

use super::Table;

impl Table {
    /// Converts the given columns into a pair of `candle` tensors ready for training: `X`
    /// (features, shape `(row_count, feature_columns.len())`) and `y` (target, shape
    /// `(row_count,)`). Values are read in `feature_columns`' order and cast to `f32`
    /// (`Bool` becomes `1.0`/`0.0`); the underlying table is only borrowed, not consumed or
    /// checked out.
    ///
    /// Every named column must already be materialized to `Int`/`Float`/`Bool` — a column still
    /// sitting as [`ColumnData::Raw`] (never checked out via [`Table::get_column`] and written
    /// back via [`Table::update_column`]) is rejected as [`TableError::NonNumericColumn`], even
    /// if its raw strings would actually parse as numbers. Materializing it is the caller's
    /// responsibility, not something this function does implicitly.
    ///
    /// # Errors
    /// Returns [`TableError::TargetColumnInFeatures`] if `target_column` also appears in
    /// `feature_columns`, [`TableError::ColumnNotFound`] if any named column doesn't exist,
    /// [`TableError::NonNumericColumn`] if any named column is `Text`/`Raw`,
    /// [`TableError::MissingValuesPresent`] if any named column has a missing value, or
    /// [`TableError::TensorCreationFailed`] if `candle` itself rejects the resulting data/shape.
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

        let mut columns_for_x: Vec<&ColumnData> = vec![];
        for feature_name in feature_columns.iter() {
            columns_for_x.push({
                self.data
                    .get(feature_name)
                    .ok_or_else(|| TableError::ColumnNotFound {
                        name: feature_name.clone(),
                    })?
            });
        }
        let column_for_y =
            self.data
                .get(&target_column)
                .ok_or_else(|| TableError::ColumnNotFound {
                    name: target_column.clone(),
                })?;

        let row_count = self.row_count();
        for i in 0..row_count {
            for (index, name) in combined.iter().enumerate() {
                let from_table: &ColumnData;
                if *name == target_column {
                    from_table = column_for_y;
                } else {
                    from_table = columns_for_x[index];
                }
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

#[cfg(test)]
mod tests {

    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_temp_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", content).unwrap();
        file
    }

    #[test]
    fn to_tensor_produces_correct_x_and_y() {
        let file = write_temp_csv("age,score,label\n25,1.5,0\n30,2.5,1\n35,3.5,0\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        for name in ["age", "score", "label"] {
            let column = table.get_column::<f64>(name).unwrap();
            table.update_column(name, column).unwrap();
        }

        let device = Device::Cpu;
        let (x, y) = table
            .to_tensor(
                vec!["age".to_string(), "score".to_string()],
                "label".to_string(),
                &device,
            )
            .unwrap();

        let x_values = x.to_vec2::<f32>().unwrap();
        let y_values = y.to_vec1::<f32>().unwrap();

        assert_eq!(
            x_values,
            vec![vec![25.0, 1.5], vec![30.0, 2.5], vec![35.0, 3.5]]
        );
        assert_eq!(y_values, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn to_tensor_fails_when_target_in_features() {
        let file = write_temp_csv("age,label\n25,0\n30,1\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        for name in ["age", "label"] {
            let column = table.get_column::<f64>(name).unwrap();
            table.update_column(name, column).unwrap();
        }

        let device = Device::Cpu;
        let result = table.to_tensor(
            vec!["age".to_string(), "label".to_string()],
            "label".to_string(),
            &device,
        );

        match result {
            Err(TableError::TargetColumnInFeatures { name }) => {
                assert_eq!(name, "label".to_string())
            }
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn to_tensor_fails_when_column_not_found() {
        let file = write_temp_csv("age,label\n25,0\n30,1\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        for name in ["age", "label"] {
            let column = table.get_column::<f64>(name).unwrap();
            table.update_column(name, column).unwrap();
        }

        let device = Device::Cpu;
        let result = table.to_tensor(vec!["city".to_string()], "label".to_string(), &device);

        match result {
            Err(TableError::ColumnNotFound { name }) => assert_eq!(name, "city".to_string()),
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn to_tensor_fails_for_non_numeric_column() {
        let file = write_temp_csv("age,city\n25,Istanbul\n30,Ankara\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let column = table.get_column::<f64>("age").unwrap();
        table.update_column("age", column).unwrap();

        let device = Device::Cpu;
        let result = table.to_tensor(vec!["city".to_string()], "age".to_string(), &device);

        match result {
            Err(TableError::NonNumericColumn { name }) => assert_eq!(name, "city".to_string()),
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn to_tensor_fails_when_missing_values_present() {
        let file = write_temp_csv("age,label\n25,0\n,1\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        for name in ["age", "label"] {
            let column = table.get_column::<f64>(name).unwrap();
            table.update_column(name, column).unwrap();
        }

        let device = Device::Cpu;
        let result = table.to_tensor(vec!["age".to_string()], "label".to_string(), &device);

        match result {
            Err(TableError::MissingValuesPresent { name }) => {
                assert_eq!(name, "age".to_string())
            }
            _ => panic!("Unexpected result."),
        }
    }
}
