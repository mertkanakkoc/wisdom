use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::models::table::{ColumnState, TableError};

use super::Table;

impl Table {
    pub fn train_test_split(
        &self,
        ratio: f64,
        seed: u64,
        columns: Option<Vec<String>>,
    ) -> Result<(Table, Table), TableError> {
        let mut indices: Vec<usize> = (0..self.row_count()).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);

        let split_point = (indices.len() as f64 * ratio) as usize;
        let train_indices = &indices[..split_point];
        let test_indices = &indices[split_point..];

        let mut train_table = Table::default();
        let mut test_table = Table::default();

        train_table.file_path = self.file_path.clone();
        test_table.file_path = self.file_path.clone();

        let mut column_names: Vec<String> = vec![];
        match columns {
            Some(names) => column_names = names,
            None => column_names = self.data.keys().cloned().collect(),
        }

        for name in column_names.iter() {
            match self.checkouts.get(name) {
                Some(ColumnState::CheckedOut) => {
                    return Err(TableError::ColumnNotAvailable { name: name.clone() });
                }
                Some(ColumnState::Available) => {
                    let column_data = self
                        .data
                        .get(name)
                        .ok_or_else(|| TableError::ColumnNotFound { name: name.clone() })?;
                    train_table
                        .data
                        .insert(name.clone(), column_data.select_rows(train_indices));
                    test_table
                        .data
                        .insert(name.clone(), column_data.select_rows(test_indices));

                    train_table
                        .checkouts
                        .insert(name.clone(), ColumnState::Available);
                    test_table
                        .checkouts
                        .insert(name.clone(), ColumnState::Available);
                }
                None => {
                    return Err(TableError::ColumnNotFound {
                        name: name.to_string(),
                    });
                }
            }
        }

        Ok((train_table, test_table))
    }
}
