use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::models::table::{ColumnState, TableError};

use super::Table;

impl Table {
    /// Splits the table's rows into a train/test pair of new, independent tables, leaving the
    /// original table untouched.
    ///
    /// `ratio` is the fraction of rows assigned to the train table (e.g. `0.8` for an 80/20
    /// split). Row order is shuffled deterministically via `seed` — the same `seed` and `ratio`
    /// always produce the same split. `columns` selects which columns to include: `None`
    /// includes every column the table currently has (including checked-out ones, which then
    /// triggers the error below), `Some(names)` includes only the named columns.
    ///
    /// # Errors
    /// Returns [`TableError::ColumnNotFound`] if a requested column doesn't exist, or
    /// [`TableError::ColumnNotAvailable`] if a requested column is currently checked out via
    /// [`Table::get_column`].
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
            None => column_names = self.checkouts.keys().cloned().collect(),
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
    fn train_test_split_produces_correct_row_counts() {
        let file = write_temp_csv("id\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n");
        let table = Table::from_csv(file.path()).unwrap();

        let (train, test) = table.train_test_split(0.7, 42, None).unwrap();

        assert_eq!(train.row_count(), 7);
        assert_eq!(test.row_count(), 3);
    }

    #[test]
    fn train_test_split_is_deterministic_with_same_seed() {
        let file = write_temp_csv("id\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n");
        let table = Table::from_csv(file.path()).unwrap();

        let (mut train1, _) = table.train_test_split(0.7, 42, None).unwrap();
        let (mut train2, _) = table.train_test_split(0.7, 42, None).unwrap();

        let col1 = train1.get_column::<i64>("id").unwrap();
        let col2 = train2.get_column::<i64>("id").unwrap();

        let values1: Vec<Option<i64>> = (0..col1.len())
            .map(|i| col1.get(i).cloned().flatten())
            .collect();
        let values2: Vec<Option<i64>> = (0..col2.len())
            .map(|i| col2.get(i).cloned().flatten())
            .collect();

        assert_eq!(values1, values2);
    }

    #[test]
    fn train_test_split_includes_only_requested_columns() {
        let file = write_temp_csv("id,name\n1,Ali\n2,Ayşe\n3,Mehmet\n4,Zeynep\n");
        let table = Table::from_csv(file.path()).unwrap();

        let (train, _) = table
            .train_test_split(0.5, 1, Some(vec!["id".to_string()]))
            .unwrap();

        assert_eq!(train.column_count(), 1);
        assert!(train.data.contains_key("id"));
    }

    #[test]
    fn train_test_split_fails_when_column_checked_out() {
        let file = write_temp_csv("id,name\n1,Ali\n2,Ayşe\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let _checked_out = table.get_column::<i64>("id").unwrap();

        let result = table.train_test_split(0.5, 1, None);

        match result {
            Err(TableError::ColumnNotAvailable { name }) => assert_eq!(name, "id".to_string()),
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn train_test_split_fails_when_column_not_found() {
        let file = write_temp_csv("id,name\n1,Ali\n2,Ayşe\n");
        let table = Table::from_csv(file.path()).unwrap();

        let result = table.train_test_split(0.5, 1, Some(vec!["city".to_string()]));

        match result {
            Err(TableError::ColumnNotFound { name }) => assert_eq!(name, "city".to_string()),
            _ => panic!("Unexpected result."),
        }
    }
}
