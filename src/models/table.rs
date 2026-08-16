use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::models::column::{Column, ColumnError};

pub enum ColumnState {
    Available,
    CheckedOut,
}

#[derive(Debug)]
pub enum TableError {
    OpenFileError(csv::Error),
    HeadersReadingError(csv::Error),
    RowReadingError(csv::Error),
    ColumnNotAvailable { name: String },
    ColumnNotFound { name: String },
    ParseFailed { name: String, source: ColumnError },
    TypeMismatch { name: String },
    ColumnLengthMismatch { length: usize },
    ColumnNotCheckedOut { name: String },
    ColumnNotUpdated { name: String },
    ColumnAlreadyExists { name: String },
}

pub enum ColumnData {
    Int(Column<i64>),
    Float(Column<f64>),
    Text(Column<String>),
    Bool(Column<bool>),
    Raw(Vec<Option<String>>),
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int(c) => c.len(),
            ColumnData::Text(c) => c.len(),
            ColumnData::Bool(c) => c.len(),
            ColumnData::Float(c) => c.len(),
            ColumnData::Raw(v) => v.len(),
        }
    }
}

pub struct Table {
    file_path: PathBuf,
    data: HashMap<String, ColumnData>,
    checkouts: HashMap<String, ColumnState>,
}

impl Table {
    pub fn from_csv(path: &Path) -> Result<Self, TableError> {
        let mut rdr = csv::Reader::from_path(path).map_err(TableError::OpenFileError)?;

        let headers: Vec<String> = rdr
            .headers()
            .map_err(TableError::HeadersReadingError)?
            .iter()
            .map(|h| h.to_string())
            .collect();

        let mut data: HashMap<String, ColumnData> = rdr
            .headers()
            .map_err(TableError::HeadersReadingError)?
            .iter()
            .map(|h| (h.to_string(), ColumnData::Raw(Vec::new())))
            .collect();

        let checkouts: HashMap<String, ColumnState> = rdr
            .headers()
            .map_err(TableError::HeadersReadingError)?
            .iter()
            .map(|h| (h.to_string(), ColumnState::Available))
            .collect();

        for result in rdr.records() {
            let record = result.map_err(TableError::RowReadingError)?;
            for (index, header) in headers.iter().enumerate() {
                let cell = record.get(index).and_then(|s| {
                    if s.is_empty() {
                        None
                    } else {
                        Some(s.to_string())
                    }
                });
                if let Some(ColumnData::Raw(vec)) = data.get_mut(header) {
                    vec.push(cell);
                }
            }
        }

        Ok(Self {
            file_path: path.to_path_buf(),
            data: data,
            checkouts: checkouts,
        })
    }

    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    pub fn row_count(&self) -> usize {
        self.data.values().next().map(|c| c.len()).unwrap_or(0)
    }

    pub fn column_count(&self) -> usize {
        self.data.len()
    }

    pub fn element_count(&self) -> usize {
        self.column_count() * self.row_count()
    }

    pub fn get_column<T>(&mut self, name: &str) -> Result<Column<T>, TableError>
    where
        T: ColumnDataVariant + std::str::FromStr,
        T::Err: std::error::Error + 'static,
    {
        if let Some(checkout) = self.checkouts.get(name) {
            match checkout {
                ColumnState::Available => {}
                ColumnState::CheckedOut => {
                    return Err(TableError::ColumnNotAvailable {
                        name: name.to_string(),
                    });
                }
            }
        } else {
            return Err(TableError::ColumnNotFound {
                name: name.to_string(),
            });
        }

        let column_from_data =
            self.data
                .remove(name)
                .ok_or_else(|| TableError::ColumnNotFound {
                    name: name.to_string(),
                })?;

        let column = match column_from_data {
            ColumnData::Raw(value) => {
                let value_backup = value.clone();
                Column::new_from_raw(value, name.to_string()).map_err(|e| {
                    self.data
                        .insert(name.to_string(), ColumnData::Raw(value_backup));
                    TableError::ParseFailed {
                        name: name.to_string(),
                        source: e,
                    }
                })?
            }
            other => T::unwrap(other).map_err(|returned_data| {
                self.data.insert(name.to_string(), returned_data);
                TableError::TypeMismatch {
                    name: name.to_string(),
                }
            })?,
        };
        self.checkouts
            .insert(name.to_string(), ColumnState::CheckedOut);
        Ok(column)
    }

    pub fn update_column<T>(&mut self, name: &str, column: Column<T>) -> Result<String, TableError>
    where
        T: ColumnDataVariant + std::str::FromStr,
        T::Err: std::error::Error + 'static,
    {
        let updated_column_len = column.len();
        let row_count = self.row_count();
        if row_count != 0 && row_count != updated_column_len {
            return Err(TableError::ColumnLengthMismatch { length: row_count });
        }

        match self.checkouts.get(name) {
            Some(ColumnState::Available) => {
                return Err(TableError::ColumnNotCheckedOut {
                    name: name.to_string(),
                });
            }
            None => {
                return Err(TableError::ColumnNotFound {
                    name: name.to_string(),
                });
            }
            Some(ColumnState::CheckedOut) => {
                let column_for_table = T::wrap(column);
                self.data.insert(name.to_string(), column_for_table);
                self.checkouts
                    .insert(name.to_string(), ColumnState::Available);
                Ok(format!("Column '{}' updated.", name.to_string()))
            }
        }
    }

    pub fn add_column<T>(&mut self, column: Column<T>) -> Result<String, TableError>
    where
        T: ColumnDataVariant + std::str::FromStr,
        T::Err: std::error::Error + 'static,
    {
        let new_column_name = column.name().to_string();
        if self.data.contains_key(&new_column_name) {
            return Err(TableError::ColumnAlreadyExists {
                name: new_column_name,
            });
        }

        let row_count = self.row_count();
        if row_count != 0 && row_count != column.len() {
            return Err(TableError::ColumnLengthMismatch { length: row_count });
        }

        let column_for_table = T::wrap(column);
        self.data.insert(new_column_name.clone(), column_for_table);
        self.checkouts
            .insert(new_column_name.clone(), ColumnState::Available);
        Ok(format!("Column '{}' added.", new_column_name))
    }
}

pub trait ColumnDataVariant: Sized {
    fn wrap(column: Column<Self>) -> ColumnData;
    fn unwrap(data: ColumnData) -> Result<Column<Self>, ColumnData>;
}

impl ColumnDataVariant for i64 {
    fn wrap(column: Column<i64>) -> ColumnData {
        ColumnData::Int(column)
    }
    fn unwrap(data: ColumnData) -> Result<Column<i64>, ColumnData> {
        if let ColumnData::Int(c) = data {
            Ok(c)
        } else {
            Err(data)
        }
    }
}

impl ColumnDataVariant for f64 {
    fn wrap(column: Column<f64>) -> ColumnData {
        ColumnData::Float(column)
    }
    fn unwrap(data: ColumnData) -> Result<Column<f64>, ColumnData> {
        if let ColumnData::Float(c) = data {
            Ok(c)
        } else {
            Err(data)
        }
    }
}

impl ColumnDataVariant for String {
    fn wrap(column: Column<String>) -> ColumnData {
        ColumnData::Text(column)
    }
    fn unwrap(data: ColumnData) -> Result<Column<String>, ColumnData> {
        if let ColumnData::Text(c) = data {
            Ok(c)
        } else {
            Err(data)
        }
    }
}

impl ColumnDataVariant for bool {
    fn wrap(column: Column<bool>) -> ColumnData {
        ColumnData::Bool(column)
    }
    fn unwrap(data: ColumnData) -> Result<Column<bool>, ColumnData> {
        if let ColumnData::Bool(c) = data {
            Ok(c)
        } else {
            Err(data)
        }
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
    fn from_csv_reads_headers_and_rows() {
        let file = write_temp_csv("age,name\n25,Ali\n30,Ayşe\n");
        let table = Table::from_csv(file.path()).unwrap();

        match table.data.get("age") {
            Some(ColumnData::Raw(values)) => assert_eq!(
                *values,
                vec![Some("25".to_string()), Some("30".to_string())]
            ),
            _ => panic!("Unexpected result"),
        }

        match table.data.get("name") {
            Some(ColumnData::Raw(values)) => {
                assert_eq!(
                    *values,
                    vec![Some("Ali".to_string()), Some("Ayşe".to_string())]
                )
            }
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn from_csv_treats_empty_call_as_missing() {
        let file = write_temp_csv("age,name\n25,\n,Ayşe\n");

        let table = Table::from_csv(file.path()).unwrap();

        match table.data.get("age") {
            Some(ColumnData::Raw(values)) => {
                assert_eq!(*values, vec![Some("25".to_string()), None])
            }
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn from_csv_fails_when_file_does_not_exist() {
        let result = Table::from_csv(Path::new("this_file_does_not_exist.csv"));

        match result {
            Err(TableError::OpenFileError(_)) => {}
            _ => panic!("Unexpected result"),
        }
    }

    #[test]
    fn file_path_returns_source_path() {
        let file = write_temp_csv("age,name\n25,Ali\n");

        let table = Table::from_csv(file.path()).unwrap();

        assert_eq!(table.file_path(), file.path())
    }

    #[test]
    fn row_count_returns_number_of_rows() {
        let file = write_temp_csv("age,name\n25,Ali\n30,Ayşe\n40,Mehmet\n");

        let table = Table::from_csv(file.path()).unwrap();

        assert_eq!(table.row_count(), 3);
    }

    #[test]
    fn row_count_returns_zero_when_no_data_rows() {
        let file = write_temp_csv("age,name\n");

        let table = Table::from_csv(file.path()).unwrap();

        assert_eq!(table.row_count(), 0);
    }

    #[test]
    fn column_count_returns_number_of_columns() {
        let file = write_temp_csv("age,name,city\n25,Ali,Istanbul\n");

        let table = Table::from_csv(file.path()).unwrap();

        assert_eq!(table.column_count(), 3);
    }

    #[test]
    fn element_count_returns_total_elements() {
        let file = write_temp_csv("age,name\n25,Ali\n30,Ayşe\n40,Mehmet\n");

        let table = Table::from_csv(file.path()).unwrap();

        assert_eq!(table.element_count(), 6);
    }

    #[test]
    fn get_column_materializes_raw_column() {
        let file = write_temp_csv("age,name\n25,Ali\n30,Ayşe\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let column = table.get_column::<i64>("age").unwrap();

        assert_eq!(column.get(0), Some(&Some(25)));
        assert_eq!(column.get(1), Some(&Some(30)));
    }

    #[test]
    fn get_column_fails_when_already_checked_out() {
        let file = write_temp_csv("age,name\n25,Ali\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let _column = table.get_column::<i64>("age").unwrap();
        let result = table.get_column::<i64>("age");

        match result {
            Err(TableError::ColumnNotAvailable { name }) => assert_eq!(name, "age".to_string()),
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn get_column_fails_when_column_not_found() {
        let file = write_temp_csv("age,name\n25,Ali\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let result = table.get_column::<i64>("city");

        match result {
            Err(TableError::ColumnNotFound { name }) => assert_eq!(name, "city".to_string()),
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn get_column_fails_with_type_mismatch() {
        let file = write_temp_csv("age,name\n25,Ali\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let column = Column::new_from_parsed(vec![Some(25)], "age".to_string()).unwrap();
        table
            .data
            .insert("age".to_string(), ColumnData::Int(column));

        let result = table.get_column::<f64>("age");
        match result {
            Err(TableError::TypeMismatch { name }) => assert_eq!(name, "age".to_string()),
            _ => panic!("Unexpected result."),
        }

        assert!(table.data.contains_key("age"))
    }

    #[test]
    fn get_column_restores_raw_data_on_parse_failure() {
        let file = write_temp_csv("age,name\nabc,Ali\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let result = table.get_column::<i64>("age");

        match result {
            Err(TableError::ParseFailed { name, source: _ }) => assert_eq!(name, "age".to_string()),
            _ => panic!("Unexpected result."),
        }

        match table.data.get("age") {
            Some(ColumnData::Raw(values)) => assert_eq!(*values, vec![Some("abc".to_string())]),
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn update_column_succeeds_after_checkout() {
        let file = write_temp_csv("age,name\n25,Ali\n30,Ayşe\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let column = table.get_column::<i64>("age").unwrap();
        let result = table.update_column("age", column);

        assert!(result.is_ok());
    }

    #[test]
    fn update_column_fails_when_not_checked_out() {
        let file = write_temp_csv("age,name\n25,Ali\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let column = Column::new_from_parsed(vec![Some(25)], "age".to_string()).unwrap();
        let result = table.update_column("age", column);

        match result {
            Err(TableError::ColumnNotCheckedOut { name }) => assert_eq!(name, "age".to_string()),
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn update_column_fails_when_column_not_found() {
        let file = write_temp_csv("age,name\n25,Ali\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let column = Column::new_from_parsed(vec![Some(1)], "city".to_string()).unwrap();
        let result = table.update_column("city", column);

        match result {
            Err(TableError::ColumnNotFound { name }) => assert_eq!(name, "city".to_string()),
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn update_column_fails_when_length_mismatches() {
        let file = write_temp_csv("age,name\n25,Ali\n30,Ayşe\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let _checked_out = table.get_column::<i64>("age").unwrap();
        let wrong_length_column =
            Column::new_from_parsed(vec![Some(1)], "age".to_string()).unwrap();

        let result = table.update_column("age", wrong_length_column);

        match result {
            Err(TableError::ColumnLengthMismatch { length }) => assert_eq!(length, 2),
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn update_column_allows_get_column_again() {
        let file = write_temp_csv("age,name\n25,Ali\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let column = table.get_column::<i64>("age").unwrap();
        table.update_column("age", column).unwrap();

        let result = table.get_column::<i64>("age");

        assert!(result.is_ok());
    }

    #[test]
    fn add_column_succeeds_with_new_name() {
        let file = write_temp_csv("age,name\n25,Ali\n30,Ayşe\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let new_column =
            Column::new_from_parsed(vec![Some(true), Some(false)], "is_active".to_string())
                .unwrap();

        let result = table.add_column(new_column);

        assert!(result.is_ok());
        assert_eq!(table.column_count(), 3);

        let fetched = table.get_column::<bool>("is_active");
        assert!(fetched.is_ok());
    }

    #[test]
    fn add_column_fails_when_name_already_exists() {
        let file = write_temp_csv("age,name\n25,Ali\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let duplicate = Column::new_from_parsed(vec![Some(1)], "age".to_string()).unwrap();
        let result = table.add_column(duplicate);

        match result {
            Err(TableError::ColumnAlreadyExists { name }) => assert_eq!(name, "age".to_string()),
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn add_column_fails_when_length_mismatches() {
        let file = write_temp_csv("age,name\n25,Ali\n30,Ayşe\n");
        let mut table = Table::from_csv(file.path()).unwrap();

        let wrong_length = Column::new_from_parsed(vec![Some(1)], "score".to_string()).unwrap();
        let result = table.add_column(wrong_length);

        match result {
            Err(TableError::ColumnLengthMismatch { length }) => {
                assert_eq!(length, table.row_count())
            }
            _ => panic!("Unexpected result."),
        }
    }
}
