use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::models::column::Column;

#[derive(Debug)]
pub enum TableError {
    OpenFileError(csv::Error),
    HeadersReadingError(csv::Error),
    RowReadingError(csv::Error),
}

pub enum ColumnData {
    Int(Column<i64>),
    Float(Column<f64>),
    Text(Column<String>),
    Bool(Column<bool>),
    Raw(Vec<Option<String>>),
}

pub struct Table {
    file_path: PathBuf,
    data: HashMap<String, ColumnData>,
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
        })
    }

    pub fn file_path(&self) -> &Path {
        &self.file_path
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
}
