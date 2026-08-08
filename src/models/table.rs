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
                let cell = record.get(index).map(|s| s.to_string());
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
}
