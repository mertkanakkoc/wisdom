use std::{collections::HashMap, path::PathBuf};

use crate::models::column::Column;

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
