use std::any::Any;
use std::collections::HashMap;

pub const MAX_NAME_LEN: usize = 200;

#[derive(Debug)]
pub enum ColumnError {
    EmptyName,
    NameTooLong { max: usize, actual: usize },
    ParseFailed(Box<dyn std::error::Error>),
}

enum BehaviorResult {
    Float(f64),
    Int(i64),
    Text(String),
    Bool(bool),
    Custom(Box<dyn Any>),
}

pub struct Column<T> {
    data: Vec<Option<T>>,
    name: String,
    behaviors: HashMap<String, Box<dyn Fn(&Vec<Option<T>>) -> BehaviorResult>>,
}

impl<T> Column<T> {
    pub fn new_from_parsed(data: Vec<Option<T>>, name: String) -> Result<Self, ColumnError> {
        validate_name(&name)?;
        Ok(Self {
            data,
            name,
            behaviors: HashMap::new(),
        })
    }
}

impl<T> Column<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + 'static,
{
    pub fn new_from_raw(raw: Vec<Option<String>>, name: String) -> Result<Self, ColumnError> {
        validate_name(&name)?;

        let data = raw
            .into_iter()
            .map(|cell| {
                cell.map(|s| {
                    s.parse::<T>()
                        .map_err(|e| ColumnError::ParseFailed(Box::new(e)))
                })
                .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            data,
            name,
            behaviors: HashMap::new(),
        })
    }
}

fn validate_name(name: &str) -> Result<(), ColumnError> {
    if name.is_empty() {
        return Err(ColumnError::EmptyName);
    }

    let chars_count: usize = name.chars().count();
    if chars_count > MAX_NAME_LEN {
        return Err(ColumnError::NameTooLong {
            max: MAX_NAME_LEN,
            actual: chars_count,
        });
    }
    Ok(())
}
