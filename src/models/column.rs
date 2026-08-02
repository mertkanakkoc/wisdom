use std::any::Any;
use std::collections::HashMap;
use std::slice::SliceIndex;

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

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn missing_count(&self) -> usize {
        self.data.iter().filter(|x| x.is_none()).count()
    }

    pub fn get<I>(&self, index: I) -> Option<&<I as SliceIndex<[Option<T>]>>::Output>
    where
        I: SliceIndex<[Option<T>]>,
    {
        self.data.get(index)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_returns_the_column_name() {
        let column = Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        assert_eq!(column.name(), "age")
    }

    #[test]
    fn len_returns_number_of_elements() {
        let column =
            Column::new_from_parsed(vec![Some(1), Some(2), None], "age".to_string()).unwrap();

        assert_eq!(column.len(), 3)
    }

    #[test]
    fn is_empty_returns_true_for_empty_column() {
        let column: Column<i64> = Column::new_from_parsed(vec![], "age".to_string()).unwrap();

        assert!(column.is_empty())
    }

    #[test]
    fn is_empty_returns_false_for_nonempty_column() {
        let column = Column::new_from_parsed(vec![Some(1)], "age".to_string()).unwrap();

        assert!(!column.is_empty())
    }

    #[test]
    fn missing_count_returns_zero() {
        let column =
            Column::new_from_parsed(vec![Some(1), Some(2), Some(3)], "age".to_string()).unwrap();

        assert_eq!(column.missing_count(), 0)
    }

    #[test]
    fn missing_count_returns_count() {
        let column =
            Column::new_from_parsed(vec![Some(1), None, Some(2)], "age".to_string()).unwrap();

        assert_eq!(column.missing_count(), 1)
    }

    #[test]
    fn missing_count_returns_total_when_all_missing() {
        let column: Column<i64> =
            Column::new_from_parsed(vec![None, None, None], "age".to_string()).unwrap();

        assert_eq!(column.missing_count(), 3)
    }

    #[test]
    fn get_returns_some_some_value() {
        let column =
            Column::new_from_parsed(vec![Some(1), Some(2), Some(3)], "age".to_string()).unwrap();

        assert_eq!(column.get(1), Some(&Some(2)))
    }

    #[test]
    fn get_returns_some_none_value() {
        let column =
            Column::new_from_parsed(vec![Some(1), None, Some(3)], "age".to_string()).unwrap();

        assert_eq!(column.get(1), Some(&None));
    }

    #[test]
    fn get_returns_none_for_out_of_bounds_index() {
        let column =
            Column::new_from_parsed(vec![Some(1), Some(2), Some(3)], "age".to_string()).unwrap();

        assert_eq!(column.get(5), None);
    }

    #[test]
    fn get_with_range_returns_slice() {
        let column =
            Column::new_from_parsed(vec![Some(1), Some(2), Some(3), Some(4)], "age".to_string())
                .unwrap();

        assert_eq!(column.get(1..3), Some(&[Some(2), Some(3)][..]))
    }
}
