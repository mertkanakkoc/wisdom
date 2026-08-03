use std::any::Any;
use std::collections::HashMap;
use std::slice::SliceIndex;

pub const MAX_NAME_LEN: usize = 200;

#[derive(Debug)]
pub enum ColumnError {
    EmptyName,
    NameTooLong { max: usize, actual: usize },
    ParseFailed(Box<dyn std::error::Error>),
    BehaviorAlreadyRegistered { name: String },
    BehaviorNotFound { name: String },
}

#[derive(Debug)]
pub enum BehaviorResult {
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

    pub fn register_behavior(
        &mut self,
        name: &str,
        f: impl Fn(&Vec<Option<T>>) -> BehaviorResult + 'static,
    ) -> Result<String, ColumnError> {
        if self.behaviors.contains_key(name) {
            return Err(ColumnError::BehaviorAlreadyRegistered {
                name: name.to_string(),
            });
        }

        self.behaviors.insert(name.to_string(), Box::new(f));
        Ok(format!("Behavior '{}' added.", name))
    }

    pub fn update_behavior(
        &mut self,
        name: &str,
        f: impl Fn(&Vec<Option<T>>) -> BehaviorResult + 'static,
    ) -> Result<String, ColumnError> {
        if !self.behaviors.contains_key(name) {
            return Err(ColumnError::BehaviorNotFound {
                name: name.to_string(),
            });
        }

        self.behaviors.insert(name.to_string(), Box::new(f));
        Ok(format!("Behavior '{}' updated.", name))
    }

    pub fn remove_behavior(&mut self, name: &str) -> Result<String, ColumnError> {
        match self.behaviors.remove(name) {
            Some(_) => Ok(format!("Behavior '{}' removed.", name)),
            None => Err(ColumnError::BehaviorNotFound {
                name: name.to_string(),
            }),
        }
    }

    pub fn call_behavior(&self, name: &str) -> Option<BehaviorResult> {
        match self.behaviors.get(name) {
            Some(behavior) => Some(behavior(&self.data)),
            None => None,
        }
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

    #[test]
    fn register_behavior_succeeds_for_new_name() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        let result = column.register_behavior("sum", |data| {
            let sum: i64 = data.iter().flatten().sum();
            BehaviorResult::Int(sum)
        });

        match result {
            Ok(message) => assert_eq!(message, "Behavior 'sum' added."),
            Err(_) => panic!("Test paniced!"),
        }
    }

    #[test]
    fn register_behavior_fails_when_name_already_exits() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        column
            .register_behavior("sum", |data| {
                let sum: i64 = data.iter().flatten().sum();
                BehaviorResult::Int(sum)
            })
            .unwrap();

        let second_behavior = column.register_behavior("sum", |data| {
            let sum: i64 = data.iter().flatten().sum();
            BehaviorResult::Int(sum)
        });

        match second_behavior {
            Err(ColumnError::BehaviorAlreadyRegistered { name }) => {
                assert_eq!(name, "sum");
            }
            _ => panic!("Unexpected result"),
        }
    }

    #[test]
    fn call_behavior_returns_none() {
        let column = Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        let behavior = column.call_behavior("sum");

        assert!(behavior.is_none())
    }

    #[test]
    fn call_behavior_executes_registered_behavior() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        column
            .register_behavior("sum", |data| {
                let total: i64 = data.iter().flatten().sum();
                BehaviorResult::Int(total)
            })
            .unwrap();

        let result = column.call_behavior("sum");

        match result {
            Some(BehaviorResult::Int(value)) => assert_eq!(value, 3),
            _ => panic!("Unexpected result!"),
        }
    }

    #[test]
    fn update_behavior_succeeds_and_replaces_logic() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        column
            .register_behavior("sum", |data| {
                let total: i64 = data.iter().flatten().sum();
                BehaviorResult::Int(total)
            })
            .unwrap();

        let update_result = column.update_behavior("sum", |data| {
            let count = data.iter().flatten().count() as i64;
            BehaviorResult::Int(count)
        });

        match update_result {
            Ok(message) => assert_eq!(message, "Behavior 'sum' updated."),
            Err(_) => panic!("Test paniced"),
        }

        let call_result = column.call_behavior("sum");
        match call_result {
            Some(BehaviorResult::Int(value)) => assert_eq!(value, 2),
            _ => panic!("Unexpected result!"),
        }
    }

    #[test]
    fn update_behavior_fails_when_not_registered() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        let result = column.update_behavior("sum", |data| {
            let total: i64 = data.iter().flatten().sum();
            BehaviorResult::Int(total)
        });

        match result {
            Err(ColumnError::BehaviorNotFound { name }) => assert_eq!(name, "sum".to_string()),
            _ => panic!("Unexpected result"),
        }
    }

    #[test]
    fn remove_behavior_succeeds_and_behavior_no_longer_callable() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        column
            .register_behavior("sum", |data| {
                let total: i64 = data.iter().flatten().sum();
                BehaviorResult::Int(total)
            })
            .unwrap();

        let remove_result = column.remove_behavior("sum");

        match remove_result {
            Ok(message) => assert_eq!(message, "Behavior 'sum' removed."),
            Err(_) => panic!("Test paniced!"),
        }

        let call_result = column.call_behavior("sum");
        assert!(call_result.is_none())
    }

    #[test]
    fn remove_behavior_fails_when_not_registered() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        let result = column.remove_behavior("sum");
        match result {
            Err(ColumnError::BehaviorNotFound { name }) => assert_eq!(name, "sum".to_string()),
            _ => panic!("Unexpected result"),
        }
    }
}
