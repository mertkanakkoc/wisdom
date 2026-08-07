use std::any::Any;
use std::collections::HashMap;
use std::collections::HashSet;
use std::slice::SliceIndex;

pub const MAX_NAME_LEN: usize = 200;

#[derive(Debug)]
pub enum ColumnError {
    EmptyName,
    NameTooLong { max: usize, actual: usize },
    ParseFailed(Box<dyn std::error::Error>),
    BehaviorAlreadyRegistered { name: String },
    BehaviorNotFound { name: String },
    IndexOutOfBounds { max: usize },
    DuplicatedIndices,
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

    pub fn update_name(&mut self, name: String) -> Result<String, ColumnError> {
        validate_name(&name)?;
        let old_name = self.name.clone();
        self.name = name;
        Ok(format!(
            "Name of the column changed from '{}' to '{}'.",
            old_name, self.name,
        ))
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

    pub fn update_element(
        &mut self,
        new_elements: Vec<(usize, Option<T>)>,
    ) -> Result<String, ColumnError> {
        if !check_len_limit(&new_elements, self.data.len()) {
            return Err(ColumnError::IndexOutOfBounds {
                max: self.data.len(),
            });
        }
        if !check_duplicate_indices(&new_elements) {
            return Err(ColumnError::DuplicatedIndices);
        }

        for (index, element) in new_elements {
            self.data[index] = element;
        }

        Ok(format!("Elements are changed."))
    }

    pub fn remove_element(&mut self, mut indices: Vec<usize>) -> Result<String, ColumnError> {
        if !check_duplicate_indices(&indices) {
            return Err(ColumnError::DuplicatedIndices);
        }

        let max_data_len = self.data.len();
        if !check_len_limit(&indices, max_data_len) {
            return Err(ColumnError::IndexOutOfBounds { max: max_data_len });
        }

        indices.sort_unstable();

        let mut remove_idx = 0;
        let mut i = 0;
        self.data.retain(|_| {
            let keep = if remove_idx < indices.len() && indices[remove_idx] == i {
                remove_idx += 1;
                false
            } else {
                true
            };
            i += 1;
            keep
        });

        Ok(format!("Elements removed from the column '{}'", self.name))
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

trait Duplicates {
    fn check_duplicate(&self) -> bool;
}

impl Duplicates for Vec<usize> {
    fn check_duplicate(&self) -> bool {
        let mut seen: HashSet<usize> = HashSet::with_capacity(self.len());
        self.iter().all(|index| seen.insert(*index))
    }
}

impl<T> Duplicates for Vec<(usize, Option<T>)> {
    fn check_duplicate(&self) -> bool {
        let mut seen: HashSet<usize> = HashSet::with_capacity(self.len());
        self.iter().all(|(index, _)| seen.insert(*index))
    }
}

fn check_duplicate_indices<T: Duplicates>(index_element_pairs: &T) -> bool {
    index_element_pairs.check_duplicate()
}

trait Limits {
    fn check_limits(&self, column_len: usize) -> bool;
}

impl Limits for Vec<usize> {
    fn check_limits(&self, column_len: usize) -> bool {
        self.iter().all(|index| *index < column_len)
    }
}

impl<T> Limits for Vec<(usize, Option<T>)> {
    fn check_limits(&self, column_len: usize) -> bool {
        self.iter().all(|(index, _)| *index < column_len)
    }
}

fn check_len_limit<T: Limits>(index_element_pairs: &T, column_len: usize) -> bool {
    index_element_pairs.check_limits(column_len)
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

    #[test]
    fn update_name_succeeds_with_valid_name() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        let result = column.update_name("years".to_string());

        match result {
            Ok(message) => assert_eq!(message, "Name of the column changed from 'age' to 'years'."),
            Err(_) => panic!("Test paniced!"),
        }

        assert_eq!(column.name(), "years".to_string());
    }

    #[test]
    fn update_name_fails_with_empty_name() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        let result = column.update_name("".to_string());
        match result {
            Err(ColumnError::EmptyName) => {}
            _ => panic!("Unexpected result"),
        }

        assert_eq!(column.name(), "age");
    }

    #[test]
    fn update_name_fails_when_too_long() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        let too_long_name = "a".repeat(MAX_NAME_LEN + 1);
        let result = column.update_name(too_long_name);
        match result {
            Err(ColumnError::NameTooLong { max, actual }) => {
                assert_eq!(max, MAX_NAME_LEN);
                assert_eq!(actual, MAX_NAME_LEN + 1);
            }
            _ => panic!("Unexpected result"),
        }
    }

    #[test]
    fn update_element_succeeds_with_valid_indices() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2), Some(3)], "age".to_string()).unwrap();

        let result = column.update_element(vec![(0, Some(10)), (2, Some(30))]);

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&Some(10)));
        assert_eq!(column.get(1), Some(&Some(2)));
        assert_eq!(column.get(2), Some(&Some(30)));
    }

    #[test]
    fn update_element_can_set_value_to_missing() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2), Some(3)], "age".to_string()).unwrap();

        let result = column.update_element(vec![(1, None)]);

        assert!(result.is_ok());
        assert_eq!(column.get(1), Some(&None));
    }

    #[test]
    fn update_element_fails_and_stays_atomic_for_out_of_bounds() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2), Some(3)], "age".to_string()).unwrap();

        let result = column.update_element(vec![(0, Some(99)), (10, Some(100))]);

        match result {
            Err(ColumnError::IndexOutOfBounds { max }) => assert_eq!(max, column.len()),
            _ => panic!("Unexpected result."),
        }

        assert_eq!(column.get(0), Some(&Some(1)));
    }

    #[test]
    fn update_element_fails_for_duplicate_indices() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2), Some(3)], "age".to_string()).unwrap();

        let result = column.update_element(vec![(0, Some(10)), (0, Some(20))]);

        match result {
            Err(ColumnError::DuplicatedIndices) => {}
            _ => panic!("Unexpected result."),
        }

        assert_eq!(column.get(0), Some(&Some(1)));
    }

    #[test]
    fn remove_element_succeeds_with_valid_indices() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2), Some(3), Some(4)], "age".to_string())
                .unwrap();

        let result = column.remove_element(vec![1, 3]);

        assert!(result.is_ok());
        assert_eq!(column.len(), 2);
        assert_eq!(column.get(0), Some(&Some(1)));
        assert_eq!(column.get(1), Some(&Some(3)));
    }

    #[test]
    fn remove_element_fails_and_stays_atomic_for_out_of_bounds() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2), Some(3)], "age".to_string()).unwrap();

        let result = column.remove_element(vec![0, 10]);

        match result {
            Err(ColumnError::IndexOutOfBounds { max }) => assert_eq!(max, column.len()),
            _ => panic!("Unexpected result."),
        }

        assert_eq!(column.len(), 3);
        assert_eq!(column.get(0), Some(&Some(1)));
    }

    #[test]
    fn remove_element_fails_for_duplicate_indices() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2), Some(3)], "age".to_string()).unwrap();

        let result = column.remove_element(vec![1, 1]);

        match result {
            Err(ColumnError::DuplicatedIndices) => {}
            _ => panic!("Unexpected result."),
        }

        assert_eq!(column.len(), 3);
        assert_eq!(column.get(0), Some(&Some(1)));
    }
}
