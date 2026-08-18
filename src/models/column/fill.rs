use crate::models::column::ColumnError;

use super::Column;

impl<T> Column<T> {
    pub fn missing_indices(&self) -> Vec<usize> {
        self.data
            .iter()
            .enumerate()
            .filter_map(|(i, val)| val.is_none().then_some(i))
            .collect()
    }

    pub fn fill_with(&mut self, value: T) -> Result<String, ColumnError>
    where
        T: Clone,
    {
        let missing_indices = self.missing_indices();
        let new_elements = missing_indices
            .into_iter()
            .map(|x| (x, Some(value.clone())))
            .collect();
        let update_result = self.update_element(new_elements);
        match update_result {
            Err(e) => return Err(e),
            _ => Ok("Missing elements were filled.".to_string()),
        }
    }

    pub fn forward_fill(&mut self) -> Result<String, ColumnError>
    where
        T: Clone,
    {
        let mut update_elements: Vec<(usize, Option<T>)> = vec![];
        let mut last_valid: Option<T> = None;
        for (i, element) in self.data.iter().enumerate() {
            match element {
                Some(val) => {
                    last_valid = Some(val.clone());
                }
                None => {
                    if last_valid.is_some() {
                        update_elements.push((i, last_valid.clone()));
                    }
                }
            }
        }
        let result = self.update_element(update_elements);
        match result {
            Err(e) => return Err(e),
            _ => Ok("Forward fill completed.".to_string()),
        }
    }

    pub fn backward_fill(&mut self) -> Result<String, ColumnError>
    where
        T: Clone,
    {
        let mut update_elements: Vec<(usize, Option<T>)> = vec![];
        let mut last_valid: Option<T> = None;
        for (i, element) in self.data.iter().enumerate().rev() {
            match element {
                Some(val) => last_valid = Some(val.clone()),
                None => {
                    if last_valid.is_some() {
                        update_elements.push((i, last_valid.clone()));
                    }
                }
            }
        }
        let result = self.update_element(update_elements);
        match result {
            Err(e) => return Err(e),
            _ => Ok("Backward fill completed.".to_string()),
        }
    }
}

impl Column<f64> {
    pub fn fill_mean(&mut self) -> Result<String, ColumnError> {
        let row_count = self.data.len();
        let missing_count = self.missing_count();
        if row_count == missing_count {
            return Err(ColumnError::AllMissingElements);
        }
        let sum: f64 = self.data.iter().flatten().sum();
        let filled_count: f64 = (row_count - missing_count) as f64;
        let mean: f64 = sum / filled_count;
        let result = self.fill_with(mean);
        match result {
            Err(e) => return Err(e),
            _ => Ok("Missing elements were filled with mean value.".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_indices_returns_indices_of_none_values() {
        let column =
            Column::new_from_parsed(vec![Some(1), None, Some(3), None], "age".to_string()).unwrap();

        assert_eq!(column.missing_indices(), vec![1, 3])
    }

    #[test]
    fn missing_indices_returns_empty_when_none_missing() {
        let column = Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        assert_eq!(column.missing_indices(), vec![])
    }

    #[test]
    fn fill_with_replaces_missing_values() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), None, Some(3), None], "age".to_string()).unwrap();

        let result = column.fill_with(0);

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&Some(1)));
        assert_eq!(column.get(1), Some(&Some(0)));
        assert_eq!(column.get(2), Some(&Some(3)));
        assert_eq!(column.get(3), Some(&Some(0)));
    }

    #[test]
    fn fill_with_is_noop_when_nothing_missing() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        let result = column.fill_with(99);

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&Some(1)));
        assert_eq!(column.get(1), Some(&Some(2)));
    }

    #[test]
    fn fill_mean_replaces_missing_with_average() {
        let mut column =
            Column::new_from_parsed(vec![Some(2.0), None, Some(4.0), None], "score".to_string())
                .unwrap();

        let result = column.fill_mean();

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&Some(2.0)));
        assert_eq!(column.get(1), Some(&Some(3.0)));
        assert_eq!(column.get(2), Some(&Some(4.0)));
        assert_eq!(column.get(3), Some(&Some(3.0)));
    }

    #[test]
    fn fill_mean_fails_when_all_missing() {
        let mut column = Column::new_from_parsed(vec![None, None], "score".to_string()).unwrap();

        let result = column.fill_mean();

        match result {
            Err(ColumnError::AllMissingElements) => {}
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn forward_fill_replaces_missing_with_last_valid_value() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), None, Some(3), None, None], "age".to_string())
                .unwrap();

        let result = column.forward_fill();

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&Some(1)));
        assert_eq!(column.get(1), Some(&Some(1)));
        assert_eq!(column.get(2), Some(&Some(3)));
        assert_eq!(column.get(3), Some(&Some(3)));
        assert_eq!(column.get(4), Some(&Some(3)));
    }

    #[test]
    fn forward_fill_leaves_leading_missing_values_as_none() {
        let mut column =
            Column::new_from_parsed(vec![None, None, Some(5)], "age".to_string()).unwrap();

        let result = column.forward_fill();

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&None));
        assert_eq!(column.get(1), Some(&None));
        assert_eq!(column.get(2), Some(&Some(5)));
    }

    #[test]
    fn forward_fill_is_noop_when_nothing_missing() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        let result = column.forward_fill();

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&Some(1)));
        assert_eq!(column.get(1), Some(&Some(2)));
    }

    #[test]
    fn backward_fill_replaces_missing_with_next_valid_value() {
        let mut column =
            Column::new_from_parsed(vec![None, Some(1), None, None, Some(4)], "age".to_string())
                .unwrap();

        let result = column.backward_fill();

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&Some(1)));
        assert_eq!(column.get(1), Some(&Some(1)));
        assert_eq!(column.get(2), Some(&Some(4)));
        assert_eq!(column.get(3), Some(&Some(4)));
        assert_eq!(column.get(4), Some(&Some(4)));
    }

    #[test]
    fn backward_fill_leaves_trailing_missing_values_as_none() {
        let mut column =
            Column::new_from_parsed(vec![Some(5), None, None], "age".to_string()).unwrap();

        let result = column.backward_fill();

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&Some(5)));
        assert_eq!(column.get(1), Some(&None));
        assert_eq!(column.get(2), Some(&None));
    }

    #[test]
    fn backward_fill_is_noop_when_nothing_missing() {
        let mut column =
            Column::new_from_parsed(vec![Some(1), Some(2)], "age".to_string()).unwrap();

        let result = column.backward_fill();

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&Some(1)));
        assert_eq!(column.get(1), Some(&Some(2)));
    }
}
