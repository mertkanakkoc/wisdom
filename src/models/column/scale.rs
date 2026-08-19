use crate::models::column::ColumnError;

use super::Column;

impl Column<f64> {
    /// Rescales every present value into the `[0, 1]` range: `(x - min) / (max - min)`, using
    /// the column's own present values as `min`/`max`. Missing elements are left as `None`.
    ///
    /// # Errors
    /// Returns [`ColumnError::AllMissingElements`] if every element is missing, or
    /// [`ColumnError::FilledElementsEqual`] if all present values are equal (`max - min` would
    /// be `0`).
    pub fn min_max_scale(&mut self) -> Result<String, ColumnError> {
        let row_count = self.data.len();
        let missing_count = self.missing_count();

        if row_count == missing_count {
            return Err(ColumnError::AllMissingElements);
        }

        let (min, max) = self
            .data
            .iter()
            .flatten()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
                (min.min(x), max.max(x))
            });

        if min == max {
            return Err(ColumnError::FilledElementsEqual);
        }

        let mut update_elements: Vec<(usize, Option<f64>)> = vec![];
        for (i, element) in self.data.iter().enumerate() {
            match element {
                Some(val) => {
                    let new_val = (*val - min) / (max - min);
                    update_elements.push((i, Some(new_val)));
                }
                None => {}
            }
        }

        let result = self.update_element(update_elements);
        match result {
            Err(e) => return Err(e),
            _ => Ok("Min-max scaling completed.".to_string()),
        }
    }

    /// Standardizes every present value to a z-score: `(x - mean) / population_std_dev`, using
    /// the column's own present values to compute the mean and (population) standard
    /// deviation. Missing elements are left as `None`.
    ///
    /// # Errors
    /// Returns [`ColumnError::AllMissingElements`] if every element is missing, or
    /// [`ColumnError::FilledElementsEqual`] if all present values are equal (standard deviation
    /// would be `0`).
    pub fn z_score_standardization(&mut self) -> Result<String, ColumnError> {
        let row_count = self.data.len();
        let missing_count = self.missing_count();
        if row_count == missing_count {
            return Err(ColumnError::AllMissingElements);
        }

        let sum: f64 = self.data.iter().flatten().sum();
        let filled_count: f64 = (row_count - missing_count) as f64;
        let mean: f64 = sum / filled_count;

        let squares_sum: f64 = self
            .data
            .iter()
            .flatten()
            .map(|x| (x - mean) * (x - mean))
            .sum();

        if squares_sum == 0.0 {
            return Err(ColumnError::FilledElementsEqual);
        }

        let division: f64 = squares_sum / filled_count;
        let standard_deviation = division.sqrt();

        let mut update_elements: Vec<(usize, Option<f64>)> = vec![];
        for (i, element) in self.data.iter().enumerate() {
            match element {
                Some(val) => {
                    let new_val = (*val - mean) / standard_deviation;
                    update_elements.push((i, Some(new_val)));
                }
                None => {}
            }
        }

        let result = self.update_element(update_elements);
        match result {
            Err(e) => return Err(e),
            _ => Ok("Z-score standardization completed.".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_max_scale_replaces_values_with_scaled_versions() {
        let mut column = Column::new_from_parsed(
            vec![Some(10.0), None, Some(20.0), Some(30.0)],
            "score".to_string(),
        )
        .unwrap();

        let result = column.min_max_scale();

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&Some(0.0)));
        assert_eq!(column.get(1), Some(&None));
        assert_eq!(column.get(2), Some(&Some(0.5)));
        assert_eq!(column.get(3), Some(&Some(1.0)));
    }

    #[test]
    fn min_max_scale_fails_when_all_missing() {
        let mut column: Column<f64> =
            Column::new_from_parsed(vec![None, None], "score".to_string()).unwrap();

        let result = column.min_max_scale();

        match result {
            Err(ColumnError::AllMissingElements) => {}
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn min_max_scale_fails_when_values_are_equal() {
        let mut column =
            Column::new_from_parsed(vec![Some(5.0), Some(5.0), None], "score".to_string()).unwrap();

        let result = column.min_max_scale();

        match result {
            Err(ColumnError::FilledElementsEqual) => {}
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn z_score_standardization_replaces_values_with_standardized_versions() {
        let mut column = Column::new_from_parsed(
            vec![
                Some(2.0),
                None,
                Some(4.0),
                Some(4.0),
                Some(4.0),
                Some(5.0),
                Some(5.0),
                Some(7.0),
                Some(9.0),
            ],
            "score".to_string(),
        )
        .unwrap();

        let result = column.z_score_standardization();

        assert!(result.is_ok());
        assert_eq!(column.get(0), Some(&Some(-1.5)));
        assert_eq!(column.get(1), Some(&None));
        assert_eq!(column.get(2), Some(&Some(-0.5)));
        assert_eq!(column.get(3), Some(&Some(-0.5)));
        assert_eq!(column.get(4), Some(&Some(-0.5)));
        assert_eq!(column.get(5), Some(&Some(0.0)));
        assert_eq!(column.get(6), Some(&Some(0.0)));
        assert_eq!(column.get(7), Some(&Some(1.0)));
        assert_eq!(column.get(8), Some(&Some(2.0)));
    }

    #[test]
    fn z_score_standardization_fails_when_all_missing() {
        let mut column: Column<f64> =
            Column::new_from_parsed(vec![None, None], "score".to_string()).unwrap();

        let result = column.z_score_standardization();

        match result {
            Err(ColumnError::AllMissingElements) => {}
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn z_score_standardization_fails_when_values_are_equal() {
        let mut column =
            Column::new_from_parsed(vec![Some(5.0), Some(5.0), None], "score".to_string()).unwrap();

        let result = column.z_score_standardization();

        match result {
            Err(ColumnError::FilledElementsEqual) => {}
            _ => panic!("Unexpected result."),
        }
    }
}
