use crate::models::column::ColumnError;

use super::Column;

impl Column<f64> {
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
}
