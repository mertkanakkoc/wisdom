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
