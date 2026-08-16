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
}
