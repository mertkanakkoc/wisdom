use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, linear};

/// Trains a single-layer linear regression model (`y ≈ x·W + b`) on `x`/`y` — typically the
/// tensors produced by [`crate::models::table::Table::to_tensor`] — using full-batch gradient
/// descent with `AdamW`.
///
/// `x` must be 2D (`(row_count, feature_count)`) and `y` 1D (`(row_count,)`); `y` is reshaped
/// internally to `(row_count, 1)` to match the layer's output before computing the sum-of-
/// squared-errors loss. The weights are randomly initialized (no fixed seed), so results vary
/// between calls — this is a deliberately minimal proof that the `candle`/`candle_nn` training
/// loop (`VarMap`/`VarBuilder`/`linear`/`AdamW`/`backward_step`) works end to end on data coming
/// out of this database, not a tuned or reusable model-training API. Prints each epoch's loss to
/// stdout.
///
/// Returns the trained `candle_nn::Linear` layer, ready to call `.forward()` on for predictions.
pub fn train_linear_regression(
    x: &Tensor,
    y: &Tensor,
    epochs: usize,
    learning_rate: f64,
    device: &Device,
) -> candle_core::Result<Linear> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let (row_count, in_features) = x.dims2()?;
    let model = linear(in_features, 1, vb.pp("linear"))?;
    let params = ParamsAdamW {
        lr: learning_rate,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    let y = y.reshape((row_count, 1))?;
    for step in 0..epochs {
        let ys = model.forward(&x)?;
        let loss = ys.sub(&y)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
        println!("{step} {}", loss.to_vec0::<f32>()?);
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_linear_regression_reduces_loss() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1f32], [2.], [3.], [4.]], &device).unwrap();
        let y = Tensor::new(&[3f32, 5., 7., 9.], &device).unwrap();

        let model = train_linear_regression(&x, &y, 100, 0.05, &device).unwrap();

        let prdedictions = model.forward(&x).unwrap();
        let y_reshaped = y.reshape((4, 1)).unwrap();
        let final_loss = prdedictions
            .sub(&y_reshaped)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();

        assert!(final_loss < 5.0);
    }
}
