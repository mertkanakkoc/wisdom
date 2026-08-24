use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, linear};

pub fn train_linear_regression(
    x: &Tensor,
    y: &Tensor,
    epochs: usize,
    learning_rate: f64,
    device: &Device,
) -> candle_core::Result<Linear> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let (_, in_features) = x.dims2()?;
    let model = linear(in_features, 1, vb.pp("linear"))?;
    let params = ParamsAdamW {
        lr: learning_rate,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    println!("shape: {:?}", model.forward(&x)?.shape());
    for step in 0..epochs {
        let ys = model.forward(&x)?;
        let loss = ys.sub(&y)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
        println!("{step} {}", loss.to_vec0::<f32>()?);
    }
    Ok(model)
}
