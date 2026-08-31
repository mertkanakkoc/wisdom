use candle_core::Device;
use wisdom::ml::training::train_linear_regression;
use wisdom::models::column::Column;
use wisdom::models::table::Table;

fn main() -> candle_core::Result<()> {
    let mut table = Table::default();

    let age = Column::new_from_parsed(
        vec![Some(25.0), Some(30.0), Some(35.0), Some(40.0)],
        "age".to_string(),
    )
    .unwrap();
    let score = Column::new_from_parsed(
        vec![Some(1.5), Some(2.5), Some(3.5), Some(4.5)],
        "score".to_string(),
    )
    .unwrap();

    table.add_column(age).unwrap();
    table.add_column(score).unwrap();

    let device = Device::Cpu;
    let (x, y) = table
        .to_tensor(vec!["age".to_string()], "score".to_string(), &device)
        .unwrap();

    println!("x shape: {:?}", x.shape());
    println!("y shape: {:?}", y.shape());

    let _training_output = train_linear_regression(&x, &y, 50, 0.1, &device)?;
    println!("Training finished.");

    Ok(())
}
