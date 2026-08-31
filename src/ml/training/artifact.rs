use candle_nn::VarMap;

pub enum ModelArchitecture {
    Linear {
        in_features: usize,
        out_features: usize,
    },
}

pub struct TrainingOutput<M> {
    model: M,
    varmap: VarMap,
    architecture: ModelArchitecture,
}
