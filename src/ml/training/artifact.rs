use candle_nn::VarMap;

pub enum ModelArchitecture {
    Linear {
        in_features: usize,
        out_features: usize,
    },
}

pub struct TrainingOutput<M> {
    pub model: M,
    pub varmap: VarMap,
    pub architecture: ModelArchitecture,
}
