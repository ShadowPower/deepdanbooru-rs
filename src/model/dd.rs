
use std::path::Path;

use image::RgbImage;
use onnxruntime::{
    environment::Environment, ndarray::Array, tensor::OrtOwnedTensor, GraphOptimizationLevel,
    session::Session, OrtError,
};

use crate::model::tags;

use super::image_util;

pub struct DeepDanbooru<'a> {
    session: Session<'a>
}

impl DeepDanbooru<'_> {
    pub fn new<'a, P: AsRef<Path>>(environment: &'a Environment, model_path: &'a P) -> Result<DeepDanbooru<'a>, OrtError> {
        let session = environment
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Basic)?
            .with_number_threads(1)?
            .with_model_from_file(model_path)?;
        Ok(DeepDanbooru {
            session,
        })
    }

    pub fn predict(&mut self, image: &RgbImage) {
        let height = 512;
        let width = 512;
        let input_tensor = Array::from_shape_fn(
            (1, height, width as usize, 3),
            |(_, y, x, ch)| {
                image_util::sampler(&image, width as u32, height as u32, x, y, ch)
            },
        );

        let outputs: Vec<OrtOwnedTensor<f32, _>> = self.session.run(vec![input_tensor]).unwrap();
        let shape = outputs[0].shape();
        for i in 0..shape[1] {
            let confidence = outputs[0][[0, i]];
            if confidence > 0.5 {
                println!("{:>6.2}% {}", confidence * 100., tags::name(i));
            }
        }
    }
}