use std::{collections::BTreeMap, path::Path};

use image::RgbImage;
use onnxruntime::{
    environment::Environment, ndarray::Array, session::Session, tensor::OrtOwnedTensor,
    GraphOptimizationLevel, OrtError,
};

use super::image_util;

pub enum Mode {
    List,
    Prompts,
}

pub struct DeepDanbooru<'a> {
    session: Session<'a>,
}

impl DeepDanbooru<'_> {
    pub fn new<'a, P: AsRef<Path>>(
        environment: &'a Environment,
        model_path: &'a P,
    ) -> Result<DeepDanbooru<'a>, OrtError> {
        let session = environment
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Basic)?
            .with_number_threads(1)?
            .with_model_from_file(model_path)?;
        Ok(DeepDanbooru { session })
    }

    pub fn predict(&mut self, image: &RgbImage, tag_list: &Vec<String>, mode: &Mode) {
        let height = 512;
        let width = 512;
        let input_tensor = Array::from_shape_fn((1, height, width as usize, 3), |(_, y, x, ch)| {
            image_util::sampler(&image, width as u32, height as u32, x, y, ch)
        });

        let outputs: Vec<OrtOwnedTensor<f32, _>> = self.session.run(vec![input_tensor]).unwrap();
        let shape = outputs[0].shape();
        let mut tags = BTreeMap::<i64, String>::new();
        for i in 0..shape[1] {
            let confidence = outputs[0][[0, i]];
            if confidence > 0.5 {
                tags.insert((confidence * 10000f32) as i64, tag_list[i].clone());
            }
        }
        match mode {
            Mode::List => tags.iter().rev().for_each(|tag| {
                println!("{:>6.2}% {}", *tag.0 as f64 * 0.01, tag.1);
            }),
            Mode::Prompts => {
                tags.iter().rev().for_each(|tag| {
                    if !tag.1.starts_with("rating:") {
                        print!(
                            "{}, ",
                            tag.1
                                .split_whitespace()
                                .next()
                                .unwrap_or("")
                                .to_lowercase()
                                .replace("_", " ")
                        )
                    }
                });
                println!()
            }
        }
    }
}
