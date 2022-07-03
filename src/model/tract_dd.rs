use std::path::Path;

use image::RgbImage;
use tract_onnx::prelude::*;

use crate::model::tags;

use super::image_util;


type Model = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct DeepDanbooru {
    model: Model,
    width: i32,
    height: i32,
}

impl DeepDanbooru {
    pub fn new<P: AsRef<Path>>(model_path: P, width: i32, height: i32) -> DeepDanbooru {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .unwrap()
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, height, width, 3)),
            )
            .unwrap();
        let model = model.into_optimized()
            .unwrap();
        let model = model.into_runnable()
            .unwrap();
        DeepDanbooru {
            model,
            width,
            height,
        }
    }

    pub fn predict(&self, image: &RgbImage) {
        let input_tensor: Tensor = tract_ndarray::Array4::from_shape_fn(
            (1, self.height as usize, self.width as usize, 3),
            |(_, y, x, ch)| {
                image_util::sampler(&image, self.width as u32, self.height as u32, x, y, ch)
            },
        )
        .into();

        let result = self.model.run(tvec!(input_tensor)).unwrap();
        let array_view = result[0].to_array_view::<f32>().unwrap();
        let shape = array_view.shape(); // -> 1, 9176
        for i in 0..shape[1] {
            let confidence = array_view[[0, i]];
            if confidence > 0.5 {
                println!("{:>6.2}% {}", confidence * 100., tags::name(i));
            }
        }
    }
}
