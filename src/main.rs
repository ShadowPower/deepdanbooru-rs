use std::{path::PathBuf, env};

use deepdanbooru_rs::model::{image_util, dd::DeepDanbooru};
use onnxruntime::{environment::Environment, LoggingLevel};

fn print_image_tag(model: &mut DeepDanbooru, image_path: &str) {
    if let Ok(image) = image_util::load(&image_path) {
        model.predict(&image);
    } else {
        println!("不支持的图片: {}", image_path);
    }
}

fn main() {
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Error)
        .build()
        .unwrap();
    let mut model_path = PathBuf::from(std::env::current_exe().unwrap());
    model_path.pop();
    model_path.push("dd_mobile.onnx");

    let mut model = DeepDanbooru::new(&environment, &model_path).unwrap();

    let mut args = env::args();
    if env::args().len() > 1 {
        let image_path = args.nth(1).unwrap();
        print_image_tag(&mut model, image_path.as_str());
    } else {
        loop {
            let mut buffer = String::new();
            println!("请输入图片路径: ");
            std::io::stdin().read_line(&mut buffer).unwrap();
            print_image_tag(&mut model, buffer.trim().trim_matches('"'));
        }
    }
}
