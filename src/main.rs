use std::{path::PathBuf, env};

use deepdanbooru_rs::model::{image_util, tract_dd::DeepDanbooru};

fn print_image_tag(model: &DeepDanbooru, image_path: &str) {
    if let Ok(image) = image_util::load(&image_path) {
        model.predict(&image);
    } else {
        println!("不支持的图片: {}", image_path);
    }
}

fn main() {
    let mut model_path = PathBuf::from(std::env::current_exe().unwrap());
    model_path.pop();
    model_path.push("dd_mobile.onnx");

    let model = DeepDanbooru::new(&model_path, 512, 512);

    let mut args = env::args();
    if env::args().len() > 1 {
        let image_path = args.nth(1).unwrap();
        print_image_tag(&model, image_path.as_str());
    } else {
        loop {
            let mut buffer = String::new();
            println!("请输入图片路径: ");
            std::io::stdin().read_line(&mut buffer).unwrap();
            print_image_tag(&model, buffer.trim().trim_matches('"'));
        }
    }
}
