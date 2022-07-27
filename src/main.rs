use std::{path::PathBuf, env};

use deepdanbooru_rs::model::{image_util, dd::DeepDanbooru, tags};
use onnxruntime::{environment::Environment, LoggingLevel};

fn print_image_tag(model: &mut DeepDanbooru, image_path: &str, tag_list: &Vec<String>) {
    if let Ok(image) = image_util::load(&image_path) {
        model.predict(&image, tag_list);
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

    let exe_path = PathBuf::from(std::env::current_exe().unwrap());
    let exe_path = exe_path.parent().unwrap();

    let model_path = exe_path.join("deepdanbooru.onnx");
    let tags_path = exe_path.join("tags.txt");

    let mut model = DeepDanbooru::new(&environment, &model_path).unwrap();
    let tag_list = tags::load_tags_to_list(tags_path);

    let mut args = env::args();
    if env::args().len() > 1 {
        let image_path = args.nth(1).unwrap();
        print_image_tag(&mut model, image_path.as_str(), &tag_list);
    } else {
        loop {
            let mut buffer = String::new();
            println!("请输入图片路径: ");
            std::io::stdin().read_line(&mut buffer).unwrap();
            print_image_tag(&mut model, buffer.trim().trim_matches('"'), &tag_list);
        }
    }
}
