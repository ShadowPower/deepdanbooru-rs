use std::{env, path::PathBuf};

use deepdanbooru_rs::model::{
    dd::{DeepDanbooru, Mode},
    image_util, tags,
};
use onnxruntime::{environment::Environment, LoggingLevel};

fn print_image_tag(
    model: &mut DeepDanbooru,
    image_path: &str,
    tag_list: &Vec<String>,
    mode: &Mode,
) {
    if let Ok(image) = image_util::load(&image_path) {
        model.predict(&image, tag_list, mode);
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
    let mut mode = Mode::List;
    if env::args().len() > 1 {
        let mut image_path = args.nth(1).unwrap();
        // 第二项参数可以指定输出模式：prompts, list
        if vec!["prompts", "list"].contains(&image_path.as_str()) {
            mode = match image_path.to_ascii_lowercase().as_str() {
                "prompts" => Mode::Prompts,
                "list" => Mode::List,
                _ => mode,
            };
            image_path = args
                .nth(0)
                .expect("usage: deepdanbooru [prompts|list] <file path>");
        }
        print_image_tag(&mut model, image_path.as_str(), &tag_list, &mode);
    } else {
        loop {
            let mut buffer = String::new();
            println!("请输入图片路径: ");
            std::io::stdin().read_line(&mut buffer).unwrap();
            match buffer.trim().to_ascii_lowercase().as_str() {
                "prompts" => mode = Mode::Prompts,
                "list" => mode = Mode::List,
                _ => {
                    print_image_tag(
                        &mut model,
                        buffer.trim().trim_matches('"'),
                        &tag_list,
                        &mode,
                    );
                }
            }
        }
    };
}
