use std::path::Path;

use image::{RgbImage, ImageError, math::Rect};

pub fn load<P: AsRef<Path>>(image_path: P) -> Result<RgbImage, ImageError> {
    let image = image::open(image_path)?;
    Ok(image.to_rgb8())
}

pub fn sampler(rgb_image: &RgbImage, width: u32, height: u32, x: usize, y: usize, channel: usize) -> f32 {
    // 计算采样窗口和图像外接矩形的缩放比例
    // 比例大于 1 时，图像比窗口大
    let radiow = rgb_image.width() as f64 / width as f64;
    let radioh = rgb_image.height() as f64 / height as f64;
    let radio = radiow.max(radioh);
    // 使图像相对于窗口居中，因此需要算出图像的左侧和顶部相对于窗口的 x y 坐标
    let top = (height as i64 - (rgb_image.height() as f64 / radio) as i64) / 2;
    let left = (width as i64 - (rgb_image.width() as f64 / radio) as i64) / 2;

    let mut scale_x = (x as f64 * radio) as i64 - left;
    let mut scale_y = (y as f64 * radio) as i64 - top;

    // let px_w = (rgb_image.width() as i64 - scale_x).min(1.max(radio as i64)) as u32;
    // let px_h = (rgb_image.height() as i64 - scale_y).min(1.max(radio as i64)) as u32;
    // if scale_x < 0 || scale_y < 0 {
    //     return 0f32;
    // } else if scale_x as u32 >= rgb_image.width() || scale_y as u32 >= rgb_image.height() {
    //     return 0f32;
    // } else {
    //     let area_rect = Rect { x:scale_x as u32, y: scale_y as u32, width: px_w, height: px_h };
    //     return (area_interpolation(rgb_image, area_rect, channel) as f64 / 255.) as f32;
    // }

    scale_x = scale_x.clamp(0, rgb_image.width() as i64 - 1);
    scale_y = scale_y.clamp(0, rgb_image.height() as i64 - 1);
    let px_w = (rgb_image.width() as i64 - scale_x).min(1.max(radio as i64)) as u32;
    let px_h = (rgb_image.height() as i64 - scale_y).min(1.max(radio as i64)) as u32;
    let area_rect = Rect { x:scale_x as u32, y: scale_y as u32, width: px_w, height: px_h };
    (area_interpolation(rgb_image, area_rect, channel) as f64 / 255.) as f32
}

pub fn area_interpolation(rgb_image: &RgbImage, rect: Rect, channel: usize) -> u8 {
    let (x, y, mut w, mut h) = (rect.x, rect.y, rect.width, rect.height);
    w = 1.max(w);
    h = 1.max(h);
    let mut sum = 0u64;
    for i in x..x + w {
        for j in y..y + h {
            sum += rgb_image.get_pixel(i, j)[channel] as u64;
        }
    }
    (sum / (w * h) as u64) as u8
}