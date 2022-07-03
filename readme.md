Deep Danbooru 可执行文件版本



将 Deep Danbooru 的模型转为 onnx 格式：

将 keras 模型转成 tensorflow 模型

```python
import tensorflow as tf

if __name__=='__main__':
    model = tf.keras.models.load_model(r"/path/to/model-resnet_custom_v3.h5")
    tf.saved_model.save(model, r"/path/to/deepdanbooru-saved")
```

使用 tf2onnx 转换

```bash
python -m tf2onnx.convert --saved-model /path/to/deepdanbooru-saved --output /path/to/deepdanbooru.onnx
```



静态量化：

```python
import os
from pathlib import Path

import numpy as np
import onnxruntime
import deepdanbooru as dd
from onnxruntime.quantization import QuantType, CalibrationDataReader, quantize_static


class DanbooruDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, augmented_model_path='augmented_model.onnx'):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            (_, height, width, _) = session.get_inputs()[0].shape
            # name height width color
            nhwc_data_list = preprocess_func(self.image_folder, height, width, size_limit=0)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


def preprocess_func(images_folder, height, width, size_limit=0):
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        image = dd.data.load_image_for_evaluate(image_filepath, width=width, height=height)
        nhwc_data = np.expand_dims(image, axis=0)
        unconcatenated_batch_data.append(nhwc_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data

if __name__=='__main__':
    model_fp32 = Path(r'/path/to/deepdanbooru.onnx')
    model_quant = Path(r'/path/to/output.onnx')
    # quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
    dr = DanbooruDataReader(r'/path/to/image-data')
    quantize_static(model_fp32,
                    model_quant,
                    dr,
                    weight_type=QuantType.QInt8)
```
