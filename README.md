# findFalling
通过新采集到的数据训练moVinet模型，支持摔倒行为监控

## 针对moVinet模型超参配置和运行效果表
| model_id | batchsize | resolution | frame | num_epochs | loss | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| a0   | 16   | 360    | 50   |  3  |   2162.128173828125  | 0.8928571343421936   |   
|  a0  | 16   |  360   | 16   | 3   |   2191.13671875   |  0.7142857313156128  |   
| a0 | 16 | 360 | 16 | 3  | 1982.689453125  | 0.8214285969734192  |  
|  a0  |   8 |    360  | 16    | 5   |    4329.39599609375  |   0.7142857313156128  |   
|  a2  |  16  |   360  |  25  |   3 |   316074.9375  |   0.6071428656578064 |   
| a3   |  16  |   360  | 25   | 3   |   279203616.0  |  0.5  |   
- 从以上表格数据中，我们可以看到第一行的参数的结果是最好的。因此我们可以发现，当其他参数类似时，frame（帧数）越高，模型对一个行为姿态的前后时序分析的能力最强，精度也最高。
  
## 数据来源：
- 我们收集和采集的训练数据：[dataFall](https://drive.google.com/drive/folders/16X_en94lnHoKCM5jbijGp1Z_7xQdcncE?usp=sharing)
- 我们收集和采集的测试数据：[testDataFall](https://drive.google.com/drive/folders/1n9ataiqDJ-SN5FPJ5erQ1KHq7-Okkcxq?usp=sharing)
## 可运行代码文件: [colab文件](https://colab.research.google.com/drive/1bvZstLrQu1zH3W9L_xejo59VjW2SM2XI?usp=sharing)
## 核心代码解析
### 1.数据预处理
  - 视频帧大小设置
```python  
  # 遍历文件夹中的所有视频文件，改写视频大小为指定尺寸
  for filename in os.listdir(input_folder_path):
    if filename.endswith(".mp4"):  # 你可以添加更多的视频格式，比如 ".avi", ".mov" 等
        input_video_path = os.path.join(input_folder_path, filename)
        output_video_path = os.path.join(output_folder_path, filename)

        # 使用opencv打开视频
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Unable to open video: {input_video_path}")
            continue

        # 获取视频的fps
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 创建一个VideoWriter对象，用于输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (360, 240))

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # 改变帧的尺寸
            resized_frame = cv2.resize(frame, (360, 240))

            # 写入输出视频
            out.write(resized_frame)

        cap.release()
        out.release()
```
  - 视频名修改
### 2.模型及相关环境安装
  - opencv-python=4.7.0.72等环境
```python
!pip install remotezip tqdm opencv-python==4.7.0.72 opencv-python-headless==4.7.0.72 tf-models-official
```
  - load data:
  - 超参配置
  - 模型下载：moVinet-a0
```python
model_id = 'a0'
batch_size = 16
num_frames = 50
resolution = 360
num_epochs = 3

tf.keras.backend.clear_session()

backbone = movinet.Movinet(model_id=model_id)
backbone.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([None, None, None, None, 3])

# Load pre-trained weights
!wget https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_base.tar.gz -O movinet_a0_base.tar.gz -q
!tar -xvf movinet_a0_base.tar.gz

# !wget https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a3_base.tar.gz -O movinet_a3_base.tar.gz -q
#!tar -xvf movinet_a3_base.tar.gz

#!wget https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a2_base.tar.gz -O movinet_a2_base.tar.gz -q
#!tar -xvf movinet_a2_base.tar.gz

checkpoint_dir = f'movinet_{model_id}_base'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()
```
### 3.建立分类器和模型训练
```python
# 建立分类器和模型训练
def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  return model

model = build_classifier(batch_size, num_frames, resolution, backbone, 2)

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
```
### 4.构建训练数据集和测试数据集按8:2数据比例
```python
# 数据集处理，输入一个视频数据集，安装8:2得到训练和测试数据集
fall_videos = pathlib.Path('./drive/MyDrive/dataFall')

image_size=(360, 360)
train_ds, test_ds = load_and_preprocess_video_data(fall_videos, num_frames, image_size, batch_size)
```
### 5.定义输入形状参数
```python
# 定义输入形状参数
input_shape = [batch_size, num_frames, resolution, resolution, 3]

# 在模型上调用 build 方法
model.build(input_shape)
```
### 6.运行和验证模型
```python
# 运行和验证模型
results = model.fit(train_ds,
                    validation_data=test_ds,
                    epochs=num_epochs,
                    validation_freq=1,
                    verbose=1)

model.evaluate(test_ds, return_dict=True)
```
### 7.保存当前模型
```python
# 保全当前模型，准备为后续推理使用
model.save('./drive/MyDrive/get16_movinet_a0_base')
```
### 8.安装预测集合
```python
# 安装预测集合
def video_to_input(video_path, num_frames, frame_size):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    cap.release()

    # 将帧堆叠在一起并添加一个批次维度
    input = np.stack(frames, axis=0)[np.newaxis, ...]

    return input

def handle_predictions(predictions):
    predicted_class = np.argmax(predictions[0])
    print(f'The predicted class is {predicted_class}.')

# 加载模型进行预测
loaded_model = tf.keras.models.load_model('./drive/MyDrive/get16_movinet_a0_base')

# 选择预测数据集文件夹
fall_pre_videos = pathlib.Path('./drive/MyDrive/testDataFall/Fall')

# 视频的帧数和帧的尺寸
num_frames = 160
frame_size = (360, 360)

for video_path in fall_pre_videos.glob('*.mp4'):
    # 将视频转换为模型的输入
    print(f'Processing video: {video_path.name}')  # 打印文件名

    input = video_to_input(str(video_path), num_frames, frame_size)

    # 使用模型进行预测
    predictions = loaded_model.predict(input)

    # 处理模型的预测
    # 这取决于你的模型的输出以及你如何解释这些输出
    handle_predictions(predictions)  # 你需要定义这个函数
```
