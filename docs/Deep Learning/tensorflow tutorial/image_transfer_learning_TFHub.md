---
layout: default
title: 텐서플로우 허브 전이학습
parent: 텐서플로우 튜토리얼
grand_parent: Deep Learning
nav_order: 101
---

# 텐서플로 허브와 전이학습(transfer learning)
텐서플로 허브는 미리 학습된 모델들을 공유하는 곳이다. [텐서플로우 모듈 허브](https://tfhub.dev/)를 보면 미리 학습된 모델들을 찾을 수 있다.
이 튜토리얼은 다음과 같은 과정들을 보여줄 것이다.

1. tf.keras로 텐서플로 허브를 사용하는 방법.
2. 텐서플로 허브를 사용하여 이미지 분류를 하는 방법.
3. 간단한 전이학습을 하는 방법.

## 셋업
```
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pylab as plt

import tensorflow as tf
```
```
!pip install -q -U tf-hub-nightly
import tensorflow_hub as hub

from tensorflow.keras import layers
```

## ImageNet 분류기(classifier)
### 분류기 다운로드
*hub.module*을 사용하여 mobilenet을 불러오고, *tf.keras.layers.Lambda*로 감싸서 keras 레이어로 만든다. tfhub.dev에서 구한 [어떠한 텐서플로우2의 비교가능한 이미지 분류기 URL](https://tfhub.dev/s?q=tf2&module-type=image-classification)도 여기서 다 작동을 한다.

```
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
``` 
```
IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])
```

### 하나의 이미지에 대해 실행해보기
이미지를 하나 받아서 모델에 적용해보자
```
import numpy as np
import PIL.Image as Image

grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper
```
```
grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape
```
```
> (224, 224, 3)
```
배치 차원을 더해주고, 모델에다 이미지를 넣는다
```
result = classifier.predict(grace_hopper[np.newaxis, ...])
result.shape
```
```
> (1, 1001)
```
결과는 logit의 1001 요소 벡터 값, 그리고 이미지의 각 클래스에 대한 확률 값이다.
따라서 가장 높은 확률의 id값은 argmax로 구할 수 있다.
```
predicted_class = np.argmax(result[0], axis=-1)
predicted_class
```
```
> 653
```

### 예측 해독하기
우리는 class ID값을 예측하고, **ImageNet** 라벨을 불러오고, 예측을 해독할 것이다.
```
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
```
```
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
```

## 단순한 전이학습(transfer learning)
TF Hub를 사용한다면 우리의 데이터셋의 클래스는 구분하기 위한 모델의 탑 레이어를 재교육하기 쉽다.

### Dataset
이 예제에서 너는 tensorFlow flowers dataset을 이용할 것이다.
```
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)
```
우리 모델에 이 데이터를 불러오는 가장 간단한 방법은 **tf.keras.preprocessing.image.ImageDataGenerator를 사용하는 것이다.

모든 텐서플로우 허브의 이미지 모듈들은 [0, 1]사이의 실수 값을 사용한다. **ImageDataGenerator**의 *rescale* 파라메터를 사용하여 변환해라

```
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)
```

결과 객체는 image_batch, label_batch 쌍을 반환하는 iterator이다
```
for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break
```
```
> Image batch shape:  (32, 224, 224, 3)
  Label batch shape:  (32, 5)
```

### 이미지의 배치에 대한 분류기(classifier) 실행하기
```
result_batch = classifier.predict(image_batch)
result_batch.shape
```
```
> (32, 1001)
```
```
predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
predicted_class_names
```
```
> array(['daisy', 'mushroom', 'Bedlington terrier', 'daisy', 'daisy', 'bee',
         'coral fungus', 'hair slide', 'picket fence', 'daisy', 'pot',
         'mushroom', 'daisy', 'bee', 'rapeseed', 'daisy', 'daisy',
         'water buffalo', 'spider web', 'cardoon', 'daisy', 'daisy', 'bee',
         'daisy', 'vase', 'daisy', 'barn spider', 'slug', 'coral fungus',
         'sea urchin', 'pot', 'coral fungus'], dtype='<U30')
```

이미지와 함께 결과를 확인해보자
```
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")
```

<img src="https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub_files/output_IXTB22SpxDLP_0.png"/><br/>

결과는 완벽하지 않지만 모델이 데이지를 제외한 클래스들에 대해 학습된게 아니라는 점을 고려해야한다.

### 헤드리스모델 다운로드
텐서플로우 허브는 또한 top classification layer를 제외한 모델을 배포한다. 이것은 전이 학습에 매우 쉽게 사용될 수 있다.

tfhub.dev에서 구한 [어떠한 텐서플로우2의 비교가능한 이미지 특징 벡터 URL](https://tfhub.dev/s?module-type=image-feature-vector&q=tf2)도 여기서 다 작동을 한다.
```
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
```
특징 추출기를 만든다
```
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))
```
각 이미지에 대해 1280 길이의 벡터를 반환한다
```
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)
```
```
> (32, 1280)
```
특징 추출 레이어(feature extractor layer)의 변수들을 고정한다. 따라서 모델에서 오직 새로운 classifier layer를 훈련하고 추가할 뿐이다.
```
feature_extractor_layer.trainable = False
```

### Classification haed 붙이기
hub layer를 **tf.keras.Sequential** 모델로 감싸고 새로운 classification layer를 추가해라
```
model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(image_data.num_classes)
])

model.summary()
```
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
keras_layer_1 (KerasLayer)   (None, 1280)              2257984   
_________________________________________________________________
dense (Dense)                (None, 5)                 6405      
=================================================================
Total params: 2,264,389
Trainable params: 6,405
Non-trainable params: 2,257,984
_________________________________________________________________
```
```
predictions = model(image_batch)
predictions.shape
```
```
> TensorShape([32, 5])
```

### 모델 학습시키기
트레이닝 과정을 설정하고 컴파일한다
```
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['acc'])
```

이제 **.fit** 메서드를 사용해서 모델을 학습시킨다

예제를 짧게 유지하기 위해 단지 2 에폭만 학습한다. 트레이닝 과정을 시각화하기위해 에폭의 평균 대신 각 배치 고유의 loss와 accuracy를 남기는 custom callback을 이용한다.
```
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()
```
```
steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit_generator(image_data, epochs=2,
                              steps_per_epoch=steps_per_epoch,
                              callbacks = [batch_stats_callback])
```
```
> Epoch 1/2
  115/115 [==============================] - 10s 91ms/step - loss: 0.3135 - acc: 0.9375
  Epoch 2/2
  115/115 [==============================] - 10s 89ms/step - loss: 0.2287 - acc: 0.9062
```
적은 학습만으로도 잘 동작한다는걸 확인할 수 있습니다.
```
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)
```
<img src="https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub_files/output_3uvX11avTiDg_1.png"/><br/>

### 예측(prediction) 확인하기
그림을 다시 그리기 위해 클래스 이름의 정렬된 순서를 가져온다
```
class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
class_names
```
```
> array(['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips'],
      dtype='<U10')
```

모델에 이미지 배치를 돌리고 클래스 이름이 위치하도록 변환한다
```
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
```
결과를 그린다
```
label_id = np.argmax(label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "green" if predicted_id[n] == label_id[n] else "red"
  plt.title(predicted_label_batch[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
```
<img src="https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub_files/output_wC_AYRJU9NQe_0.png"/><br/>

## Export your model
훈련시킨 모델을 저장하고 내보낸다
```
import time
t = time.time()

export_path = "/home/testopia-01/workspace/saved_models/{}".format(int(t))
model.save(export_path, save_format='tf')
```
이제 우리는 모델을 불러오고 같은 결과를 기대할 수 있다.
```
reloaded = tf.keras.models.load_model(export_path)

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

abs(reloaded_result_batch - result_batch).max()
```
```
> 0.0
```