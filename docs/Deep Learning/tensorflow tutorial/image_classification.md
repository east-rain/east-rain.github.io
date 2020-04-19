---
layout: default
title: 텐서플로우 이미지 분류하기
parent: 텐서플로우 튜토리얼
grand_parent: Deep Learning
nav_order: 1
---

# 이미지 분류
이 학습서는 이미지에서 고양이 또는 개를 분류하는 방법을 보여줍니다. tf.keras.Sequential 모델을 사용하여 이미지 분류기를 작성하고 tf.keras.preprocessing.image.ImageDataGenerator를 사용하여 데이터를 로드합니다.

다음과 같은 개념에 대한 실용적인 경험을 쌓고 직관력을 키울 수 있습니다.
* tf.keras.preprocessing.image.ImageDataGenerator를 사용한 data input 파이프라인을 구축하고 디스크에 있는 데이터를 모델과 함께 효율적으로 컨트롤 할 수 있습니다.
* Overfitting(과적합) - 식별하고 예방하는 방법
* 데이터 확대 및 드롭 아웃 - 컴퓨터 비전 task를 데이터 파이프 라인 그리고 이미지 분류 모델에 포함하여 Overfitting을 방지하는 중요한 기술

이 튜토리얼은 일반적인 머신러닝 워크플로우를 따릅니다.
1. 데이터 검사 및 이해
2. 입력 파이프 라인 구축
3. 모델 구축
4. 모델 훈련
5. 모델 테스트
6. 모델 개선 및 프로세스 반복

## 패키지 가져오기
필요한 패키지를 가져와 시작하겠습니다. os패키지는 파일 및 디렉토리 구조를 판독하는데 사용됩니다. NumPy는 파이썬 list 형태를 numpy array 형태로 변환하는데 사용되고, 요구되는 매트릭스 연산을 수행합니다. matplotlib.pyplot은 학습 그리고 검증 그래프를 보여줍니다.

모델을 구성하는 데 필요한 Tensorflow 및 Keras 클래스를 가져옵니다.

```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
```

## 데이터 로드
데이터 세트를 다운로드하여 시작하십시오. 이 튜토리얼은 Kaggle 의 필터링 된 버전의 Dogs vs Cats 데이터 세트를 사용합니다. 데이터 세트의 아카이브 버전을 다운로드하여 "/tmp/"디렉토리에 저장하십시오.
```
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
```

데이터셋은 다음과 같은 디렉토리 구조를 가집니다.
```
cats_and_dogs_filtered
|__ train
    |______ cats: [cat.0.jpg, cat.1.jpg, cat.2.jpg ....]
    |______ dogs: [dog.0.jpg, dog.1.jpg, dog.2.jpg ...]
|__ validation
    |______ cats: [cat.2000.jpg, cat.2001.jpg, cat.2002.jpg ....]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg, dog.2002.jpg ...]
```

압축을 해제한 후 학습 및 검증 세트에 적합한 파일 경로로 변수를 지정하십시오.
```
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures
```

## 데이터 이해
훈련 및 검증 디렉토리에 고양이와 개 이미지가 몇 개인 지 살펴 보겠습니다.

```
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val
```
```
print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
```
```
total training cat images: 1000
total training dog images: 1000
total validation cat images: 500
total validation dog images: 500
--
Total training images: 2000
Total validation images: 1000
```

편의를 위해 데이터 세트를 사전 처리하고 네트워크를 훈련하는 동안 사용할 변수를 설정하십시오.

```
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
```

## 데이터 준비
네트워크에 데이터를 공급하기 전에 이미지를 부동 소수점 tensor로 변환해야 합니다

1. 디스크에서 이미지를 읽습니다.
2. 이 이미지의 내용을 디코딩하여 RGB 내용에 따라 적절한 grid 형식으로 변환합니다.
3. 이 값들을 부동 소수점 tensor로 변환합니다.
4. 신경망은 작은 입력 값을 처리하는 것을 선호하므로 텐서를 0에서 255 사이의 값에서 0에서 1 사이의 값으로 재조정하십시오.

다행히 이러한 모든 작업은 tf.keras에 의해 제공되는 ImageDataGenerator 클래스에서 수행 할 수 있습니다. 디스크에서 이미지를 읽고 적절한 텐서로 사전 처리 할 수 ​​있습니다. 또한 이 이미지들을 텐서의 배치로 변환하여 네트워크 학습에 도움을 줄 수 있습니다.

```
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
```

학습 그리고 검증 이미지를 위해 생성기를 정의하고 난 뒤 flow_from_directory 메서드는 디스크로부터 이미지를 불러오고, rescaling을 적용하고, 요구되는 면적에 맞게 이미지를 resize 합니다.

```
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
```
```
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
```

### 학습 이미지 시각화
트레이닝 생성기로부터 이미지의 배치를 추출하고 시각화한다
```
sample_training_images, _ = next(train_data_gen)
```
next 함수는 데이터셋의 배치를 반환한다. next 함수의 반환값은 (x_train, y_train) 형태이다. 여기서 x_train은 학습 특징들이고 y_train은 라벨이다. 학습 데이터만 보여주기 위해서 라벨은 숨긴다
```
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```
```
plotImages(sample_training_images[:5])
```
<img src="https://www.tensorflow.org/tutorials/images/classification_files/output_d_VVg_gEVrWW_0.png"></img><br/>

## 모델 생성하기
모델에는 max pool layer를 가지고 있는 3개의 convolution block이 존재한다. 512개의 유닛이 존재하는 fully connected layer가 가장 상단에 존재하고 활성화 함수로 relu 함수가 사용된다.
```
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
```

### 모델 컴파일하기
이 튜토리얼에서는 ADAM 옵티마이저와 cross entropy 손실 함수를 선택하였다. 학습 그리고 검증 정확도를 보기 위해 각각의 학습 에폭에 metrics 아규먼트를 넘겨준다.
```
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

### 모델 요약
summary 메서드를 이용하여 네트워크의 모든 레이어를 관찰할 수 있다.
```
model.summary()
```
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 150, 150, 16)      448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 75, 75, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 75, 75, 32)        4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 37, 37, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 37, 37, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 18, 18, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 20736)             0         
_________________________________________________________________
dense (Dense)                (None, 512)               10617344  
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513       
=================================================================
Total params: 10,641,441
Trainable params: 10,641,441
Non-trainable params: 0
_________________________________________________________________
```

### 모델 학습시키기
**ImageDataGenerater** 클래스의 **fit_generatoer** 메서드를 이용해서 네트워크를 학습시킨다.
```
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
```

### 트레이닝 결과 시각화하기
```
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

<img src="https://www.tensorflow.org/tutorials/images/classification_files/output_K6oA77ADVrWp_0.png"></img><br/>

그림에서 알 수 있듯이 트레이닝 정확도와 밸리데이션 정확도는 큰 폭으로 증가하였습니다. 그리고 밸리데이션 정확도는 70% 정도를 달성하였습니다.
무엇이 잘못되었는지 살펴보고 모델의 전반적인 성능을 향상 시켜보겠습니다.

## 과적합(Overfitting)
위의 그림에서 학습 정확도는 시간이 지날수록 선형적으로 증가하지만, 검증 정확도는 70%를 넘지 못합니다. 또한 학습데이터와 검증데이터의 정확도 차이는 과적합의 명확한 신호입니다.

적은 개수의 학습 데이터에서, 학습 데이터의 원치않는 디테일이나 노이즈를 모델이 학습해버리는 경우가 있습니다. 이것은 새로운 데이터를 인식하는 모델의 정확도에 않좋은 영향을 끼칩니다. 이 현상은 과적합이라고 알려져있습니다. 이 의미는 모델이 새로운 데이터셋이 들어왔을때 일반화하지 못한다는 의미입니다.

오버피팅을 방지하는 여러가지 방법들이 학습 단계에 존재합니다. 이 튜토리얼에서 우리는 *데이터 보강* 방법과 드롭아웃 기법을 사용할 것입니다.

## 데이터 보강
과적합은 일반적으로 트레이닝 데이터가 부족할 때 발생합니다. 이 문제를 해결하는 한 방법은 데이터를 보강하여 충분한 개수의 트레이닝 데이터를 얻는 것입니다. 데이터 보강은 존재하는 학습 데이터에서 그럴듯하게 보이는 이미지를 랜덤하게 생성해내는 방법이다. 목표는 모델은 절대 두번이상 같은 데이터를 학습하지 않는 것이다. 이것은 모델이 더 다양한 데이터를 접하고 일반화하게 만든다.

tf.keras의 **ImageDataGenerator** 클래스를 사용한다. 데이터셋의 다양한 변화를 시도한다.

### 데이터 보강과 시각화
랜덤 수평 flip 보강을 시작하고, 각각의 이미지들의 변화 후 모습을 보자

### 수평 flip 적용하기
ImageDataGeneraotr 클래스에 인자로 horizontal_flip 을 넘겨준다 그리고 True 값을 줘서 이 인자를 적용시킨다.
```
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))
```

트레이닝 에제로부터 하나의 샘플 이미지를 가져오고 5번 반복한다, 즉 같은 이미지에 대해서 5번 증강을 시도한다
```
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
```
```
# Re-use the same custom plotting function defined and used
# above to visualize the training images
plotImages(augmented_images)
```
<img src="https://www.tensorflow.org/tutorials/images/classification_files/output_EvBZoQ9xVrW9_0.png"></img><br/>

### 이미지를 랜덤하게 회전하기
회전이라 불리는 증강방법을 살펴보자 그리고 트레이닝 예제들을 45도 회전해보자 랜덤하게

```
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
```
```
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
```
```
plotImages(augmented_images)
```
<img src="https://www.tensorflow.org/tutorials/images/classification_files/output_wmBx8NhrVrXK_0.png"></img><br/>

### 확대 증강 적용하기
이미지를 랜덤하게 50프로 확대하는 증강법을 알아보자
```
# zoom_range from 0 - 1 where 1 = 100%.
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5) # 
```
```
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
```
```
plotImages(augmented_images)
```
<img src="https://www.tensorflow.org/tutorials/images/classification_files/output_-KQWw8IZVrXZ_0.png"></img><br/>

### 모두 적용해보기
이전의 증강법을 모두 적용해보자. 이 방법으로 너는 rescale, 45도 회전, 가로 쉬프트, 세로 쉬프트, 수평 플립 그리고 확대 증강법을 사용할 수 있다.
```
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )
```
```
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')
```
하나의 이미지에 대해서 5번 이 증강법을 랜덤하게 적용했을 때 어떤 변화가 일어나는지 확인해보자
```
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
```
<img src="https://www.tensorflow.org/tutorials/images/classification_files/output_z2m68eMhVrXm_0.png"></img><br/>

### 검증 데이터 생성기 만들기
일반적으로 데이터 증강은 훈련 데이터에 대해서만 시행한다. 이번에는, 검증 데이터를 단순히 rescale하고 ImageDataGenerator를 사용하여 검증 데이터를 배치로 변환하는 것을 해보겠다.
```
image_gen_val = ImageDataGenerator(rescale=1./255)
```
```
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')
```

## 드롭아웃 - Dropout
과적합을 방지하는 다른 테크닉은 네트워크에 드롭아웃을 적용하는 것이다. 이것은 정규화의 한 방법으로써 네트워크에서 작은 부분의 가중치만 선택하게 함으로써 가중치를 분산시키고 작은 훈련 데이터에서 과적합을 방지하는 것이다.

드롭아웃을 레이어에 적용할 때, 훈련 과정동안 랜덤하게 뉴런의 일정부분을 꺼놓는다. 드롭아웃은 0.1, 0.2, 0.4와 같이 아주 작은 인풋을 받는다. 이 의미는 10%, 20%, 40%의 뉴런에 대해서 랜덤하게 드롭아웃을 적용하라는 의미다.

어떤 특정한 레이어에 대해 0.1 드롭아웃을 적용하면, 이것은 각 에폭마다 랜덤하게 10%의 뉴런을 끄고 학습을 진행한다.

이 드롭아웃 기법을 컨볼루션과 fully-connected layers에 적용하여 새롭게 모델을 설계해라

## 드롭아웃을 사용하여 새로운 네트워크 생성하기
여기, 너는 첫번째와 마지막 max pool 레이어에 드롭아웃을 적용한다. 드롭아웃은 각 훈련 에폭동안 20%의 뉴런을 제외하고 학습을 진행한다. 이것은 트레이닝 데이터셋에서 오버피팅을 방지하는데 도움을 준다
```
model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
```

### 모델 컴파일하기
네트워크에 드롭아웃을 적용한 다음에 모델을 컴파일하고 레이어의 서머리를 확인해보자
```
model_new.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model_new.summary()
```
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 150, 150, 16)      448       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 75, 75, 16)        0         
_________________________________________________________________
dropout (Dropout)            (None, 75, 75, 16)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 75, 75, 32)        4640      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 37, 37, 32)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 37, 37, 64)        18496     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 18, 18, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 18, 18, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 20736)             0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               10617344  
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 513       
=================================================================
Total params: 10,641,441
Trainable params: 10,641,441
Non-trainable params: 0
_________________________________________________________________
```

### 모델 학습하기
성공적으로 데이터 증강을 학습 데이터들에 적용하고 네트워크에 드롭아웃을 적용하고 이 새로운 네트워크를 학습시켜보자!
```
history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
```

### 모델 시각화하기
훈련 후에 새로운 모델을 시각화한다. 너는 상당히 과적합이 줄어든 것을 확인할 수 있다. 정확도는 에폭이 증가할 수록 점점 더 증가한다.
```
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```
<img src="https://www.tensorflow.org/tutorials/images/classification_files/output_7BTeMuNAVrYC_0.png"></img><br/>
