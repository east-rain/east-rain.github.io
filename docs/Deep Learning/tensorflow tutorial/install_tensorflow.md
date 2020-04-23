---
layout: default
title: 텐서플로우 설치 - windows
parent: 텐서플로우 튜토리얼
grand_parent: Deep Learning
nav_order: 99
---

# 텐서플로우 설치 - windows

1. Python 3 설치 - anaconda 사용<br>
https://www.anaconda.com/distribution/ 에서 anaconda installer 다운로드 및 설치

2. NVIDIA GPU driver 설치<br>
https://www.nvidia.co.kr/Download/index.aspx?lang=kr

3. NVIDIA CUDA Toolkit 설치<br>
https://developer.nvidia.com/cuda-downloads

4. cuDDN 설치<br>
https://developer.nvidia.com/cudnn<br>
다운받은 압축파일을 적당한 위치에 푼다.
여기서는 C:\tools/cuda 에 풀겠다.

5. 환경변수 추가<br>
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin;<br>
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\extras\CUPTI\libx64;<br>
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include;<br>
C:\tools\cuda\bin;<br>
해당 경로들을 환경변수에 추가한다.

6. 텐서플로우 설치 - pip<br>
```
$ pip install tensorflow
```

텐서플로우2 버전부터는 해당 명령어로 CPU와 GPU를 모두 설치해준다

7. 설치 확인
```
$ python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

> 2020-04-21 20:36:09.675162: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
  tf.Tensor(-342.69507, shape=(), dtype=float32)
```
결과가 나오면 설치가 잘 된것이다.
위에 I 메시지는 내 cpu는 더 많은 명령어를 지원하지만, pip로 받은 텐서플로우 빌드 버전에서는 해당 명령어를 사용하지 못한다는 의미이다. 만약 해당 메시지가 싫거나 좀 더 빠른 동작을 원한다면 텐서플로우 소스를 받아서 빌드하면 된다.
