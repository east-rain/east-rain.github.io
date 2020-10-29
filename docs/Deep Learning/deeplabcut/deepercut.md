---
layout: default
title: deepercut(base model)
parent: DeepLabCut
grand_parent: Deep Learning
nav_order: 101
date: 2020-05-27
---

# Deepercut 논문 요약
Bottom-up 방식의 pose estimation 모델이다.

1. Bottom-up 
    영상에서 키포인트를 먼저 찾은다음, 그 키포인트들 간에 관계를 분석하여 자세를 추정한다.
    정확도는 Top-down 방식에 비교해서 떨어지지만 처음에 사람을 detect 하는 과정을 생략하기 때문에 빠르다는 장점이 있고,
    Real-time에 사용하는 모델들에 주로 사용되는 방식이다.

2. Top-down
    영상에서 먼저 사람의 위치를 detect하고, detect 되어진 bounding box 내부에서 자세를 추정하는 방식이다.
    정확도는 Bottom-up에 비해서 높지만, Multi-person일 경우에 detectin 된 사람마다 자세를 추정하기 때문에 느리다는 단점이 있다.
    
과거의 pose 추정 모델들이 사람의 구조적 특성에 집중하였다면, 요즘의 pose 추정 모델들은 강력해진 CNN 모델들을 이용하여
신체 자체를 측정하는데 좀 더 초점을 둔다.

먼저 학습 된 CNN 모델을 통해서 신체부위의 클래스를 예측한 후에 SVM이 신체 포인트 쌍의 관계 점수를 매기는데 사용된다.

* SVM - 서포트 백터 머신
서포트 벡터 머신(support vector machine, SVM)은 기계 학습의 분야 중 하나로 패턴 인식, 자료 분석을 위한 지도 학습 모델이며, 주로 분류와 회귀 분석을 위해 사용한다. 두 카테고리 중 어느 하나에 속한 데이터의 집합이 주어졌을 때, SVM 알고리즘은 주어진 데이터 집합을 바탕으로 하여 새로운 데이터가 어느 카테고리에 속할지 판단하는 비확률적 분류 모델을 만든다.
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Kernel_Machine.svg/1920px-Kernel_Machine.svg.png"/>


DeeperCut을 이해하려면 먼저 DeepCut을 알아야 한다. DeepCut은 multi-persion pose estimation의 state-of-the-art이다.

집합 D = 신체 부위 후보군들(주어진 이미지에서 예측되는 신체 부위들)
집합 C = 신체 부위 클래스들(머리, 어깨, 무릎 등의 신체 부위들)

집한 D는 일반적으로 body part detector에 의해 생성되며, 각각의 후보자 d ∈ D는 모든 body part class c ∈ C 에 대응하는 점수를 가지고 있다. 이 점수에 기반하여 DeepCut은 각각의 d가 c에 대해 가지는 점수 세트 αdc ∈ R를 만든다. 추가적으로 모든 

 
최근 이미지넷으로 훈련시킨 Resnet이 인간의 판단 능력을 넘어섰다. 그러한 Resnet을 이용하였으며 152개의 layer를 쌓았다. 
