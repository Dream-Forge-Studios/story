---
date: '2020-07-29'
title: 'YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors 핵심 요약'
categories: ['Computer Vision']
summary: 'YOLOv7을 이해해보아요'
thumbnail: './test.png'
---

YOLO v7은 5 ~ 160 FPS 범위의 속도와 정확도 측면에서 현재까지 나온 모든 Object Detector의 성능을 능가한다고 합니다.(2022년 7월 기준)

그렇다면 어떻게 이러한 성능을 달성했는지 살펴 보겠습니다.

<div id="E-ELAN"></div>

# E-ELAN

> Extended-ELAN

ELAN(Efficient Layer Aggregation Network)을 확장한 것으로, ELAN의 장점을 유지하면서도 성능을 향상시켰습니다.

그렇다면 ELAN은 무엇일까요?

<div id="ELAN"></div>

## ELAN

> 기존의 SR 모델(Super Resolution Model)들이 낮은 해상도 이미지에서 장거리 정보를 추출하는 데 어려움을 겪는 문제를 해결하기 위해 제안

*SR 모델(Super Resolution Model) : 저해상도 이미지를 고해상도 이미지로 복원하는 모델

*장거리 정보 : 멀리 떨어진 픽셀 간의 관계

- SR 모델이 낮은 해상도 이미지에서 장거리 정보를 추출하는 데 어려움을 겪는 이유
    - 저해상도 이미지에는 장거리 정보가 부족합니다. 저해상도 이미지는 고해상도 이미지에서 일부 픽셀이 제거된 이미지입니다. 따라서 저해상도 이미지에는 장거리 정보가 부족할 수 있습니다.
    - SR 모델은 일반적으로 짧은 거리의 정보를 더 잘 처리하도록 설계됩니다. SR 모델은 일반적으로 convolutional neural network(CNN)을 기반으로 합니다. CNN은 가까운 픽셀 간의 관계를 학습하도록 설계되었습니다. 따라서 SR 모델은 짧은 거리의 정보를 더 잘 처리할 수 있습니다.


<div id="기존 연구 비교"></div>

## 기존 연구 비교

<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*yy4lkzUjatTTQX1d2Gq4qw.png">
*"3x3, 2c, 2c, 2"의 의미: 3x3 크기의 convolutional 필터를 사용하는데 입력 채널 2c, 출력 채널 2c, 2번 반복

-> 2번 반복하는데 출력 채널이 같은 이유는 group convolution

- VoVNet
  - DenseNet의 구조를 기반으로 합니다. DenseNet은 이전 레이어의 출력을 다음 레이어의 입력으로 연결하여 정보의 손실을 줄이는 구조입니다. 그러나 DenseNet은 계산량이 많다는 단점이 있습니다.
    <img style="width: 70%; margin-bottom: 40px;" id="output" src="https://pytorch.org/assets/images/densenet1.png">
  *DenseNet의 구조
  - OSA(One-Shot Aggregation)를 통해 input channel 채널 수를 일정하게 유지
    - DenseNet은 1x1 Depthwise conv를 사용하여 채널 수를 줄입니다. 1x1 Depthwise conv는 채널 수를 줄이는 데 효과적이지만 계산량이 많습니다.  
    - VoVNet에서는 마지막 feature map을 여러 개의 그룹으로 나누고, 각 그룹의 feature를 결합하여 새로운 feature를 생성합니다. 
    - 새로운 feature의 채널 수는 마지막 feature map의 채널 수와 동일하게 만들기 위해 다음과 같은 방법을 사용합니다.
              
      *가중 평균(Weighted Average): 각 그룹의 중요도를 고려하여 새로운 feature를 생성합니다. 따라서 성능이 가장 좋을 수 있습니다. 그러나 계산량이 가장 많이 소요됩니다.
          
      *최대값(Max): 각 그룹에서 가장 중요한 정보를 선택하여 새로운 feature를 생성합니다. 따라서 성능이 좋을 수 있습니다. 그러나 계산량이 가중 평균보다 적게 소요됩니다.
          
      *최소값(Min): 각 그룹에서 가장 불필요한 정보를 제거하여 새로운 feature를 생성합니다. 따라서 성능이 약간 떨어질 수 있습니다. 그러나 계산량이 가장 적게 소요됩니다.
    
    - 마지막 feature map에서 모든 feature를 한 번에 연결하여 입력 크기를 일정하게 유지하면서 출력의 크기를 증가시킵니다.

- CSPVoVNet
  - VoVNet에 Cross Stage Partial(CSP)구조를 추가한것으로 input channel을 반으로 나누어(partial) 왼쪽의 c는 그대로 transition layer에 더해집니다.
  
    *transition layer: 딥러닝 모델에서 feature map의 크기를 줄이고, 채널 수를 조정하는 역할을 하는 레이어 
  
  - 나눠진 c 때문에 기존보다 gradient flow가 truncate되어 과도한 양의 gradient information을 방지합니다.

    *gradient flow는 모델의 가중치를 업데이트하기 위한 정보입니다. 이 정보가 너무 많으면 오히려 모델의 학습을 방해할 수 있습니다.
  
  - c를 나누고 transition layer에서 병합하기 때문에 gradient path는 2배로 증가하여 다양한 features를 학습할 수 있습니다.

- ELAN
    - CSPVoVNet에서 모듈을 간소화 시키므로써 가장 짧고 그리고 가장 긴 gradient path 차이를 더 극대화하였습니다.
    - CSPVoVNet의 비해 gradient path를 짧고 균등하게 하므로써 deep한 네트워크도 학습가능하고 수렴도 효과적으로 잘되게 합니다.
    - CSPVoVNet의 비해 Inference time이 짧고 학습 속도도 빠릅니다. 

하지만 Computation block을 너무 많이 쌓게 되면 

- 네트워크의 안정성(stable state)이 훼손
  - 이는 네트워크가 너무 깊어질 경우 발생하는 vanishing/exploding gradient 문제, 과적합(overfitting), 학습의 불안정성 등의 문제를 야기할 수 있습니다.
- 낮은 parameter utilization
  - 네트워크의 깊이가 극도로 증가하면, 많은 파라미터가 실제로 학습에 기여하지 않을 수 있습니다

이러한 ELAN의 단점때문에 YOLOv7에서는 마음껏 쌓아도 학습이 잘되도록 하기 위해 E-ELAN을 제안합니다.

<div id="E-ELAN"></div>

## E-ELAN

<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://s4.itho.me/sites/default/files/images/%EF%BC%88%E6%94%BE%E7%AC%AC1%E9%A0%81%E4%B8%8B%E9%9D%A2%EF%BC%89%E8%A3%9C%E5%9C%96-%E5%9C%96%E7%89%87%E4%BE%86%E6%BA%90-%E7%8E%8B%E5%BB%BA%E5%A0%AF.png">

1. Expand, shuffle, merge cardinality를 통해 compuational block을 많이 쌓아도 학습능력이 우수합니다.
2. 오직 computational block만 바뀌고 transition layer는 전혀 바뀌지 않았습니다.

- ResNeXt 모델의 cardinality 활용하여 expand
  <img style="width: 80%; margin-top: 40px;" id="output" src="https://www.researchgate.net/profile/Yahia-Said/publication/357905310/figure/fig2/AS:1113612953370625@1642517426275/Building-block-of-the-ResNeXt-model-compared-to-ResNet-model.ppm">
  - ResNeXt에서는 여러 개의 동일한 변환 블록을 병렬로 배치하여 각 블록이 동일한 연산을 수행하게 합니다. 그런 다음, 이러한 병렬 블록들의 출력을 합치게 됩니다.
  - 위와 같은 연산으로 늘어난 채널 수는 group convolution을 통해 입력과 출력이 동일하도록 유지합니다.

<div id="Compound Model Scaling"></div>

# Compound Model Scaling

<div id="Model Scaling"></div>

## Model Scaling

> 모델의 학습 능력을 향상시키기 위해 모델의 크기를 조정하는 것

1. Depth Scaling (깊이 스케일링)
   - 모델의 레이어 수를 증가시켜 모델의 깊이를 늘립니다. 이를 통해 모델이 더 복잡한 패턴을 학습할 수 있게 됩니다.
2. Width Scaling (너비 스케일링)
   - 각 레이어의 뉴런 수를 증가시켜 모델의 너비를 늘립니다. 이를 통해 모델이 더 다양한 특징을 학습할 수 있게 됩니다.
3. Resolution Scaling (해상도 스케일링)
   - 입력 이미지의 해상도를 증가시킵니다. 이를 통해 모델이 더 세밀한 정보를 학습할 수 있게 됩니다.

<div id="concatenationbased models"></div>

## concatenationbased models

<img style="width: 80%; margin-bottom: 40px;" id="output" src="https://viso.ai/wp-content/uploads/2022/08/explained-how-yolov7-compound-model-scaling-works-768x640.png">

연결 기반 모델은 위의 그림과 같이 a에서 b로 scaling하면 전환(Transition) 레이어의 in-degree는 감소하거나 증가됩니다.

따라서, 연결 기반 모델을 위한 상응하는 복합 모델 스케일링 방법을 c를 제안하였습니다.

*Depth Scaling: 네트워크의 깊이를 증가시키는 것으로, 레이어를 추가하여 네트워크를 더 깊게 만듭니다. 이를 통해 모델은 더 복잡한 특징을 학습할 수 있게 됩니다.

*Width Scaling: 네트워크의 너비를 증가시키는 것으로, 각 레이어에서의 필터(뉴런)의 개수를 증가시킵니다. 이를 통해 모델은 더 다양한 특징을 동시에 학습할 수 있게 됩니다.

<div id="Planned re-parameterized convolution"></div>

# Planned re-parameterized convolution

<div id="re-parameterized convolution(RepConv)"></div>

## re-parameterized convolution(RepConv)


Re-parameterized Convolution(RepConv)은 일반적인 컨볼루션 연산을 개선한 방법으로, 모델의 효율성과 성능을 향상시키기 위해 제안되었습니다.
이 방법은 기존의 컨볼루션 연산의 파라미터를 재매개변수화(re-parameterize)하여, 연산의 효율성을 높이고, 모델의 표현력을 개선합니다.

> RepConv는 기존의 컨볼루션 연산을 개선하기 위한 방법으로, 3x3 convolution, 1x1 convolution 및 identity connection을 하나의 convolutional layer에 결합합니다.

1. 3x3 Convolution
   - 목적: 주변 픽셀과의 상호작용을 학습하여 공간적인 특징을 추출합니다.
   - 작동 방식: 각 입력 픽셀이 3x3 크기의 이웃과 함께 컨볼루션 연산을 수행합니다.
2. 1x1 Convolution:
   - 목적: 채널 간의 상호작용을 학습하여 채널의 수를 조정하거나 피쳐 맵의 정보를 집약합니다.
   - 작동 방식: 각 입력 픽셀이 1x1 크기의 커널과 컨볼루션 연산을 수행합니다.
3. Identity Connection:
   - 목적: 입력을 직접 출력에 연결하여 그래디언트의 흐름을 개선하고, 네트워크의 학습을 안정화합니다.
   - 작동 방식: 입력 피쳐 맵이 변형 없이 출력 피쳐 맵에 더해집니다.

RepConv와 다른 아키텍처의 조합 및 해당 성능을 분석한 후 RepConv의 identity connection이 ResNet의 잔차와 DenseNet의 연결을 대체하여 different feature maps에 대해 더 다양한 gradient를 제공한다는 것을 발견했습니다.

따라서 본 논문에서는  RepConv with-out identity connection (RepConv)을 사용하여 planned reparameterized convolution의 아키텍쳐를 설계합니다.

 <img style="width: 100%; margin-top: 40px;" id="output" src="https://viso.ai/wp-content/uploads/2022/08/yolov7-architecture-planned-re-parameterized-model.png">

<div id="Coarse for Auxiliary and Fine for Lead Loss"></div>

# Coarse for Auxiliary and Fine for Lead Loss

 <img style="width: 100%; margin-top: 40px;" id="output" src="https://viso.ai/wp-content/uploads/2022/08/yolov7-architecture-auxiliary-head-and-label-assigner.png">

YOLO 아키텍처는 객체 탐지를 수행하는 딥러닝 모델로, 백본(Backbone), 넥(Neck), 그리고 헤드(Head)의 세 부분으로 구성되어 있습니다

- 백본: 모델의 기본 구조를 형성하며, 주로 특징 추출을 담당합니다.
- 넥: 백본에서 추출된 특징 맵(Feature Map)을 처리하고, 이를 헤드로 전달합니다.
- 헤드: 최종적으로 객체의 위치, 클래스 등을 예측합니다.

YOLOv7은 단일 헤드로 제한되지 않습니다. 그러나 다중 헤드 프레임워크가 도입된 것은 이번이 처음이 아닙니다. 

DL 모델에서 사용되는 기술인 Deep Supervision은 여러 헤드를 사용합니다 . 

YOLOv7에서는 최종 출력을 담당하는 헤드를 리드 헤드(Lead Head)라고 합니다. 그리고 중층 훈련을 보조하는데 사용되는 헤드를 보조 헤드(Auxiliary Head) 라고 합니다 .

- 리드 헤드(Lead Head): 모델의 최종 출력을 담당합니다.
- 보조 헤드(Auxiliary Head): 중간 계층에서 훈련을 보조합니다. 이는 모델의 학습을 돕기 위해 도입되었습니다.

<div id="Label Assigner"></div>

## Label Assigner

과거의 방식은 모델에게 정확한 라벨(Ground Truth, GT)을 할당하는 것이었습니다. 이를 Hard Label이라고 부릅니다.

> 모델의 예측 결과와 실제 정답(GT)을 함께 고려하여, Soft Label을 할당하는 방식

이 Soft Label Set는 auxiliary head 또는 lead head 모두에 대한 target training mode로 사용됩니다

<div id="필요한 이유"></div>

## 필요한 이유

> generalized residual learning

1. Soft Label은 Lead head가 학습한 정보를 반영합니다.
2. 이는 lead head가 학습한 정보를 auxiliary head가 추가로 학습함으로써, lead head는 더 고차원의, 또는 더 복잡한 패턴을 학습할 수 있게 된다는 것을 의미합니다.
3. shallower auxiliary head가 lead head가 학습한 정보를 직접 학습하게 함으로써 lead head는 아직 학습되지 않은 learning residual information에 더 집중할 수 있습니다.

<div id="Experiments"></div>

# Experiments

<img style="width: 100%; margin-top: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb4cXxY%2FbtrHrMtH3HO%2Fw82K9xD7wSSbfWxutrI2KK%2Fimg.png">

<img style="width: 100%; margin-top: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcROtEJ%2FbtrHvo6snjM%2FhdX3UIr1ovQ6EzyXK3AuG1%2Fimg.png">

<img style="width: 100%; margin-top: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUZ8ie%2FbtrHxNj1Wq8%2FPUnPv60RGscU1pI0Q4Ma9k%2Fimg.png">
