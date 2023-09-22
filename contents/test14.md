---
date: '2020-07-29'
title: 'YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors 핵심 요약'
categories: ['Web', 'SEO', 'Optimization']
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