---
date: '2020-07-29'
title: 'FCOS, Fully Convolutional One-Stage Object Detection 논문 리뷰'
categories: ['Web', 'SEO', 'Optimization']
summary: 'FCOS, Fully Convolutional One-Stage Object Detection을 이해해보아요'
thumbnail: './test.png'
---

본 논문은 전통적인 object detection에서 사용되는 anchor-based 방식을 사용하지 않는 one-stage object detector입니다.

그렇다면 anchor-based 방식을 사용하지 않으므로써 얻을 수 있는 이점은 무엇일까요?

<div id="Anchor-free"></div>

# Anchor-free

<div id="이점"></div>

## 이점

1. 간결성
   - 앵커를 사용하지 않기 때문에 모델 구조가 간결해집니다. 이는 모델을 이해하고 구현하기 쉬워지며, 디버깅 또한 간편해집니다.

2. 하이퍼파라미터 최소화
   - 앵커 박스의 크기, 비율, 그리고 각 피처 레벨에서의 앵커 개수 등 앵커 관련된 하이퍼파라미터의 설정이 필요 없어집니다. 
   - 이는 학습 프로세스를 간소화하고, 최적의 하이퍼파라미터를 찾는 과정에서의 복잡성을 줄입니다.

3. 동적 범위
   - FCOS는 피처 맵의 각 위치에서 객체의 중심점을 예측하며, 이를 통해 다양한 크기의 객체를 동적으로 탐지할 수 있습니다. 이는 고정된 앵커 박스 크기와는 대조적입니다.

4. 성능 향상
   - 앵커 프리 접근법은 일부 데이터셋에서 더 나은 성능을 보이기도 합니다. 앵커가 없기 때문에, 여러 겹치는 앵커 박스를 처리하거나, 불필요한 앵커 박스를 필터링하는 과정이 필요 없습니다.

그렇다면 어떻게 Anchor-free를 하면서 객체가 있는 위치를 찾아내는 것일까?

<div id="작동 원리"></div>

# 작동 원리

<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc23myC%2Fbtq4Bxaa8II%2FqAlpPAUoK1aBeGOC7TM320%2Fimg.png">

1. Backbone Network

    - 기본적으로 FCOS는 ResNet 계열의 네트워크를 backbone으로 사용하며, 이를 통해 기본적인 feature extraction을 수행합니다.

2. Top-down Pathway & Lateral Connections

   - 가장 깊은 layer에서 시작해서 상위 layer로 역순으로 진행하면서 feature map의 크기를 업샘플링합니다. 
   - 동시에, 해당 크기의 bottom-up pathway의 feature map과 결합(lateral connection)을 이룹니다. 
   - 이렇게 함으로써, 세세한 정보와 coarse한 정보를 동시에 포착하는 feature map을 구성하게 됩니다.
   
3. Prediction Heads

   - 각 pyramid level에서는 객체의 위치 및 클래스를 예측하는데 필요한 별도의 convolutional heads를 가집니다.
     - Classification Head: 각 pixel이 어떤 클래스에 속하는지 예측하는 부분입니다. 여기서는 여러 개의 convolutional layers와 activation function을 거치며, 최종적으로 각 클래스에 대한 확률을 출력합니다.
     - Regression Head: 객체의 바운딩 박스의 위치와 크기를 예측하는 부분입니다. 일반적으로 4개의 값을 출력합니다: 바운딩 박스의 중앙점 (x, y)와 너비, 높이 (w, h).
     - Centerness Head (FCOS 특징): 바운딩 박스의 중심이 객체의 중심에 얼마나 가까운지를 예측하는 부분입니다. 이는 FCOS의 anchor-free 접근 방식에서 잘못된 탐지를 줄이는 데 도움을 줍니다.
   - 이를 통해 해당 pyramid level의 크기에 적합한 객체들을 탐지하게 됩니다.


<div id="특징"></div>

# 특징

1. l, t, r, b를 추정

<img style="width: 40%; margin-bottom: 0px;" id="output" src="https://gaussian37.github.io/assets/img/vision/detection/fcos/13.png">

2. FCOS의 장점
   - 모든 위치에서 바운딩 박스의 오프셋을 직접 학습하므로 앵커 기반 방법에서는 학습되지 않는 많은 위치에서 바운딩 박스의 정보를 학습할 수 있습니다.
   - 이유는?
     - 일반적인 앵커 기반의 모델은 미리 정의된 여러 크기와 비율의 앵커 박스를 사용합니다. 
     - 이 앵커 박스와 Ground Truth(GT) 바운딩 박스 간의 IoU(Intersection over Union)가 특정 임계값보다 높은 경우만, 해당 앵커가 foreground(즉, 객체)로 간주되어 학습에 사용됩니다.

3. 결과
   - feature map의 모든 픽셀에 대하여 해당 픽셀이 속한 클래스
      c
      와 bounding box를 추측하기 위한 4개의 값 (
      l
      ,
      t
      ,
      t
      ,
      b
      )를 추정하도록 네트워크를 구성합니다. 따라서 한 픽셀 당 (
      l
      ,
      t
      ,
      r
      ,
      b
      ,
      c
      ) 5개의 값을 예측합니다.
   - 각 스케일에서 얻은 예측값들을 독립적으로 사용하여 최종 탐지 결과를 생성합니다. 예를 들어, 큰 스케일의 feature map에서는 큰 객체들에 대한 예측을 주로 하며, 작은 스케일의 feature map에서는 작은 객체들에 대한 예측을 주로 합니다.
    
   *bounding box가 겹치는 경우 작은 박스는
   p
   3
   에서 검출되고 큰 박스는
   p
   6
   에서 검출
     
   <img style="width: 40%; margin-top: 20px;" id="output" src="https://gaussian37.github.io/assets/img/vision/detection/fcos/31.png">

<div id="Loss"></div>

# Loss
<img style="width: 80%; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FnVrjx%2Fbtq4AteyZIr%2FFrYDyJmqo27FVgGA7xoidk%2Fimg.png">

- classification을 위한 Loss는 Focal Loss를 사용하였고 Regression을 위한 Loss는 IoU Loss를 사용하였습니다. 각 Loss는 positive 샘플의 갯수 만큼 나누어서 normalization을 하였습니다.
- regression 부분에서
  λ
  를 사용하여 loss의 weight를 조절할 수 있으나 기본적으로는 1을 사용하였습니다. 클래스 인덱스가 0보다 큰 경우 즉, positive sample일 때에는 모든 feature map에서 연산이 되는 반면 클래스 인덱스가 0인 경우에는 negative sample로 간주하여 연산이 무시됩니다.


<div id="Center-ness for FCOS"></div>

# Center-ness for FCOS
<img style="width: 60%; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FM5Asr%2Fbtq4AuEnPp7%2FXVkzXUicqaypJAL4y93R51%2Fimg.png">

*l : 바운딩 박스의 중심으로부터 왼쪽 경계까지의 거리

*r : 바운딩 박스의 중심으로부터 오른쪽 경계까지의 거리

*t : 바운딩 박스의 중심으로부터 상단 경계까지의 거리

*b : 바운딩 박스의 중심으로부터 하단 경계까지의 거리

> FCOS에서는 상대적으로 low-quality bounding box가 많이 예측이 되었기 때문인데 이러한 box들은 실제 물체의 중앙점에서 멀리 떨어진 상태로 추정되는 경향 있는데 이를 해결하기 위해 고안

- 이유는?
    1. 앵커가 없기 때문에, 모든 위치에서 물체의 모든 크기와 형태에 대해 예측을 수행해야 합니다. 이로 인해 예측의 범위가 넓어져서 잘못된 위치에서도 바운딩 박스를 예측하는 경향이 생깁니다.
    2. FCOS가 각 위치에서 해당 위치가 물체의 중심점인지를 판별하는 과정에서 완벽하게 필터링 할 수 없습니다. 

- center-ness는 중심점과의 거리를 정규화 하여 어떤 박스가 중심점과 가깝다면 1에 가까운 값을 배정하고, 중심정과 멀다면 0에 가까운 값을 갖도록 만듭니다.
- 위 식에서 나타내는 center-ness를 classification score 출력에 곱해주게 되면 마지막 layer의 NMS에서 객체의 중앙과 멀리 떨어져서 위치를 추정한 박스는 걸러지도록 만들 수 있습니다.