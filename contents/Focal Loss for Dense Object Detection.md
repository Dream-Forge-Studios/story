---
date: '2020-07-29'
title: 'Focal Loss for Dense Object Detection 논문 리뷰'
categories: ['CV']
summary: 'Focal Loss for Dense Object Detection을 이해해보아요'
thumbnail: './test.png'
---

본 논문은 One-stage detector의 학습 불균형이라는 단점을 해결하기 위한 내용이다.

그렇다면 One-stage detector가 무엇일까요?

<div id="One-stage detector"></div>

# One-stage detector

> 객체의 위치와 클래스를 동시에 한 단계에서 예측하는 방식입니다.
> 
> 이러한 방식의 주요 장점은 빠른 속도와 간단한 구조입니다.

대표적인 예로는 YOLO(You Only Look Once)와 SSD(Single Shot MultiBox Detector)가 있습니다.

<div id="작동 원리"></div>

## 작동 원리

1. 입력 이미지
   - 일반적으로 전처리 단계를 거쳐 고정된 크기로 리사이징된 이미지가 네트워크의 입력으로 사용됩니다.

2. 특징 추출
    - 주어진 이미지에서 특징을 추출하기 위해 Convolutional Neural Network (CNN)이 사용됩니다. 이 CNN은 일반적으로 이미지 분류 문제에 대해 사전 훈련된 모델을 기반으로 합니다 (예: VGG16, ResNet 등).

3. 그리드 분할
    - 추출된 특징 맵을 그리드 셀로 나눕니다. 예를 들어, YOLOv1에서는 7x7 그리드를 사용했습니다.

4. 객체 예측
   - 각 그리드 셀에는 물체가 있을 확률(객체의 중심이 존재하는지), 바운딩 박스 정보(오프셋), 그리고 해당 물체의 클래스에 대한 확률 정보를 예측합니다.

<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://blog.nerdfactory.ai/assets/images/posts/2021-07-01-You-Only-Look-Once-YOLO/Untitled3.png">
*YOLOv1의 구조

5. 앵커 박스 사용
    - SSD나 YOLOv2부터는 앵커 박스 또는 기본 박스(default boxes)를 사용하여 각 그리드 셀에서 여러 개의 바운딩 박스를 예측합니다. 앵커 박스는 다양한 크기와 비율을 가진 사전 정의된 박스로, 각 박스에 대해 물체가 있을 확률과 클래스 확률, 그리고 바운딩 박스 오프셋을 예측합니다.

6. 예측값 필터링:
    - 일반적으로 confidence threshold를 사용하여 낮은 확률을 가진 예측값을 필터링합니다.
      - confidence threshold
        - 일정 수준 이상의 confidence score를 가진 예측만을 선택함으로써 결과의 수를 줄일 수 있습니다.  
        - Threshold를 너무 높게 설정하면 많은 객체를 놓칠 수 있으며, 너무 낮게 설정하면 잘못된 예측이 많아질 수 있습니다.

7. 비최대 억제 (NMS):
    - 여러 겹치는 박스 중에서 가장 확률이 높은 박스만을 선택하는 과정으로, 겹치는 박스들을 제거하여 최종 결과를 보다 명확하게 만듭니다.
8. 최종 객체 감지 결과:
    - NMS를 거친 후, 최종적으로 선택된 바운딩 박스와 클래스 레이블이 최종 감지 결과로 반환됩니다.

그런데, 해당 방식은 학습 중 배경에 대한 바운딩 박스의 수가 실제 객체에 대한 바운딩 박스의 수보다 훨씬 많이 생성 됩니다.

이는 학습 중에서 배경에 대한 박스를 출력하면 오류라고 학습이 되지만 그 빈도수가 너무 많다는 것이 학습에 방해가 된다는 뜻입니다. (easy negative)

그래서 이러한 문제를 해결하기 위해 Focal Loss를 고안하였습니다.

<div id="Focal Loss"></div>

# Focal Loss

배경에 해당하는 클래스가 주로 많기 때문에 이전에 활용되었던 Cross Entropy Loss는  학습 데이터에서 긍정적인 샘플(실제 객체)과 부정적인 샘플(배경) 간의 불균형이 발생합니다.

> Focal Loss는 이러한 문제를 해결하기 위해, 잘못 분류된 예제에 더 큰 가중치를 부여하고, 올바르게 분류된 예제에는 낮은 가중치를 부여하는 방식으로 작동합니다. 

<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://gaussian37.github.io/assets/img/dl/concept/focal_loss/1.png">

- 위 그래프는 γ=0일 때, Cross Entropy Loss와 같고 γ가 커질수록 잘못 분류 됬을 때 더 큰 가중치를 부여합니다.
- γ의 값이 5일 때 loss가 가장 빠르게 줄어드는 것을 볼 수 있습니다. 
- Cross Entropy Loss의 잘 못 분류된 확률을 추가했고, γ의 따라 정도를 조절합니다.
- 본 논문에서는 γ를 **focusing parameter**라고 부릅니다.

    