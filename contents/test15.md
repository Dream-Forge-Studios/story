---
date: '2020-07-29'
title: 'DETR: End-to-End Object Detection with Transformers 핵심 요약'
categories: ['Web', 'SEO', 'Optimization']
summary: 'DETR: End-to-End Object Detection with Transformers을 이해해보아요'
thumbnail: './test.png'
---

<div id="Abstract"></div>

# Abstract

<div id="기존의 Object Detection 방법들"></div>

## 기존의 Object Detection 방법들

- 기존의 방법들은 non-maximum suppression이나 anchor generation 같은 복잡한 과정과 손수 디자인해야 하는 요소들이 필요했습니다.
- 이러한 과정들은 모델을 복잡하게 만들고, 성능 최적화에 어려움을 주었습니다.

<div id="DETR의 접근 방식"></div>

## DETR의 접근 방식
- DETR은 이러한 복잡한 과정들을 제거하여, detection 파이프라인을 간소화하였습니다.
- DETR의 핵심 구성 요소는 Transformer의 encoder-decoder 구조와 '양자간 매칭(bipartite matching)'을 통한 유니크한 예측입니다.
- 이를 통해, DETR은 object와 이미지 전체의 context 사이의 관계를 추론하고, 최종적인 예측 set을 곧바로 반환할 수 있습니다.

<div id="DETR의 특징"></div>

## DETR의 특징
- 간단한 구조: DETR은 개념적으로 매우 간단하며, 특별한 라이브러리를 필요로 하지 않습니다.
- 고성능: COCO 데이터셋에서 Faster R-CNN과 동등한 정확도와 런타임 속도를 보였습니다.
- 다목적성: DETR은 쉽게 일반화할 수 있어, panoptic segmentation도 생성할 수 있습니다. 이는 경쟁 모델들을 뛰어넘는 성능을 보였습니다.
*Panoptic Segmentation: 이미지에서 모든 픽셀을 레이블링하여 각 픽셀이 어떤 객체에 속하는지, 또는 어떤 세그멘트에 속하는지를 예측하는 작업

<div id="Introduction"></div>

# Introduction

이전 모델들은 위 모델들의 성능은 거의 겹치는 예측들을 후처리하거나, anchor set을 디자인하거나, 휴리스틱하게 target boxes를 anchor에 할당하는 데에 크게 의존합니다.

본 논문은 위와 같은 과정을 간소화 하기 위해서 suurogate task를 패스하고 direct set prediction을 수행하는 방법론을 제안합니다.

*suurogate task: 주요 작업을 직접 수행하기 어려울 때, 그 주요 작업을 대신할 수 있는 보조적인 작업을 의미

## 기존 연구 비교

<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*yy4lkzUjatTTQX1d2Gq4qw.png">

