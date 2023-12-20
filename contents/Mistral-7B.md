---
date: '2023-12-05'
title: 'Mistral 7B 논문 리뷰'
categories: ['Large Language']
summary: 'Mistral 7B 완벽 이해하기.'
thumbnail: './test.png'
---

<div id="ABSTRACT"></div>

# ABSTRACT

Mistral 7B는 Llama 2 13B와 Llama 1 34B 모델보다 여러 벤치마크에서 더 우수한 성능을 보였습니다. 특히 추론, 수학, 코드 생성 분야에서 두드러진 성능을 보였습니다.

<br>

'그룹화된 쿼리 주의(Grouped-Query Attention, GQA)' 기술을 사용해 빠른 추론을 가능하게 합니다. 또한 '슬라이딩 윈도우 주의(Sliding Window Attention, SWA)'를 통해 임의의 길이의 시퀀스를 효과적으로 처리할 수 있으며, 추론 비용을 줄일 수 있습니다.

<br>

'Mistral 7B – Instruct'라는 이름의 모델도 있으며, 이는 지시사항을 따르도록 특별히 조정된 버전입니다. 이 버전은 인간과 자동화된 벤치마크 모두에서 'Llama 2 13B – chat model'을 능가한다고 합니다.

<br>

Mistral 7B 모델은 Apache 2.0 라이선스 하에 공개되었습니다.

<div id="Introduction"></div>

# Introduction

NLP 분야에서는 모델 성능을 높이기 위해 모델의 크기를 키우는 경향이 있습니다. 하지만, 이는 계산 비용과 추론 지연을 증가시켜 실제 환경에서의 배포를 어렵게 합니다. 따라서 고성능과 효율성을 모두 갖춘 모델 개발이 중요합니다.<br><br>

Mistral 7B는 이전 최고의 13B 모델(Llama 2)을 모든 벤치마크에서 능가하고, 34B 모델(LLaMa 34B)보다 수학과 코드 생성에서 우수한 성능을 보였습니다. 또한, 코드 관련 벤치마크에서는 Code-Llama 7B에 근접한 성능을 보였지만, 코드와 관련 없는 벤치마크에서의 성능을 포기하지 않았습니다. <br><br>

'그룹화된 쿼리 주의(Grouped-Query Attention, GQA)'와 '슬라이딩 윈도우 주의(Sliding Window Attention, SWA)' 메커니즘을 사용합니다. GQA는 추론 속도를 크게 향상시키고 디코딩 중 메모리 요구량을 줄여 실시간 애플리케이션에 중요한 높은 처리량을 가능하게 합니다. SWA는 더 긴 시퀀스를 더 효과적으로 처리하며 계산 비용을 줄이는 데 도움을 줍니다. <br><br>

참조 구현을 포함하여, 사용자가 로컬 환경이나 AWS, GCP, Azure와 같은 클라우드 플랫폼에서 쉽게 배포할 수 있도록 지원합니다. 이를 위해 'vLLM' 추론 서버와 'SkyPilot'이 사용됩니다. <br><br>

Hugging Face와의 통합이 간소화되어 있어, 더 쉽게 모델을 사용하고 통합할 수 있습니다. <br><br>

또한 다양한 작업에 대해 쉽게 fine-tuning을 할 수 있도록 설계되었습니다. <br><br>

<div id="Architectural details"></div>

# Architectural details

<div id="Sliding Window Attention"></div>

## Sliding Window Attention

<img style="width: 100%; margin-bottom: 40px;" id="output" src="./mistralImg/swa.PNG">

트랜스포머 모델의 주의 메커니즘을 개선한 방법으로, 긴 시퀀스의 처리 효율성을 높이기 위해 고안되었습니다. 주요 포인트는 다음과 같습니다.

1. 윈도우 크기 활용

SWA는 트랜스포머의 각 층에서 일정한 크기의 '윈도우'(W)를 사용하여 주변 정보에 주의를 기울입니다. 예를 들어, 윈도우 크기가 W인 경우, 특정 위치의  hidden state (hi)는 이전 층의 i-W부터 i까지에만 주의를 기울입니다.

2. Recursive information access: 모델의 각 층이 이전 층의 출력(은닉 상태)에 기반하여 정보를 처리하고, 이 과정이 여러 층에 걸쳐 반복되는 방식

각 층을 거치면서, hidden state는 입력 층의 토큰에 대한 정보를 최대 W × k 토큰 거리까지 접근할 수 있습니다. 여기서 k는 현재 층의 위치입니다.

3. 이론적 주의 범위

마지막 층에서 윈도우 크기 W = 4096을 사용하면, 이론적으로 약 131K 토큰에 대한 주의 범위를 가집니다

4. 실제 성능 개선

실제 시퀀스 길이가 16K이고 W = 4096인 경우, FlashAttention과 xFormers의 개선을 통해  vanilla attention baseline 대비 2배의 속도 향상을 얻을 수 있습니다.
