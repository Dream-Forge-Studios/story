---
date: '2024-05-02'
title: 'SimCSE: Simple Contrastive Learning of Sentence Embeddings 논문 리뷰'
categories: ['LLM']
summary: 'LLM2Vec에서 사용된 SimCSE에 대해 자세하게 알아보자.'
thumbnail: './test.png'
---

<div id="Abstract"></div>

# Abstract

이 논문은 SimCSE라는 간단한 contrastive learning framework를 소개하며, 이를 통해 최고 수준의 문장 임베딩을 크게 향상시키는 방법을 제시합니다.

<br>

먼저, unsupervised 접근 방식에 대해 설명하는데, 이는 입력된 문장을 받아 standard dropout만을 노이즈로 사용하면서 contrastive 목표를 통해 스스로를 예측하도록 합니다.

<br>

이 간단한 방법은 놀랍도록 잘 작동하여 이전의 supervised 방법들과 맞먹는 성능을 보여줍니다. 연구진은 dropout이 최소한의 data augmentation 역할을 하며, 이를 제거하면 representation collapse된다는 것을 발견했습니다.

<br>

이어서, supervised 접근 방식을 제안하는데, 여기서는 자연어 추론 데이터셋의 주석이 달린 쌍을 사용하여 contrastive learning 프레임워크에 통합합니다.

<br>

특히 "entailment" 쌍은 긍정적 예로, "contradiction" 쌍은 부정적 예로 사용됩니다.

<br>

SimCSE는 표준 의미적 텍스트 유사성(STS) 작업에서 평가되며, BERTbase를 사용하는 unsupervised 및 supervised 모델은 각각 평균 76.3% 및 81.6%의 Spearman’s correlation를 달성하여 이전 최고 결과보다 각각 4.2% 및 2.2% 향상되었습니다.

<br>

*Spearman’s correlation: 모델이 생성한 문장 임베딩 간의 유사성과 사람이 평가한 문장 유사성 사이의 순위 일치도

<br>

또한, contrastive learning 목표가 pre-trained embeddings’ anisotropic space을 더 균일하게 정규화하며, supervised signals이 제공될 때 긍정적인 쌍을 더 잘 정렬한다는 것을 이론적으로 및 경험적으로 보여줍니다.

<div id="Introduction"></div>

# Introduction

이 연구에서는 최신 문장 임베딩 방법을 발전시키고, 사전 훈련된 언어 모델들(BERT, RoBERTa 등)과 결합될 때 contrastive 목표가 매우 효과적일 수 있음을 보여줍니다.

<br>

연구팀은 SimCSE라는 간단한 contrastive 문장 임베딩 프레임워크를 제시하며, 이를 통해 레이블이 없거나 있는 데이터에서 우수한 문장 임베딩을 생성할 수 있습니다.

<br>

unsupervised SimCSE는 입력된 문장 자체를 예측하는 방식으로, 노이즈로서 드롭아웃만을 사용합니다.

<br>

구체적으로, 동일한 문장을 사전 훈련된 인코더에 두 번 전달하여, 표준 드롭아웃을 두 번 적용함으로써 “긍정적 쌍”으로서 서로 다른 두 개의 임베딩을 얻습니다.

<br>

이러한 접근 방식은 매우 간단해 보일 수 있지만, 다음 문장 예측이나 단어 삭제 및 교체와 같은  discrete data augmentation(이산 데이터 증강 방법)을 사용하는 훈련 결과들을 크게 앞서며, 이전의 감독된 방법들과도 맞먹는 성능을 보여줍니다.

<br>

신중한 분석을 통해, dropout이 숨겨진 표현의 최소한의 “data augmentation” 역할을 하며 이를 제거하면 representation collapse로 이어진다는 것을 발견했습니다.

<br>

NLI 데이터셋은 문장 쌍과 그 사이의 관계(예: 함축(entailment), 중립(neutral), 모순(contradiction))에 대한 주석이 포함되어 있습니다.

[한국 NLI 데이터셋]('https://huggingface.co/datasets/kor_nli)
- 함축 (Entailment):
  - 함축 관계는 첫 번째 문장(전제)이 참일 때, 두 번째 문장(결론)이 반드시 참이 되어야 하는 관계를 의미합니다. 
  - 즉, 전제가 결론을 논리적으로 지지하는 경우입니다. 예를 들어, 전제 "그녀는 요리를 하고 있다"가 주어졌을 때, 결론 "그녀는 부엌에 있다"는 함축됩니다.
- 중립 (Neutral):
  - 중립 관계는 전제에서 제시된 정보만으로는 결론이 참인지 거짓인지를 판단할 수 없을 때를 나타냅니다. 
  - 전제와 결론 사이에 명확한 논리적 관계가 존재하지 않습니다. 예를 들어, 전제 "그녀는 요리를 하고 있다"에 대해 결론 "그녀는 음악을 좋아한다"는 중립적 관계에 있습니다.
- 모순 (Contradiction):
  - 모순 관계는 전제가 참일 때 결론이 거짓이 되어야 하는 경우입니다. 
  - 즉, 전제와 결론 사이에 충돌이 있으며, 두 문장이 서로 배치됩니다. 예를 들어, 전제 "그녀는 요리를 하고 있다"가 주어졌을 때, 결론 "그녀는 집 밖에 있다"는 모순됩니다.

SimCSE는 함축 관계에 있는 문장 쌍을 긍정적인 예로 사용합니다. 이는 두 문장이 서로 관련이 있음을 의미하며, 이를 통해 문장 임베딩이 더 정확하게 유사한 의미를 가진 문장을 인식할 수 있게 합니다.

<br>

또한, 모순 관계에 있는 문장 쌍을 hard negatives로 추가함으로써 모델의 성능을 더욱 향상시킵니다.

<br>

이러한 간단한 사용법은 동일한 데이터셋을 사용한 이전 방법들에 비해 상당한 개선을 이루었습니다.

<br>

이 연구는 또한 다른 레이블이 붙은 문장 쌍 데이터셋과의 비교를 통해 NLI 데이터셋이 문장 임베딩 학습에 특히 효과적임을 발견했습니다.

<br>

이는 NLI 데이터셋이 문장 간의 의미적 관계를 다루는 데 특히 유용하기 때문에 문장 임베딩에 효과적인 학습 자료로 작용합니다.

<br>

Wang과 Isola (2020)의 분석 도구를 차용하여, 문장 임베딩의 품질을 측정하기 위해 의미론적으로 관련된 긍정적인 쌍 간의 alignment과 전체 표현 공간의 uniformity을 평가합니다.

<br>

분석 결과, unsupervised SimCSE는 dropout 노이즈를 통해 잘못된 alignment을 피하면서 uniformity을 본질적으로 향상시키는 것으로 나타났습니다.

<br>

이는 임베딩의 표현력을 개선하는 데 기여합니다. 또한, 같은 분석을 통해 NLI 훈련 신호가 긍정적인 쌍 간의 alignment을 더욱 개선하고 더 나은 문장 임베딩을 생성할 수 있음을 보여줍니다.

<br>

이 연구는 또한 사전 훈련된 단어 임베딩이 anisotropy(임베딩 공간에서 단어 벡터들이 특정 방향에 집중) 문제를 겪고 있다는 최근의 발견(Ethayarajh, 2019; Li et al., 2020)과 연결지어 설명합니다.

<br>

contrastive learning objectiv가 문장 임베딩 공간의 특이값 분포를 "평탄화"함으로써 균일성을 개선한다는 것을 스펙트럼 관점에서 증명합니다.

<div id="Background: Contrastive Learning"></div>

# Background: Contrastive Learning

contrastive learning은 의미적으로 가까운 이웃(neighbor)을 서로 가깝게 끌어당기고, 그렇지 않은 비이웃(non-neighbors)은 밀어내는 방식으로 효과적인 표현을 학습하는 방법입니다.

<br>

이 접근법은 서로 의미적으로 관련 있는 쌍의 예제 집합 $D={(x_i,x_i^+)}_{i=1}^m$로 정의합니다.

<br>

contrastive framework는 Chen et al. (2020)을 따르며, 배치 내의 부정적 예제(in-batch negatives)와 함께 cross-entropy objective를 사용합니다.

<br>

여기서 $h_i$ 와 $h_i^+$는 각각 $x_i$와 $x_i^+$의 표현을 나타냅니다. 훈련 목표는 미니배치에 있는 N 쌍을 사용하여 다음과 같이 정의됩니다:

<br>

<img style="width: 50%; margin-right: 0px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;" id="output" src="SimCSE/sic1.PNG">

τ는 temperature hyperparameter

$sim(h^1,h^2)$는 두 표현의 cosine similarity
### Long-context LLMs

일반적으로 LLM은 사전 정의된 문맥 길이로 사전 훈련됩니다. 예를 들어, LLaMA는 2048 토큰, Llama2는 4096 토큰으로 설정됩니다.

<br>

하지만 처음부터 긴 문맥을 가진 LLM을 훈련하는 것은 대부분의 연구자들에게는 비용이 너무 많이 들어 실행하기 어렵습니다. 최근 몇몇 연구에서는 파인튜닝을 통해 LLM의 문맥 길이를 확장하려고 시도했습니다.

<br>

예를 들어, Position Interpolation (Chen et al., 2023)은 로터리 위치 인코딩을 수정하여 LLaMA의 문맥 길이를 32768까지 확장합니다. Focused Transformer (Tworkowski et al., 2023)는 contrastive learning을 사용하여 LongLLaMA를 훈련합니다.

<br>

이러한 방법들은 모두 완전한 파인튜닝에 의존하며, 이는 많은 계산 자원을 요구합니다 (예: 128 A100 GPUs / 128 TPUv3).

<br>

Landmark attention (Mohtashami & Jaggi, 2023)은 효율적인 접근법이지만, 약간의 손실을 동반합니다. 이 방법은 긴 문맥 입력을 retrieved tokens으로 압축합니다.

<br>

이 방법은 긴 문맥 입력을 검색된 토큰으로 압축합니다. 우리의 방법은 파인튜닝 비용을 상당히 절감하면서도 원래의 주의 품질을 유지합니다. 추론 시에는 수정되지 않은 주의를 통해 전체 입력에 완전히 접근합니다.

<div id="LONGLORA"></div>

# LONGLORA

<div id="SHIFTED SPARSE ATTENTION"></div>

## SHIFTED SPARSE ATTENTION

### Pilot Study

첫 번째 시도에서는 figure2의 pattern 1 short attention만을 사용하여 모델을 훈련시킵니다. 

<br>

이는 주로 긴 문맥에서 높은 계산 비용이 self-attention으로부터 발생하기 때문에, 긴 입력을 여러 그룹으로 나누어 각 그룹에서 self-attention를 수행합니다.

<br>

예를 들어, 모델은 훈련 및 테스트 단계에서 8192개의 토큰을 입력으로 받지만, 각 그룹의 크기는 2048로 self-attention이 수행됩니다. 그룹 수는 4개입니다.

<br>

이러한 분할 방식은 효율적이지만 매우 긴 문맥에서는 효과가 없으며, 문맥 길이가 길어질수록 perplexity가 증가합니다.

<br>

이는 다른 그룹 간에 정보 교환이 없기 때문입니다.

<br>

그룹 간 통신을 도입하기 위해 그룹 분할을 반 그룹 크기만큼 이동시키는 figure2의 pattern 2를 사용합니다.

<br>

예를 들어, 전체 문맥 길이가 8192인 경우, 첫 번째 그룹은 1번째부터 2048번째 토큰까지 self attention을 수행하고, 두 번째 패턴에서는 그룹 분할이 1024만큼 이동하여, 첫 번째 주의 그룹이 1025번째부터 3072번째 토큰까지 시작합니다.

<br>

패턴 1과 2를 각각 half self-attention heads에서 사용합니다. 이 방식은 추가적인 계산 비용을 증가시키지 않으면서 다른 그룹 간의 정보 흐름을 가능하게 합니다.

<br>

이러한 접근 방식은 standard attention baseline과 가까운 성능을 보여줍니다.

### Consistency to Full Attention

많은 효율적인 주의 설계들이 긴 문맥의 LLM의 효율을 개선할 수 있지만, 대부분은 긴 문맥의 파인튜닝에 적합하지 않습니다.

<br>

이는 이러한 변형된 트랜스포머들이 처음부터 훈련되도록 설계되었기 때문에, 사전 훈련에 사용된 standard full attention와 차이가 있기 때문입니다.

<br>

$S^2 -Attn$은 fine-tuning뿐만 아니라 full attention testing도 지원합니다. 이는 fine-tuning 과정에서 사용된 attention mechanism을 테스트 과정에서도 그대로 사용해야 한다는 의미입니다.

<br>

또한, $S^2 -Attn$은 특정 주의 패턴에 모델이 over-fitted을 방지하는 specific attention patterns을 포함합니다.

### Easy Implementation

$S^2 -Attn$는 구현하기 쉽습니다.

1. half attention heads에서 shifting tokens
2. 토큰 차원에서 배치 차원으로 특성을 transposing합니다.
이를 위한 코드는 두 줄로 충분합니다.

```
# B: batch size; S: sequence length or number of tokens; G: group size;
# H: number of attention heads; D: dimension of each attention head
# qkv in shape (B, N, 3, H, D), projected queries, keys, and values
# Key line 1: split qkv on H into 2 chunks, and shift G/2 on N
qkv = cat((qkv.chunk(2, 3)[0], qkv.chunk(2, 3)[1].roll(-G/2, 1)), 3).view(B*N/G,G,3,H,D)
# standard self-attention function
out = self_attn(qkv)
# out in shape (B, N, H, D)
# Key line 2: split out on H into 2 chunks, and then roll back G/2 on N
out = cat((out.chunk(2, 2)[0], out.chunk(2, 2)[1].roll(G/2, 1)), 2)
```

<div id="IMPROVED LORA FOR LONG CONTEXT"></div>

## IMPROVED LORA FOR LONG CONTEXT

LoRA 방식은 파라미터의 수를 줄이면서도 효과적으로 모델을 적응시킬 수 있지만, 타겟 문맥 길이가 길어질수록 전체 파인튜닝과의 성능 격차가 커집니다.

<br>

또한, LoRA에서 rank를 높이는 것만으로는 이 격차를 줄이기 어렵다는 점이 실험적으로 관찰되었습니다.

<br>

이 격차를 해소하기 위해, 연구에서는 임베딩과 정규화 레이어를 훈련에 포함시키기로 결정했습니다.

<br>

특히 정규화 레이어는 전체 모델에서 매우 적은 비율의 파라미터(0.004%)를 차지하지만, 긴 문맥 적응에 있어 중요한 영향을 미칩니다. 이러한 접근을 통해 LoRA의 개선된 버전인 $LoRA^+$가 실험에서 사용되었습니다.

<div id="EXPERIMENT"></div>

# EXPERIMENT

<div id="EXPERIMENTAL SETTINGS"></div>

## EXPERIMENTAL SETTINGS

### Models

7B, 13B, 70B 크기의 Llama2 모델이 사용되었습니다.

<br>

최대 문맥 윈도우 크기:

- 7B 모델의 경우 최대 100,000
- 13B 모델의 경우 최대 65,536
- 70B 모델의 경우 최대 32,768

<br>

모든 모델의 위치 인덱스는 'Position Interpolation' 기법을 사용하여 재조정되었습니다.

### Position Interpolation

- 옵티마이저
  - AdamW 옵티마이저가 사용되며, β1 = 0.9, β2 = 0.95의 값을 가집니다.

- 학습률
  - 7B 및 13B 모델: 2 × 10−5
  - 70B 모델: 10−5

- linear learning rate warmup
- weight decay: 0
- per-device batch size: 1, gradient accumulation steps: 8 <br> (8개의 GPU를 사용할 때 global batch size 64)
- train 1000 steps

### Datasets

- 학습 데이터셋: Redpajama
- 평가 데이터셋
  - PG19 (Rae et al., 2020): 이 데이터셋은 책 코퍼스로 구성되어 있으며, 긴 시퀀스 언어 모델링 성능 평가에 사용됩니다. 평가에는 PG19의 테스트 분할, 즉 100개의 문서가 포함된 부분이 사용됩니다.
  - Proof-pile 데이터셋 (Azerbayev et al., 2022): 이는 정리된 Arxiv 수학 증명 자료로 구성된 데이터셋입니다. 이 데이터셋의 테스트 분할 역시 모델 평가에 사용됩니다.
- 성능 평가 방식: perplexity
  - 'Press et al., 2022'의 방법을 따라 슬라이딩 윈도우 접근법을 사용합니다. 이 방법에서 윈도우 크기(S)는 256으로 설정됩니다. 
  - perplexity는 모델이 데이터를 얼마나 잘 이해하고 있는지를 측정하는 지표로, 낮을수록 모델의 성능이 더 좋다고 평가됩니다.

<div id="MAIN RESULTS"></div>

## MAIN RESULTS

### Long-sequence Language Modeling

특정 훈련 문맥 길이에서, 모델은 더 긴 문맥 크기에서 더 나은 perplexity를 달성합니다. 이는 효율적인 파인튜닝 방법의 효과를 나타냅니다.

<br>

예를 들어, Llama2 7B 모델의 문맥 창 크기를 8192에서 32768로 증가시킬 때 perplexity는 2.72에서 2.50으로 개선됩니다(-0.22의 변화). Llama2 13B 모델의 경우, perplexity가 -0.28만큼 감소합니다.

<br>

8 × A100 머신에서 파인튜닝할 수 있는 최대 문맥 길이를 추가로 검토합니다. Llama2 7B, 13B, 70B 모델을 각각 100k, 65536, 32768 문맥 길이로 확장합니다.

<br>

<img style="width: 100%; margin-right: 0px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;" id="output" src="longLora/table4.PNG">

<br>

확장된 모델들에서 작은 문맥 크기에 대한 perplexity 저하가 관찰됩니다. 이는 Position Interpolation 방법의 알려진 한계점입니다.

### Retrieval-based Evaluation

이 연구의 주요 목적은 긴 대화에서 목표 주제를 검색하는 것입니다. 이러한 대화의 길이는 3k에서 16k 토큰에 이르며, 일부 질문은 16k를 초과합니다.

<br>

이 연구에서는 Llama2 13B 모델을 다른 open LLMs와 비교하여 평가합니다. 비교 대상 중 하나는 LongChat-13B 모델로, 이는 같은 태스크에서 최고의 성능을 보이는 모델입니다.

<br>

Llama2 13B는 18k 문맥 길이로 파인튜닝되었으며, 훈련 비용은 16k 토큰을 학습할 때와 비슷합니다.

<br>

실험 결과, 이 모델은 LongChat-13B와 비교할 만한 성능을 보였으며, 16k 평가에서는 약간 더 우수한 성능을 보였습니다.

<br>

<img style="width: 100%; margin-right: 0px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;" id="output" src="longLora/figure4.PNG">

<br>

이 태스크는 문서에서 임의의 'passkey'를 찾는 것으로, Landmark Attention 기법을 사용하여 수행됩니다.

<br>

Llama2 7B와 LongLoRA 모델이 32768 문맥 길이로 파인튜닝되어 1k부터 34k까지의 문서 길이에서 passkey 검색 정확도를 테스트합니다.

<br>

모델은 33k 또는 34k까지 합리적인 passkey 검색 정확도를 보여줍니다.

<br>

최대 위치 인코딩을 48k로 확장하여, 모델이 더 긴 문서를 처리할 수 있도록 개선되었습니다.

<br>

그러나 Llama2 7B는 4k 문맥 길이 이후에 정확도가 급격히 떨어지는 문제가 있습니다.

<div id="ABLATION STUDY"></div>

## ABLATION STUDY

### Ablation on Fine-tuning Steps

<img style="width: 100%; margin-right: 0px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;" id="output" src="longLora/figure5.PNG">

<br>

파인튜닝을 하지 않은 상태(0 단계)에서 모델은 긴 문맥에서 제한적인 능력을 보여주며, perplexity는 15.82입니다. 이는 모델이 아직 최적화되지 않았을 때 긴 문맥을 어떻게 처리하는지를 나타냅니다.

<br>

파인튜닝이 진행됨에 따라 perplexity가 빠르게 감소하는 것을 관찰할 수 있습니다. 이는 모델이 점차 긴 문맥을 더 잘 처리하게 되고, 언어 모델의 성능이 개선됨을 의미합니다.

<br>

Full fine-tuning은 low-rank training보다 더 빠르게 수렴합니다. 이는 Full fine-tuning이 모델의 모든 파라미터를 조정하면서 보다 효과적으로 모델을 최적화한다는 것을 보여줍니다.

<br>

200 steps 이후, Full fine-tuning과 low-rank training 간의 성능 격차는 크지 않습니다. 이는 초기에는 Full fine-tuning이 유리하지만, 장기적으로는 두 방법이 비슷한 성능을 보여준다는 것을 의미합니다.

### Attention Patterns

Llama2 7B 모델을 다양한 attention patterns을 사용하여 파인튜닝하고 그 효과를 평가한 실험에 대해 설명하고 있습니다. 이 모델은 Redpajama 데이터셋을 기반으로 32768 문맥 길이로 파인튜닝되었으며, PG19 검증 세트를 사용하여 perplexity를 평가합니다.

<br>

LongLoRA에서 사용된 Shift 작업에는 세 가지 옵션이 있습니다:

1.  disabling it
2. shifting between sequential layers
3. shifting among attention heads

실험 결과, 두번째 방법은 괜찮은 성능을 보이지만 최선의 방법은 아니라고 합니다. 즉, 어느 정도 효과는 있지만, 다른 방법들에 비해 상대적으로 뛰어나지는 않습니다.

<br>

또한 데이터의 위치를 왼쪽으로 옮기나 오른쪽으로 옮기나 성능 차이가 거의 없었다고 합니다. 즉, 이러한 위치 변경이 모델의 전반적인 성능에 큰 영향을 주지 않는다는 것을 발견했습니다.
