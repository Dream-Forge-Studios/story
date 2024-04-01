---
date: '2024-03-29'
title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks 논문 리뷰'
categories: ['LLM']
summary: 'RAG 완벽 이해하기.'
thumbnail: './test.png'
---

<div id="ABSTRACT"></div>

# ABSTRACT

Large pre-trained language models은 그들의 파라미터에 사실적인 지식을 저장할 수 있으며, 하위 NLP 작업에 미세 조정될 때 최첨단 성능을 달성할 수 있음이 보여졌습니다

<br>

그러나, 지식을 접근하고 정확하게 조작하는 능력은 여전히 제한적이며, 지식 집약적인 작업에서 그들의 성능은 작업 특정 아키텍처에 뒤처집니다.

<br>

또한, 그들의 결정에 대한 근거를 제공하고 세계 지식을 업데이트하는 것은 여전히 열린 연구 문제입니다.

<br>

Pretrained models with a differentiable access mechanism to explicit non-parametric memory(외부 지식)은 지금까지 추출형 downstream tasks에 대해서만 조사되었습니다.

<br>

우리는 retrieval-augmented generation (RAG) 모델을 위한 general-purpose fine-tuning recipe를 탐구합니다.

<br>

우리는 parametric memory가 사전 훈련된 seq2seq 모델이고, non-parametric memory가 pre-trained neural retriever로 접근되는 위키피디아의 dense vector index인 RAG 모델을 소개합니다.

<br>

우리는 생성 과정 전체에서 동일한 검색된 정보를 참조하는 RAG 형식과 토큰마다 다른 정보를 참조할 수 있는 다른 형식을 비교합니다.

<br>

wide range of knowledgeintensive NLP tasks로 우리의 모델을 fine tune하고 평가하며, 세 개의 오픈 도메인 QA 작업에서 최고의 성능을 달성하여 parametric seq2seq models과 task-specific retrieve-and-extract architectures를 능가합니다.

<div id="Introduction"></div>

# Introduction

Pre-trained neural language models은 외부 메모리에 접근하지 않고도, 모델 파라미터 내에 암시적인 지식 베이스로서 지식을 저장할 수 있습니다.

<br>

이러한 발전은 흥미롭지만, 이 모델들이 기억을 쉽게 확장하거나 수정할 수 없고, 예측에 대한 통찰을 직접적으로 제공하기 어렵고, 때로는 사실이 아닌 hallucinations를 생성할 수 있다는 단점이 있습니다.

<br>

이러한 문제를 해결하기 위해 parametric memory(모델 파라미터에 저장된 지식)와 non-parametric memory(검색 기반의 외부 지식)를 결합한 하이브리드 모델이 제안되었습니다.

<br>

이러한 모델들은 지식을 직접 수정하고 확장할 수 있으며, 접근한 지식을 검사하고 해석할 수 있다는 장점이 있습니다.

<br>

REALM과 ORQA 같은 최근에 소개된 모델들은 masked language models과 differentiable retriever를 결합하여, 오픈 도메인에서의 추출형 질문 답변에 대해 유망한 결과를 보였지만, 이러한 접근은 주로 해당 분야에 국한되어 탐색되었습니다.

<br>

본 논문에서는 hybrid parametric and non-parametric memory를 seq2seq 모델로 가져와서, 이 모델들이 직면한 일부 한계를 해결하고자 노력하였습니다.

<br>

RAG 모델은 두 주요 구성 요소, 즉 Dense Passage Retriever(DPR)라는 검색기와 BART라는 seq2seq 모델을 사용합니다.

이 구절은 검색-증강 생성(Retrieval-Augmented Generation, RAG) 모델의 작동 원리를 설명합니다. RAG 모델은 두 주요 구성 요소, 즉 Dense Passage Retriever(DPR)라는 검색기와 BART라는 seq2seq 모델을 사용합니다. 여기서 각 구성 요소의 역할과 상호 작용 방식을 자세히 살펴보겠습니다.

- Dense Passage Retriever (DPR)

  - DPR은 주어진 입력에 기반하여 관련 잠재 문서(또는 텍스트 조각)를 검색하는 역할, 이 과정에서 입력 쿼리에 가장 관련성이 높다고 판단되는 문서들을 선택
  - DPR은 입력(예: 질문)을 받아 데이터베이스(예: 위키피디아) 내에서 해당 질문에 대한 답변을 포함할 가능성이 높은 문서들을 찾아냄 (이를 위해 문서들은 벡터 형태로 인코딩되어 미리 계산된 밀집 벡터 인덱스에서 검색)

- BART (seq2seq 모델)
  - BART는 DPR로부터 검색된 잠재 문서와 원래의 입력 정보를 함께 고려하여 최종 출력(예: 답변, 번역 등)을 생성
  - BART는 전통적인 seq2seq 구조를 따르며 인코더-디코더 아키텍처를 사용하여 입력 시퀀스를 출력 시퀀스로 변환, 여기서 인코더는 입력과 검색된 문서를 결합된 컨텍스트로 처리하고 디코더는 이 컨텍스트를 바탕으로 관련 출력을 생성

잠재 문서를 top-K approximation를 통해 마진화(여러 잠재 문서들 중 어떤 문서들이 최종 출력을 생성하는 데 가장 큰 영향을 미쳤는지를 결정하는 과정)를 하는데, 이는 출력 기반(동일한 문서가 모든 토큰에 대해 책임 있다고 가정) 또는 토큰 기반(서로 다른 문서가 서로 다른 토큰에 대해 책임)에 따라 다릅니다.

<br>

T5 또는 BART와 같이, RAG는 어떤 seq2seq 작업에도 미세 조정될 수 있으며, 이 과정에서 생성기와 검색기가 함께 학습됩니다.

<br>

기존의 연구들은 memory networks, stackaugmented networks, memory layers 등과 같이 처음부터 훈련된 non-parametric memory를 사용하는 아키텍처를 제안했습니다.

<br>

그러나 본 논문에서는 pre-trained access mechanisms을 사용함으로써 추가적인 훈련 없이도 지식에 접근할 수 있습니다.

<br>

이 연구의 결과는 knowledge-intensive tasks에 parametric and non-parametric memory를 생성 과정과 결합하는 이점을 강조합니다.

<div id="Methods"></div>

# Methods

1. 검색기 (Retriever)

    - 입력 $x$에 대해 텍스트 패시지를 반환하는 검색기 $p_η(z∣x)$를 포함합니다. 여기서 $η$는 검색기의 파라미터를 나타냅니다. 이 검색기는 주어진 쿼리에 대해 상위 K개의 텍스트 패시지를 반환합니다.

2. 생성기 (Generator)

    - 검색된  text passages $z$, 이전의 토큰 $y_{1:i−1}$, 그리고 원본 입력 $x$를 바탕으로 현재 토큰 $y_i$를 생성하는 생성기 $p_θ(y_i∣x,z,y_{1:i−1})$를 포함합니다. 여기서 $θ$는 생성기의 파라미터입니다.

<div id="Models"></div>

## Models

RAG 모델은 검색된 문서를 잠재 변수로 취급하고, 이를 통해 생성기와 검색기를 end-to-end 학습할 수 있습니다. 이 모델은 잠재 문서를 다르게 marginalize하여 생성된 텍스트에 대한 분포를 생성하는 두 가지 접근 방식을 제안합니다:

- RAG-Sequence
  - 이 방식은 모든 타겟 토큰을 예측할 때 같은 문서를 사용
  - $P_{RAG-Sequence}(y|x)≈\sum_{z∈top-k(p(·|x))}p_η(z∣x)\prod_{i=1}^{N}p_θ(y_i∣x,z,y_{1:i−1})$
- RAG-Token
  - 이 방식은 각 타겟 토큰을 예측할 때 다른 문서를 사용
  - $P_{RAG-Sequence}(y|x)≈\prod_{i=1}^{N}\sum_{z∈top-k(p(·|x))}p_η(z∣x)p_θ(y_i∣x,z,y_{1:i−1})$

<div id="Retriever: DPR"></div>

## Retriever: DPR

Bi-encoder 구조를 사용합니다. 

<br>

Bi-encoder는 DPR은 문서($d(z)$)와 쿼리($q(x)$)를 각각 인코딩하는 두 개의 BERT 모델을 사용합니다. 이 두 인코더는 각각 문서 인코더($BERT_{d}$)와 쿼리 인코더($BERT_{q}$)로, 문서와 쿼리를 밀집 벡터로 변환합니다.

<br>

$p_η(z∣x) ∝ exp(d(z)^Tq(x))$

$d(z)=BERT_{d}(z), q(x)=BERT_{q}(x)$

<br>

RAG 모델의 검색기 DPR는 사전 훈련된 bi-encoder를 사용하여 초기화됩니다. 이 검색기는 TriviaQA와 Natural Questions 데이터셋에 대한 문서를 검색하도록 훈련되었습니다. 이러한 방식으로, RAG 모델은 높은 품질의 문서 검색 능력을 바로 사용할 수 있게 됩니다.

<div id="Generator: BAR"></div>

## Generator: BAR

본 논문에서는 생성기($p_θ(y_i∣x,z,y_{1:i−1})$)는 어떤 인코더-디코더 모델을 사용하여도 모델링될 수 있지만, 여기서는 400M 파라미터를 가진 사전 훈련된 seq2seq 변환기인 BART-large를 사용합니다.

<br>

BART 생성기의 파라미터 $θ$는 "parametric memory"로 언급됩니다. 이는 모델의 학습 가능한 파라미터를 의미하며, 모델이 어떻게 입력을 처리하고 적절한 출력을 생성할지 결정하는 데 중요한 역할을 합니다.

<div id="Training"></div>

## Training

이 부분은 RAG 모델의 검색기(retriever)와 생성기(generator) 컴포넌트를 함께 학습시키는 과정에 대해 설명하고 있습니다. 

<br>

이 과정은 어떤 문서가 검색되어야 하는지에 대한 직접적인 지도가 없이 진행됩니다. 학습의 목표는 입력/출력 쌍 ($x_j, y_j$)의 훈련 데이터셋을 사용하여, 각 타겟에 대한 negative marginal log-likelihood를 최소화하는 것입니다.

<br>

이는 모델이 주어진 입력 $x_j$에 대해 정확한 출력 $y_j$를 생성할 확률을 최대화하는 것을 목표로 합니다.

<br>

문서 인코더를 학습하는 것은 계산 비용이 많이 듭니다. 왜냐하면 문서 인덱스를 주기적으로 업데이트해야 하기 때문입니다.

<br>

그러나 이 연구에서는 문서 인코더와 인덱스를 고정시키고, 강력한 성능을 유지하면서 쿼리 인코더 $BERT_q$와 BART generator만 fine-tuning하는 것으로 충분하다고 발견했습니다.

<div id="Decoding"></div>

## Decoding

RAG-Sequence와 RAG-Token 모델은 각 모델은 주어진 입력 x에 대해 최적의 출력 y를 추정하는 과정에서 서로 다른 접근 방식을 사용합니다.

<br>

생성 과정에서 프롬프트(시작 텍스트 또는 초기 조건)는 미리 알려져 있습니다.

<br>

이 프롬프트를 이용하여 키(key)와 값(value) 캐시를 미리 채울 수 있습니다. 이러한 pre-fill은 모델이 초기 상태에 대한 정보를 빠르게 처리할 수 있게 해줍니다.

<br>

좀 더 쉽게 설명하면, 이전 정보를 처리할 때 이미 key와 value가 캐시에 저장되어 있어, 추가적인 key와 value를 구하는 계산없이 바로 활용할 수 있는 이점이 있습니다.

<br>

또한, 메모리가 아닌 캐시를 사용함으로써 메모리 사용량을 최적화하고, 긴 시퀀스를 처리할 때 발생할 수 있는 메모리 부담을 줄일 수 있습니다.

<br>

프롬프트가 매우 큰 경우, 이를 더 작은 조각(청크)으로 나누어 처리할 수 있습니다.

<br>

그림에서 보여주듯이,  attention mask는 캐시와 현재 청크에 대한 주의 계산을 적절히 조절합니다.
이 mask는 모델이 어떤 토큰에 주의를 기울여야 하는지를 결정하는 데 사용됩니다.

### 아래 링크로 들어가시면 자세한 코드 리뷰를 볼 수 있습니다.

[Mistral Transformer 코드에 대한 한국어 분석 및 주석 작성](https://github.com/thankyouflow/MistralKoreanAnalysis/tree/main)

<div id="Results"></div>

# Results

<img style="width: 100%; margin-bottom: 40px;" id="output" src="./mistralImg/result.PNG">
*MMLU (Massive Multitask Language Understanding): 다양한 주제와 분야에 걸쳐 광범위한 언어 이해 능력을 평가하는 벤치마크

*BBH (BIG-Bench Hard): 현재 언어 모델의 기능을 넘어서는 것으로 여겨지는 작업에 초점을 맞춘 다양한 평가

*Comprehension: 이해력 평가

<div id="Instruction Finetuning"></div>

# Instruction Finetuning

Mistral 7B 모델은 특별한 데이터나 교육 트릭 없이 instruction 데이터셋에 파인튜닝되었습니다.

<br>

이는 Mistral 7B 모델이 기본 상태에서 쉽게 파인튜닝되어 좋은 성능을 달성할 수 있음을 보여주는 간단하고 초기적인 시연입니다.

<br>

파인튜닝된 Mistral 7B – Instruct 모델은 MT-Bench에서 7B 모델들을 모두 능가하는 우수한 성능을 보였으며, 13B – Chat 모델과 비교할 수 있는 수준의 성능을 나타냈습니다.

<img style="width: 60%; margin-bottom: 40px; margin-top: 40px;" id="output" src="./mistralImg/instruct.PNG">
*Chatbot Arena ELO Rating: 각 챗봇의 성능은 다른 챗봇과의 상호작용을 통해 평가되며, 승리, 패배 또는 무승부에 따라 점수가 변동됩니다.

*MT Bench (Machine Translation Benchmark): 기계 번역(Machine Translation)의 성능을 평가하기 위한 벤치마크

<div id="Adding guardrails for front-facing applications"></div>

# Adding guardrails for front-facing applications

AI가 생성하는 내용에 대해 특정한 제약이나 규칙을 설정하는 것은 사용자와 직접 상호작용하는 응용 프로그램에서 매우 중요합니다.

<br>

이러한 규칙 또는 가드레일은 부적절한 내용을 필터링하고, 품질이 높은 콘텐츠를 보장하는 데 도움을 줍니다.

<br>

시스템 프롬프팅(system prompting)은 AI 모델에 특정한 출력 제약을 추가적으로 적용하는 방법입니다.

<br>

이를 통해 모델이 생성하는 내용에 대해 추가적인 통제를 할 수 있으며, 원하는 기준이나 가이드라인에 부합하는 출력을 생성하도록 할 수 있습니다.

<br>

Mistral 7B 모델은 세밀한 content moderation 기능을 가지고 있습니다.

<br>

이 기능은 응용 프로그램 내에서 품질이 높고 적절한 콘텐츠를 유지하는 데 사용될 수 있습니다.

<br>

content moderation과 guardrails을 통해 응용 프로그램은 사용자에게 보다 안전하고, 품질 높은 경험을 제공할 수 있습니다.

### System prompt to enforce guardrails

시스템 프롬프트는 모델이 안전하고 윤리적인 방식으로 답변을 생성하도록 유도하는 지침을 제공합니다.

<br>

이 프롬프트는 "항상 세심하고, 존중하며, 진실되게 도와줄 것"과 같은 지침을 포함하여 모델이 유용하면서도 안전한 답변을 제공하도록 합니다.

<br>

시스템 프롬프트를 사용함으로써 사용자는 모델의 유용성과 가드레일 집행 사이에서 균형을 찾을 수 있습니다.

<br>

이는 모델이 유용한 정보를 제공하는 동시에 해로운, 비윤리적, 편견적 또는 부정적인 콘텐츠를 피하도록 하는 것을 목표로 합니다.

<br>

175개의 위험한 프롬프트를 사용하여 모델의 안전성을 평가합니다.

<br>

추천된 시스템 프롬프트를 사용할 때, 모델은 해로운 질문에 대해 100% 거절하는 것으로 보입니다.

<br>

예시로, "리눅스 프로세스를 어떻게 죽일 수 있나요?"라는 질문에 대한 Mistral 7B – Instruct와 Llama 2 Chat 13B 모델의 답변이 제공됩니다.

<br>

Mistral 7B는 올바른 대답을 제공하는 반면, Llama 2는 답변을 거절합니다.

### Content moderation with self-reflection

Mistral 7B – Instruct은 content moderator로 사용될 수 있습니다. 

<br>

모델은 사용자의 프롬프트나 생성된 답변이 다음 범주 중 하나에 해당하는지를 판단합니다:

<br>

불법 활동(예: 테러리즘, 아동 학대, 사기)

증오, 괴롭힘, 폭력적인 내용(예: 차별, 자해, 괴롭힘)

비적격 조언(예: 법률, 의료, 재정 분야)

<br>

이를 위해 Mistral 7B가 사용자 프롬프트나 생성된 답변을 스스로 분류할 수 있도록 'self-reflection prompt'를 설계하였습니다.

<br>

이는 balanced dataset of adversarial and standard prompts를 포함하는 수작업으로 큐레이팅된 데이터셋에서 평가되고,  99.4%의 정밀도(precision)와 95.6%의 재현율(recall)을 달성했습니다.

<br>

이 기능은 소셜 미디어 댓글이나 포럼의 콘텐츠 모더레이션부터 인터넷상의 브랜드 모니터링에 이르기까지 다양하게 활용될 수 있습니다.

<br>

특히, 최종 사용자는 자신의 특정 사용 사례에 맞게 필터링할 카테고리를 선택할 수 있습니다.

<div id="Conclusion"></div>

# Conclusions

Mistral 7B 연구는 언어 모델이 기존에 생각했던 것보다 더 많은 지식을 압축할 수 있음을 입증합니다.
이는 언어 모델과 관련된 연구 분야에 새로운 관점을 제시합니다.

<br>

이 분야는 그동안 모델 능력과 훈련 비용 간의 관계에 중점을 두는 2차원적 접근법에 주목해왔습니다.

<br>

그러나 문제는 모델 능력, 훈련 비용, 추론 비용을 포함하는 3차원적인 것으로, 가장 작은 모델로 최고의 성능을 얻기 위한 탐색이 아직 많이 남아 있습니다.