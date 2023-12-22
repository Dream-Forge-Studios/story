---
date: '2023-12-21'
title: 'Precedent-Enhanced Legal Judgment Prediction with LLM and Domain-Model Collaboration 논문 리뷰'
categories: ['Large Language', 'RAG', 'Legal']
summary: 'Precedent-Enhanced Legal Judgment Prediction with LLM and Domain-Model Collaboration 완벽 이해하기.'
thumbnail: './test.png'
---

<div id="Introduction"></div>

## Introduction

법률 인공지능(Legal AI)은 수십 년 동안 다양한 법률 작업을 지원하기 위해 연구되어 왔습니다. 이러한 작업에는 법률 질의응답(Legal QA), 법정 의견 생성, 법률 실체 인식 등이 포함됩니다. 

<br>

이러한 작업 중에서도 가장 중요한 것 중 하나는 법적 판결 예측(Legal Judgment Prediction, LJP)입니다. LJP의 목적은 사건 사실 설명을 기반으로 해당 사건의 법적 판결을 예측하는 것입니다. 법적 판결에는 일반적으로 법률 조항, 혐의, 그리고 징역 기간이 포함됩니다.

<br>

선례는 유사한 사실을 가진 이전 사건들을 의미하며, 법적 판결 예측에서 중요하게 고려됩니다.

<br>

LJP 작업에 사용되는 기술은 크게 LLMs와 도메인 특화 모델로 나눌 수 있습니다.

<br>

LLMs는 광범위한 훈련을 통해 복잡한 자연어를 이해하고 생성하는 데 능숙하며, 맥락 내 학습에서도 강점을 가집니다. 반면에, 도메인 특화 모델은 특정 작업에 맞춰 설계되어 비용 효율적인 해결책을 제공합니다.

<br>

하지만 LLMs는 프롬프트 길이에 제한되어 있어 다수의 추상적 레이블의 의미를 파악하고 적절한 것을 선택하는 데 어려움을 겪으며, 도메인 모델의 경우 선례와 주어진 사례 사이의 유사성과 차이점을 이해하고 구별하는 능력이 제한적입니다.

<br>

이 연구에서는 LLMs와 도메인 특화 모델을 협력하여 새로운 선례 강화 법적 판결 예측 프레임워크(Precedent-Enhanced Legal Judgment Prediction, PLJP)를 제안합니다.

<br>

구체적으로, <u>도메인 모델은 후보 레이블을 제공하고 사례 데이터베이스에서 적절한 선례를 효과적으로 찾는 역할</u>을 하며, <u>LLMs는 맥락 내 선례 이해를 통해 최종 예측을 결정</u>합니다.

<img style="width: 60%; margin-top: 40px;" id="output" src="precedent/process.PNG">

### 연구 방법

- 이전의 LJP 연구들(Zhong et al., 2018; Yue et al., 2021; Dong and Niu, 2021)을 따라 공개된 실제 법률 데이터셋에서 실험을 수행

    <br>

- 테스트 데이터는  2022년 이후에 발생한 사건들로 구성

    <br>
  
- 원래 테스트 세트와 추가 테스트 세트에서 모두 최첨단(SOTA) 성능을 달성

<div id="Related Work"></div>

## Related Work

**Legal AI**

- 법률 질의응답(Legal Question Answering, QA): 법률 문제에 대한 질문에 답변하는 시스템을 개발하는 것입니다 (Monroy et al., 2009).

    <br>

- 법률 실체 인식(Legal Entity Recognition): 법률 문서에서 중요한 실체나 용어를 자동으로 식별하는 기술입니다 (Cardellino et al., 2017).

    <br>

- 법정 의견 생성(Court View Generation): 법정에서의 의견이나 판결문을 생성하는 기술입니다 (Wu et al., 2020).

    <br>

- 법률 요약(Legal Summarization): 법률 문서의 주요 내용을 간결하게 요약하는 기술입니다 (Hachey and Grover, 2006; Bhattacharya et al., 2019).

    <br>

- 법률 언어 이해(Legal Language Understanding): 법률적 언어와 문서의 의미를 이해하고 분석하는 기술입니다 (Chalkidis et al., 2022).

<div id="Problem Formulation"></div>

## Problem Formulation

**1. 사실 설명(Fact Description)**

- 사건의 간결한 서술로, 일반적으로 사건의 시간 순서, 각 당사자의 행동이나 행위, 그리고 사건과 관련된 기타 필수적인 세부사항을 포함합니다.

  <br>
  
- token sequence로 정의되며, $f=\left\{w_{t}^{f}\right\}_{t=1}^{l_f}$
  - $f$: 이는 토큰 시퀀스 전체를 나타내는 변수입니다. 일반적으로, 이는 문장이나 문서 등의 텍스트 데이터를 나타냅니다.
  
    <br>
    
  - $w_{t}^{f}$: 이는 시퀀스 $f$내의 개별 토큰을 나타냅니다. 여기서 $t$는 시퀀스 내의 특정 위치를 나타내는 인덱스이며, $w_{t}^{f}$는 그 위치에 있는 토큰입니다.

    <br>

  - $\left\{w_{t}^{f}\right\}_{t=1}^{l_f}$: 이 표현은 토큰 시퀀스의 전체 범위를 나타냅니다. 여기서 $t=1$은 시퀀스의 시작을, $l_f$는 시퀀스의 끝을 나타냅니다. 즉, 이는 첫 번째 토큰부터 $l_f$번째 토큰까지 모든 토큰을 포함합니다.

    <br>
  
  - $l_f$: 이는 토큰 시퀀스 $f$의 전체 길이를 나타내는 변수입니다. 즉, 시퀀스에 있는 토큰의 총 개수입니다.

**2. 판결(Judgment)**

- 판사가 법률 사건에 대해 사실과 선례를 바탕으로 내린 최종 결정입니다. 일반적으로 법률 조항, 혐의, 그리고 징역 기간으로 구성됩니다.
  
    <br>
  
- 사건의 판결은 $j = (a, c, t)$로 표현되며, 여기서 $a, c, t$는 각각 조항, 혐의, 징역 기간의 레이블을 나타냅니다.

<br>

**3. 선례(Precedent)**

- 유사한 사실을 가진 이전 사건입니다. 선례의 판결은 현재 사건에 대한 중요한 참고자료입니다.

    <br>

- 선례는 $p = (f_p, j_p)$로 정의되며, 여기서 $f_p$는 그 사실 설명이고, $j_p$는 그 판결입니다. 주어진 사건에 대해 여러 선례가 있을 수 있으며, 이는 $P = {p_1, p_2, ..., p_n}$으로 표시될 수 있으며, 여기서 $n$은 선례의 수입니다.

<br>

**법적 판결 예측**

<br>

사실 설명 $f$가 주어졌을 때, 우리의 작업은 선례 $P$를 얻고 이해한 다음, 판결 $j = (a, c, t)$를 예측하는 것입니다.

<div id="Precedent-Enhanced LJP (PLJP)"></div>

## Precedent-Enhanced LJP (PLJP)

### Case Database Construction

선례를 활용하기 전에, 다수의 이전 사건들을 수집하여 케이스 데이터베이스를 구축해야 합니다.

<br>

사건의 사실 설명이 일반적으로 길고 상세하기 때문에, 모델이 적절한 선례를 찾는 것은 어려운 작업입니다. 이를 해결하기 위해, 대규모 언어 모델(LLMs)의 도움을 받아 이전 사건들의 사실 설명을 재구성합니다.

<br>

**Fact Reorganization**

<br>

이 부분의 목표는 사건의 사실 설명을 요약하고 재구성하는 것입니다. 이 과정은 세 가지 관점에서 진행됩니다. 사실 재구성 과정은 인간의 주석 없이 LLMs을 사용하여 완료됩니다. 구체적인 prompt는 아래와 같습니다.

<br>

"사실 설명은 주관적 동기, 객관적 행동, 그리고 사후 사정으로 분류될 수 있습니다. 

<br>

주관적 동기는 가해자의 해로운 행동과 그 결과에 대한 심리적 태도를 의미하며, 이에는 범죄의 의도, 부주의, 그리고 목적 등이 포함됩니다. 

<br>

객관적 행동은 관찰 가능한 활동 측면에서 범죄를 구성하는 필수 조건에 해당하며, 여기에는 해로운 행동, 해로운 결과, 그리고 행동과 결과 사이의 인과 관계가 포함됩니다. 사후 사정은 처벌의 중증도를 결정할 때 고려되는 다양한 사실적 상황을 의미합니다. 

<br>

경감 사정으로는 자수와 공로 행위가 있으며, 가중 사정으로는 재범 등이 있습니다. 

<br>

제공된 정보에 기반하여, 다음 사실들을 요약하십시오."

<br>

재구성 후, 사실 설명 $f$는 삼중 항목($sub, obj, ex$)으로 변환됩니다. 여기서 $sub$는 주관적 동기, $obj$는 객관적 행동, $ex$는 사후 사정을 나타냅니다. 

<br>

마지막으로, 케이스 데이터베이스의 이전 사례는 재구성된 사실과 판결의 쌍으로 저장됩니다.

<img style="width: 100%; margin-top: 40px;" id="output" src="precedent/fact.PNG">

실선은 선행 검색 과정이고, 점선은 예측 과정을 나타냅니다.

### Legal Judgment Prediction

도메인 모델은 특정 데이터셋에 대해 훈련되어 특정 작업을 해결하는 데 목표를 두고 있으며, 여기서는 두 종류의 도메인 모델, 즉 예측 모델과 검색 모델을 사용합니다

<br>

**Predictive model**

- 예측 모델은 사실 설명을 입력으로 받아 세 개의 하위 작업(법률 조항, 혐의, 징역 기간)에 대한 후보 레이블을 출력합니다.

  <br>
  
- 사실 설명 $\left\{w_{t}^{f}\right\}_{t=1}^{l_f}$은 단어 시퀀스이므로, 먼저 인코더를 사용하여 임베딩 시퀀스 $H^f ∈ \mathbb{R}^{l_f ×d}$로 변환합니다. 여기서 $H^f = h_{1}^{f} , h_{2}^{f}  , ..., h_{l_f}^{f}$이고, $d$는 임베딩의 차원입니다.<br>: $H^f = Encode(f)$

  <br>

- 임베딩 시퀀스에서 Max Pooling 연산을 수행하여 가장 중요한 특성 벡터 $h^f ∈ \mathbb{R}^d$를 얻습니다.<br>: $h^f = MaxPooling(H^f)$

  <br>

- 이 벡터를 fully-connected network with softmax activation에 입력하여 레이블 확률 분포 $P ∈ R^m$를 얻습니다. 여기서 $W^p ∈ \mathbb{R}^{m×d}$와 $b^p ∈ \mathbb{R}^m$은 학습 가능한 매개변수입니다. 주의할 점은 $m$은 다른 하위 작업에서 달라질 수 있습니다.<br>: $P = Softmax(Wp · h^f + b^p )$

  <br>

- 이제 모델은 확률이 가장 높은 상위 $n$개의 후보 레이블로 선택합니다.

<br>

**Retrieval Model**

- 검색 모델은 주어진 사건의 재구성된 사실(주관적 동기, 객관적 행동, 사후 사정 - $sub$, $obj$, $ex$)을 기반으로 적절한 선례를 찾습니다.

  <br>

- $D_1$(재구성된 사실)과 $D_2$(케이스 데이터베이스)의 유사성 점수를 구하기 위해, 먼저 같은 인코더를 사용하여 각각을 독립적으로 인코딩합니다.<br>: $h_{D_1} = Encoder ( D_1 )$, $h_{D_2}=Encoder(D_2)$
  
  <br>

- $h_{D_1}∈R^{d′}$과 $h_{D_2}∈R^{d′}$는 각각의 임베딩을 나타내며, d′는 이들 임베딩의 차원을 나타냅니다.

  <br>

- similarity score $s(D_1, D_2)$는 cosine similarity로 계산됩니다.<br>: $s(D_1, D_2)= \frac{h_{D_1}\cdot h_{D_2}}{∥hD1 ∥ ∥hD2 ∥}$

  <br>

- 이 과정을 통해, 검색 모델은 주어진 사건과 유사한 선례들 사이의 유사성을 평가하고, 관련성 높은 선례( $P=\left\{ p_1,p_2,...,p_n \right\}$ )를 찾습니다. 이는 레이블의 보충적 설명으로 활용됩니다.

<div id="LLMs"></div>

## LLMs

이전의 설명했던 것 처럼 LLMs를 활용하여 Fact Reorganization(주관적 동기, 객관적 행동, 사후 사정) 합니다.
  
<br>

LLMs는 복잡한 자연어를 이해할 수 있는 능력을 가지고 있기 때문에, 주어진 사건과 그 선례들을 결합하고, 맥락 내 선례 이해를 통해 최종 판결을 내립니다.

<br>

구체적인 prompt:

<br>

"사실에 기반하여, 우리는 도메인 모델에 의해 후보 법률 조항들을 선택하고, 후보 법률 조항에 기반하여 다음 세 가지 선례들을 선택했습니다. 선례들 간의 차이를 이해한 후, 이 사건의 사실과 비교하여 최종 레이블을 선택해 주세요."

<br>

the topological dependencies among the three sub-tasks (Zhong et al., 2018) 연구를 기반으로 요금 예측에서 관련된 법률 기사를 프롬프트에 추가합니다. 형기 예측에는 관련된 법률 조항과 혐의를 추가합니다.

<br>

**Training**

- domain model: 도메인 모델들을 법률 데이터셋에서 훈련시키지만, 대규모 언어 모델(LLMs)은 변경하지 않습니다. 즉, LLMs는 사전 훈련된 상태를 유지하며, 도메인 특화 모델들만 법률 데이터에 맞춰 추가적으로 훈련됩니다.

  <br>

- Predictive model: Cross-entropy Loss 사용

  <br>
  
- Retrieval Model:  Izacard et al. (2022) 연구와 같이 contrastive loss 사용

<div id="Experiments"></div>

## Experiments

### Datasets

<img style="width: 100%; margin-top: 40px;" id="output" src="precedent/dataset.PNG">

중국의 'CAIL2018' 데이터셋을 사용하여 수행됩니다. 이 데이터셋은 중화인민공화국의 법률 문맥에서 실제 사건을 포함하며, 각 사건은 사실 설명과 법률 조항, 혐의, 징역 기간을 포함한 완전한 판결로 구성되어 있습니다.

<br>

CAIL2018 데이터셋은 훈련 세트, 검증 세트, 테스트 세트로 8:1:1의 비율로 무작위 분할됩니다.

<br>

또한 2022년 이후에 발생한 법률 사건만을 포함한 'CJO22' 데이터셋도 사용합니다. 이는 크기가 제한적이어서 도메인 모델의 훈련에는 적합하지 않습니다. 따라서 이 데이터셋은 추가 테스트 세트로만 사용됩니다.

<br>

케이스 데이터베이스에 있는 이전 사건들은 훈련 데이터셋에서 샘플링되며, 샘플 수는 4000개로 설정됩니다.

### Baselines

**JEC-QA 작업의 평가 지표: Accuracy**

- 자동 평가를 위해 출력 형식을 동일하게 유지하는 것이 어려우므로, 특히 7B LLM이 JEC-QA 작업에 대해 미세 조정되지 않았기 때문에, 평가의 정확성을 보장하기 위해 인간 평가를 선택했습니다.

<br>

**유사 사례 검색 작업의 평가 지표: Precision@k 및 MAP**

- Precision@k: 검색 결과나 추천 목록의 상위 k개 항목 중에서 관련 있는 항목의 비율을 측정합니다.

  <br>

- MAP (Mean Average Precision): 
  - 평균 정밀도(Average Precision, AP): 각 쿼리에 대해, 관련 있는 항목이 검색 결과에 나타날 때마다 정밀도를 계산하고, 이러한 정밀도의 평균을 취합니다.
  
    <br>
  
  - 예를 들어, 검색 결과에서 첫 번째 관련 항목이 1위, 두 번째가 3위, 세 번째가 5위에 있다면, 각 위치에서의 정밀도는 각각 1/1, 2/3, 3/5가 됩니다. 이를 평균내면 AP는 (1/1 + 2/3 + 3/5) / 3입니다.

    <br>
    
  - MAP는 이러한 AP를 모든 쿼리에 대해 평균낸 값입니다.

###  MAIN RESULTS

제안된 방법은 직접 생성(direct generations)과 질문 기반 검색(retrieval-based generations using the query)과 비교하여 모든 기준에서 큰 차이로 우수한 성능을 보였습니다.

<br>

7B 법률 LLM은 GPT-4를 크게 능가했으며, GPT-4의 검색 기반 생성에 비해서도 세 작업에서 더 우수한 성능을 보였고, JEC-QA 작업에서는 경쟁력 있는 결과를 보였습니다.

<br>

관련 법률 지식의 증거를 제공한 후, GPT-4는 반응을 상당히 개선했습니다(+9.92%).

<br>

이는 검색 기반 방법이 도메인 지식 부족에 의한 hallucinations을 줄이는 적절한 방법이며, 강력한 증거 평가 능력을 갖춘 GPT-4는 설득력 있는 증거가 있을 때 중국 법률 분야에 잘 적응할 수 있음을 나타냅니다.

<br>

7B 법률 모델이 생성한 초안 답변을 검색 및 수정에 사용했을 때, 두 질문 기반 검색 기준을 큰 폭(15.4% 및 23.9%)으로 능가했습니다.

<br>

또한 추가로 흥미로운 관찰에는 GPT-4로 최종 답변을 생성하지 않고, 7B 법률 모델로 domain evidence를 통해 최종 답변을 생성했을 때 7B 법률 모델의 초안 답변과 차이가 없었습니다.

<br>

이는 7B 법률 모델은 **zero evidence-assessing capacity**을 의미합니다.

<div id="제거 연구 및 추가 분석"></div>

# 제거 연구 및 추가 분석

### ANALYSIS OF RETRIEVAL METHODS

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="retriieve/answer.PNG">

answer-based approach은 query-based retrieval 보다 법률 분야와 같은 전문적인 도메인에서 더 좋은 성능을 발휘합니다.

<br>

query는 매우 간략하여 충분한 정보를 제공하지 못하는 반면, 법률 조항과 근거를 포함하는 answer은 훨씬 더 정보가 풍부합니다. 

<br>

**RETRIEVING A QUERY OR RETRIEVING AN ANSWER?**

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="retriieve/test.PNG">

<br>

각 검색에서 가장 유사한 상위 법률 조항들을 순서대로 나열하고, 상위 k 검색 결과에 실제 법률 조항이 포함되어 있는지를 확인하여 recall을 평가했습니다.

<br>

answer-based retrieval은 query-based retrieval보다 모든 k에서 큰 폭으로 우수한 성능을 보였습니다.

<br>

예를 들어, 답변을 기반으로 한 상위 1개의 검색 결과는 쿼리를 기반으로 한 상위 5개 결과와 경쟁할 수 있었습니다.

<br>

이는 초안 답변이 검색을 위한 쿼리보다 훨씬 더 많은 정보를 포함하고 있음을 나타냅니다.

<br>

유사 사례 검색에서도 answer-based retrieval이 query-based retrieval보다 좋은 성능을 보였습니다. (다른 설정은  https://github.com/myx666/LeCaRD를 따름)

<br>

**DOES THE QUALITY OF ANSWER MATTER FOR ANSWER-BASED RETRIEVAL?**

<br>

연구팀은 GPT-4와 7B 법률 LLM이 생성한 답변을 사용한 검색 결과를 비교했습니다.

<br>

query-based retrieval과 GPT-4의 답변 기반 검색을 비교했을 때, answer-based retrieval이 LCR, CP, LegalQA의 성능을 저하시켰습니다.

<br>

이는 GPT-4 답변의 도메인 지식 부족이 검색 결과의 질을 저하시켰음을 나타냅니다.

<br>

한편, 도메인 적응을 거친 7B 법률 LLM은 검색에서 강력한 답변을 제공했고, 최고의 성능을 달성했습니다.

### CASE ANALYSIS OF THE IMPROVEMENTS AFTER THE GPT-4 REVISION

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="retriieve/case.PNG">

1. 법률 조항 참조 추가

   - 때때로 7B 법률 LLM은 입력 지시를 따르지 않고, 참조된 법률 이름과 조항 인덱스의 핵심 정보를 제공하지 않고 유창한 답변만을 제공합니다.
   
      <br>
    
   - 이 경우 사용자는 답변의 신뢰성을 확인할 수 없습니다.
    
      <br>
    
   - 그러나 수정 후에는 각 답변이 참조된 법률 조항을 포함하게 되어, 답변의 정확성을 쉽게 확인할 수 있게 됩니다.

2. 증거 내의 hallucinations 수정

  - 도메인 적응된 LLM은 학습된 법률 지식을 바탕으로 증거를 제공할 수 있지만, 여전히 어느 정도 hallucinations(예: 잘못된 조항 인덱스)이 남아있을 수 있습니다.

     <br>

  - 근거 내용이 정확하면, 답변 기반 검색은 올바른 증거를 찾아내고, GPT-4에 의한 수정은 이 hallucinations을 해결하여 더 견고한 반응을 생성합니다.

3. 올바른 증거 선택

- 더 큰 시나리오에서, 7B 법률 모델의 답변이 부분적으로 hallucinations을 포함하더라도, 검색 구성 요소는 근거 생성에서 부분적으로 정확한 설명을 통해 올바른 증거를 식별할 수 있습니다.

   <br>

- 수정 단계에서 GPT-4는 올바른 증거를 평가할 수 있으며, 이는 정확한 답변을 생성하는 데 기여합니다.

### DOES THE ITERATION MAKE THE GENERATION BETTER?

본 논문에서 제안된 방법의 절차를 반복하여 응답을 개선할 수 있는지 여부에 대한 실험도 진행하였습니다.

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="retriieve/iterations.PNG">

위의 표와 같이 이러한 반복이 일관된 개선을 보이지 않습니다. 따라서, 성능 개선의 효과가 없는 것으로 나타납니다.

<div id="RELATED WORK"></div>

# RELATED WORK

### TASKS IN THE CHINESE LEGAL DOMAIN

- the Challenge of AI in Law (CAIL, http://cail.cipsc.org.cn/index.html)
- LeCaRD (Ma et al., 2021), JECQA (Zhong et al., 2020)
- EQUALS (Chen et al., 2023)

### CHINESE LEGAL LLMS

- the series of LaWGPT (Song, 2021)
  - Chinese-LLaMA-7B (Cui et al., 2023b), ChatGLM (Du et al., 2022), Chinese-alpaca-plus-7B (Cui et al., 2023b) 기반으로 제작
  
- Lawyer LLaMa (Huang et al., 2023)
  - 더욱 발전된 Chinese-LLaMa-13B (Cui et al., 2023b) 기반으로 제작

- LexiLaw (Hai, 2023)
  - ChatGLM-6B (Du et al., 2022) 기반으로 제작
  - LoRA (Hu et al., 2022), P-tuning (Liu et al., 2021), and fine-tuning 기술 활용

- Chatlaw (Cui et al., 2023a) 
  - Ziya-LLaMA-13B-v1 (IDEA-CCNL, 2023), Anima-33B (lyogavin, 2023) 기반으로 제작

- DISC-LawLLM Yue et al. (2023)
  - supervised fine-tuning datasets과 법적 추론 기능을 갖춘 LLM을 만들기 위해 legal syllogism prompting strategies을 채택하였습니다.

하지만 이러한 기존 모델들은 이미 공개된 법률 작업에 대해 훈련되어, zero-shot 능력(모델이 사전에 특정 작업에 대해 훈련되지 않았음에도 불구하고, 그 작업을 수행할 수 있는 능력)에 어려움을 겪을 수 있습니다.

### RETRIEVAL-AUGMENTED INFERENCE

- RAG(Retrieval-Augmented Generation)(Lewis et al., 2020b)
  - RAG 시스템은 BERT-based (Devlin et al., 2019) Document Retrieval Process (DRP)을 포함합니다.
  - 이 후 BART (Lewis et al., 2020a) 를 사용하여 답변을 생성합니다.

- EMDR2 (Yu et al., 2023)
  - 기대값-최대화(Expectation-Maximization) 알고리즘을 사용하여 여러 검색된 문서를 고려합니다.
  - 이 방식은 문서 검색과 추론 과정을 결합하여, 더 정확한 답변 생성을 목표로 합니다.

- Atlas (Izacard et al., 2022)
  - EMDR2 프레임워크를 기반으로 하고, 검색기(retriever)와 독자(reader) 구성 요소를 협력적으로 훈련
  - 540B PalM (Chowdhery et al., 2022)에 필적하는 few-shot learning 능력을 보여줍니다.

- RETRO (Borgeaud et al., 2022)
  - 사전 훈련 단계에서 광범위한 corpora에 대한 검색 메커니즘을 활용
  - 기존의 Transformer 기반 언어 모델과 달리, 검색 기능을 통합하여 모델의 성능을 향상시키는 것이 특징
  - GPT-3(Brown et al., 2020b)의 성능과 일치하는 성능

<div id="CONCLUSIONS AND FUTURE DISCUSSIONS"></div>

# CONCLUSIONS AND FUTURE DISCUSSIONS

연구에서는 LLM의 zero shot 도메인 콘텐츠 생성을 **Adapt-Retrieve-Revise** 절차로 재구성했습니다.

<br>

이 접근 방식은 smaller 7B LLM for domain adaptation, 외부 지식 기반에서 견고한 증거 검색, GPT-4의 증거 평가 및 수정 능력을 효과적으로 결합합니다.

<br>

이러한 방법은 중국 법률 작업에서 GPT-4의 zero shot 성능을 크게 향상시켰습니다.

<br>

하지만, 실험 비용이 증가하는 것과 평가의 유효성 사이의 균형을 맞추는 것은 향후 GPT-4 연구에서 남아있는 도전 과제입니다.

<div id="Legal Instruction Tuning을 위한 데이터셋 템플릿"></div>

# Legal Instruction Tuning을 위한 데이터셋 템플릿

- 중화인민공화국 도로교통안전법 제91조에 의거: [술을 마시고 자동차를 운전하는 자는 1개월 이상 자동차 운전면허를 일시 압수당할 수 있으며, 3개월을 초과하면 200위안 이상, 500위안 이하의 벌금을 병과한다. 술에 취한 상태에서 자동차를 운전하는 경우, 술에 취하지 않을 때까지 공안기관 교통관리부서의 제한을 받고, 15일 이하의 구류에 처하며, 자동차 운전면허증을 일시 압수당한다. 3개월 이상 6개월 이하, 500위안 이상 2000위안 이하의 벌금을 병과한다. 음주 후 영업용 자동차를 운전한 경우, 3개월간 자동차 운전면허증을 임시 압류하고 500위안의 벌금도 병과한다. 술에 취한 상태에서 영업용 자동차를 운전하는 경우, 술에 취한 상태에서 공안기관 교통관리부서의 제한을 받고, 15일 이하의 구류에 처하며, 자동차를 일시 압수하는 처벌을 받습니다. 6개월간 운전면허를 취득하면 2000위안의 벌금도 부과됩니다. 전 2항의 음주운전으로 1년 이내에 2회 이상 처벌을 받은 경우에는 운전면허를 취소하고, 5년 이내에 영업용 자동차를 운전하지 못한다.]

  <br>
  
- 사실 고려: [남자가 술에 취한 상태에서 오토바이를 타고 있었다.]

  <br>

- 판결: [공안기관 교통관리부서가 술에 취하지 않을 때까지 제한하고, 15일 이하의 구류에 처하며, 3개월 이상의 자동차 운전면허증을 일시 압류할 수 있다. 6개월 이상 시 500위안 이상, 2000위안 이하의 벌금을 병과한다.]
