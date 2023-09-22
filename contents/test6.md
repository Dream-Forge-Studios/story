---
date: '2023-02-14'
title: '딥러닝 필수 기초 수학 4탄'
categories: ['AI']
summary: '딥러닝을 위해 반드시 알아야할 필수 기초 수학을 총정리 하였습니다.'
thumbnail: './test.png'
---

<div id="13. 평균과 분산"></div>

# 13. 평균과 분산

> 확률 분포를 설명하는 두 가지 대푯값

<div id="평균"></div>

## 평균

평균에는 산술 평균, 기하 평균, 조화 평균이 있는데,

딥러닝에서는 시행을 무한번 반복하고 산술 평균을 구하는 **기댓값(Expectation)**
을 의미

- 질량 함수의 기대값 정의

<div style="display: flex; margin-top: 0px">
<img style="width: 180px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?E[X]=\sum_ix_ip_i ">
</div>

- 밀도(연속) 함수의 기대값 정의

<div style="display: flex; margin-top: 0px">
<img style="width: 250px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?E[X]=\int_{-\infty }^{\infty }xp(x)dx ">
</div>

<div style="display: flex; margin-top: 36px">
<div style="margin-top: 10px">보통</div>
<img style="width: 120px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?E[X]=\mu ">
<div style="margin-top: 10px">로 표기</div>
</div>

<div id="분산"></div>

## 분산

- 퍼진 정도
<div style="display: flex; margin-top: -30px"></div>

- 편차(평균과의 차이)를 제곱

  *절댓값을 안 쓰는 이유는 (-7, -1, 1, 7)과 (-4,-4, 4, 4)을 절댓값으로 분산을 구하면 같게 나온다.
<div style="display: flex; margin-top: -30px"></div>

- 질량 함수의 분산 정의

<div style="display: flex; margin-top: 0px">
<img style="width: 280px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?V[X]=\sum_i(x_i-\mu)^2p_i ">
</div>

- 밀도(연속) 함수의 분산 정의

<div style="display: flex; margin-top: 0px">
<img style="width: 340px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?V[X]=\int_{-\infty }^{\infty }(x_i-\mu)^2p(x)dx ">
</div>

<div style="display: flex; margin-top: 20px"></div>

- 표준편차 (<img style="width: 14px; margin-right: 2px; margin-left: 2px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\sigma ">)

<div style="display: flex; margin-top: 0px">
<img style="width: 90px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\sqrt{V[X]} ">
</div>

<div style="display: flex; margin-top: 24px"></div>
*루트를 하기 전에는 값이 너무 커지므로 단위를 맞춰주기 위해 사용

---

<div id="14. 균등 분포와 정규 분포"></div>

# 14. 균등 분포와 정규 분포

<div id="균등 분포"></div>

## 균등 분포

<div style="margin-top: 10px;">
<img style="width: 440px;" id="output" src="test4Img/ud_graph.PNG">
</div>

- Uniform distribution

<div style="display: flex; margin-top: -30px"></div>

- 평평하게 생겼다. (주사위, 동전)

<div style="display: flex; margin-top: -30px"></div>

- 식

<div style="display: flex; margin-top: 0px">
<img style="width: 168px; margin-right: 8px; margin-left: 20px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(x)=\left\{\begin{matrix}\frac{1}{b-a}  \\ 0 \end{matrix}\right.">
<img style="width: 90px; margin-right: 8px; margin-left: 20px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\begin{matrix} a\leq x\leq b\\otherwise \end{matrix}">
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 140px; margin-right: 8px; margin-left: 20px; margin-top: 40px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?X \sim U(a,b)">
</div>

- 평균

<div style="display: flex; margin-top: 0px">
<img style="width: 100px; margin-right: 8px; margin-left: 20px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{1}{2}(a+b)">
</div>

- 분산

<div style="display: flex; margin-top: 0px">
<img style="width: 130px; margin-right: 8px; margin-left: 20px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{1}{12}(b-a)^2">
</div>

<div id="정규 분포"></div>

## 정규 분포

<div style="margin-top: 10px;">
<img style="width: 440px;" id="output" src="http://infoso.kr/wp/wp-content/uploads/2020/10/%EC%A0%95%EA%B7%9C%EB%B6%84%ED%8F%AC1.png">
</div>

- Normal distribution or Gaussian distribution

<div style="display: flex; margin-top: -30px"></div>

- 종모양 (키)

<div style="display: flex; margin-top: -30px"></div>

- 식

<div style="display: flex; margin-top: 0px">
<img style="width: 280px; margin-right: 8px; margin-left: 20px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(x)=\frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}">
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 160px; margin-right: 8px; margin-left: 20px; margin-top: 40px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?X \sim N(\mu,\sigma^2)">
</div>

<div style="display: flex; margin-top: 10px"></div>

- 평균

<div style="display: flex; margin-top: 0px">
<img style="width: 22px; margin-right: 8px; margin-left: 20px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\mu">
</div>

<div style="display: flex; margin-top: 10px"></div>

- 분산

<div style="display: flex; margin-top: 0px">
<img style="width: 36px; margin-right: 8px; margin-left: 20px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\sigma^2">
</div>

---

<div id="15. 최대 우도 추정 (MLE)"></div>

# 15. 최대 우도 추정 (MLE)

<div id="조건부확률과 likelihood"></div>

## 조건부확률과 likelihood

<div style="margin-top: 10px;">
<img style="width: 440px;" id="output" src="test4Img/abBox.PNG">
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 140px; margin-right: 8px; margin-left: 26px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(cb|A)=\frac{1}{2}">
<img style="width: 140px; margin-right: 8px; margin-left: 94px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(cb|B)=\frac{2}{3}">
<img style="width: 100px; margin-right: 8px; margin-left: 40px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?likelihood">
</div>

<div style="display: flex; margin-top: 10px">
<img style="width: 140px; margin-right: 8px; margin-left: 26px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(b|A)=\frac{1}{2}">
<img style="width: 140px; margin-right: 8px; margin-left: 94px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(b|B)=\frac{1}{3}">
<img style="width: 100px; margin-right: 8px; margin-left: 40px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?likelihood">
</div>

<div style="display: flex; margin-top: 20px">
<div style="font-size: 24px; margin-left: 40px;">조건부확률</div>
<div style="font-size: 24px; margin-left: 124px;">조건부확률</div>
</div>

- likelihood
  - 예를 들어 어떤 상자에서 꺼낸지 모르는 상태에서 색 공을 뽑았을 때 두 상자에서 색공이 나올 가능도(likelihood)
  - 어떤 값이 관측되었을 때, 이것이 어떤 확률 분포에서 왔을 지.
  - 조건부 확률 값, but 확률 분포는 아니다. (합이 1이 아님)

<div id="MLE란"></div>

## MLE란

> Maximum Likelihood Estimation

<img style="width: 100%;" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99CDF1435B20DEC20A">
* 가우시안 분포로 가정

---

<div id="16. 최대 사후 확률 (MAP)"></div>

# 16. 최대 사후 확률 (MAP)

>likelihood 뿐만 아니라 prior distribution(사전 분포)까지 고려한 posterior(사후 확률)를 maximize 하는 것

<div id="Bayesian rule"></div>

## Bayesian rule

<div style="display: flex; margin-top: 10px">
<img style="width: 400px; margin-right: 0px; margin-left: 8px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(x|z)=\frac{p(x,z)}{p(z)}=\frac{p(z|x)p(x)}{p(z)}">
</div>

<br>

어떻게 저런 식이 나왔을까?

<div style="display: flex; margin-top: 26px">
<img style="width: 240px; margin-right: 0px; margin-left: 0px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(A|B)=\frac{p(A \cap B)}{p(B)}">
</div>

<div style="display: flex; margin-top: 26px">
<img style="width: 240px; margin-right: 0px; margin-left: 0px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(B|A)=\frac{p(A \cap B)}{p(A)}">
<div style="margin-top: 32px; margin-left: 10px;">이면</div>
<img style="width: 280px; margin-right: 0px; margin-left: 24px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(A \cap B)=p(B|A)p(A)">
</div>

<div id="MAP 식"></div>

## MAP 식

<div style="display: flex; margin-top: 26px">
<img style="width: 600px; margin-right: 0px; margin-left: 0px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\hat{x}=argmax_x\frac{p(z|x)p(x)}{p(z)}=argmax_x p(z|x)p(x)">
</div>

<div style="display: flex; margin-top: 26px">
<div style="margin-top: 24px">MLE와 비교했을 때</div>
<img style="width: 50px; margin-right: 0px; margin-left: 10px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(x)">
<div style="margin-top: 24px; margin-left: 10px;">가 추가된 것으로 x의 분포를 사전에 알고 있다는 의미이다. (prior distribution)</div></div>

단, 잘못된 사전 정보는 오히려 추정 성능에 악영향을 미친다.

---

<div id="17. 정보 이론 기초"></div>

# 17. 정보 이론 기초

<div id="Entropy"></div>

## Entropy

- 불확실성
  - 정보 이론에서 많이 나오는 글자는 짧은 비트로 적게 나오는 글자는 긴 비트로 표현하여 가장 효율적으로 표현하는 방법은 평균 코드 길이를 가장 적게 하는 것이다. 그런데 발생 확률이 균등하게 분포되어 있으면 평균 코드 길이는 길어지게 된다.
  - 즉, 평균 코드 길이가 최소화되기 위해서는 엔트로피(불확실성)을 최소화 해야한다.

- 식 : 가장 이상적인 함수(그래프를 보면 확인 가능)

<img style="width: 140px; margin-right: 0px; margin-left: 10px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image? \sum_{i}-p_ilogp_i">
<img style="width: 50%; margin-right: 0px; margin-left: 10px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://melonicedlatte.com/assets/images/201912/BB240ECE-0EEB-4601-B2FD-69D07553BBCB.jpeg">


<div id="Cross-entropy"></div>

## Cross-entropy

- 실제로는 <img style="width: 20px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p_i">를 따르지만 <img style="width: 18px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q_i">로 구하는 경우

  (실제 값을 모르는 경우, 결과로 정수값이 필요한데 정수값으로 나오지 않는 경우 등)
- 딥러닝에서는 <img style="width: 20px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q_i">가 출력, 최대한 <img style="width: 18px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p_i">와 비슷하게 만드려고 노력

<div id="KL-divergence"></div>

## KL-divergence

<img style="width: 90%; margin-right: 0px; margin-left: 10px; margin-top: 10px; margin-bottom: 0px;" id="output" src="./test6Img/kl_div.PNG">

<img style="width: 15px; margin-right: 4px; margin-left: 0px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p">와 <img style="width: 13px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q">의 분포 차이 (<img style="width: 60px; margin-right: 4px; margin-left: 4px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q - p">)

<div id="Mutual information"></div>

## Mutual information

<img style="width: 46%; margin-right: 0px; margin-left: 10px; margin-top: 10px; margin-bottom: 10px;" id="output" src="./test6Img/mutal.PNG">

x, y가 관련있는 정도(독립이면 0)
