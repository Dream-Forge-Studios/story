---
date: '2023-02-14'
title: '딥러닝 필수 기초 수학 2탄'
categories: ['AI']
summary: '딥러닝을 위해 반드시 알아야할 필수 기초 수학을 총정리 하였습니다.'
thumbnail: './test.png'
---

<div id="10. 쉽게 미분하는 법"></div>

# 10. 쉽게 미분하는 법

<div id="스칼라를 벡터로 쉽게 미분하는 법"></div>

## 스칼라를 벡터로 쉽게 미분하는 법

<div style="display: flex; margin-top: 0px">
<img style="width: 12px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?f">
<div style="margin-top: 10px">에 변화량을 구해보면</div>
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 550px; margin-right: 8px; margin-left: 0px; margin-top: 18px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?df=\frac{\partial f}{\partial x_1}dx_1+\frac{\partial f}{\partial x_2}dx_2=\begin{bmatrix}dx_1&dx_2\\\end{bmatrix}\begin{bmatrix} \frac{\partial f}{\partial x_1}\\ \frac{\partial f}{\partial x_2} \end{bmatrix}=d\underline{x}\frac{\partial f}{\partial x^T}">
</div>

<div style="display: flex; margin-top: 0px">
<div style="margin-top: 32px">의 식이 나오는데</div>
<img style="width: 40px; margin-right: 8px; margin-left: 10px; margin-top: 18px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{\partial f}{\partial x^T}">
<div style="margin-top: 32px">는 미분할 때의 식이다.</div>
</div>

<div style="display: flex; margin-top: 0px">
<div style="margin-top: 36px">즉,</div>
<img style="width: 16px; margin-right: 4px; margin-left: 10px; margin-top: 28px;" id="output" src="https://latex.codecogs.com/svg.image?f">
<div style="margin-top: 36px">의 변화량을 구하는 식을 계산하면 되는 것이다.</div>
</div>

<div id="벡터를 벡터로 미분하는 법"></div>

## 벡터를 벡터로 미분하는 법

<img style="width: 560px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?df=\begin{bmatrix} df_1&df_2  \\ \end{bmatrix}=\begin{bmatrix} dx_1&dx_2  \\ \end{bmatrix}\begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_2}{\partial x_1} \\ \frac{\partial f_1}{\partial x_2} & \frac{\partial f_2}{\partial x2} \\ \end{bmatrix}=dx\frac{\partial f}{\partial x^T}">

<div id="벡터를 벡터로 미분할 때의 연쇄법칙"></div>

## 벡터를 벡터로 미분할 때의 연쇄법칙

<div style="display: flex; margin-top: 0px">
<img style="width: 100px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?x\to y\to z">
<div style="margin-top: 10px">로 이어질 때 연쇄 법칙이 작용한다.</div>
</div>

<div style="display: flex; margin-top: 20px">
<img style="width: 56px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?x\to y">
<div style="margin-top: 18px">에서</div>
<img style="width: 120px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\partial y=\partial x\frac{\partial y}{\partial x^T}">
</div>

<div style="display: flex; margin-top: 10px">
<img style="width: 56px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?y\to z">
<div style="margin-top: 18px">에서</div>
<img style="width: 120px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\partial z=\partial y\frac{\partial z}{\partial y^T}">
</div>

<div style="display: flex; margin-top: 10px">
<div style="margin-top: 18px">그러므로</div>
<img style="width: 100px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?x\to y\to z">
<div style="margin-top: 18px">는</div>
<img style="width: 164px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\partial z=\partial x\frac{\partial y}{\partial x^T}\frac{\partial z}{\partial y^T}">
</div>

<div id="스칼라를 행렬로 미분하는 법"></div>

## 스칼라를 행렬로 미분하는 법

<img style="width: 200px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?df=tr(dX\frac{\partial f}{\partial X^T})">

<div id="행렬을 행렬로 미분하는 법"></div>

## 행렬을 행렬로 미분하는 법

<img style="width: 340px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?dvec(F)=dvec(X)\frac{\partial vec(F)}{\partial vec^T(X)}">

<img style="width: 400px; margin-right: 8px; margin-left: 10px; margin-top: 26px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?* vec(\begin{bmatrix} x_11 & x_12 \\ x_21 & x_22 \\ \end{bmatrix})=\begin{bmatrix} x_11& x_12 & x_21 & x_22 \\ \end{bmatrix}">

<div id="벡터를 행렬로 미분하는 법"></div>

## 벡터를 행렬로 미분하는 법

<div style="display: flex; margin-top: 10px">
<img style="width: 460px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?y=\begin{bmatrix} y_1&y_2  \\ \end{bmatrix}, x=\begin{bmatrix} x_1&x_2  \\ \end{bmatrix}, W = \begin{bmatrix} w_{11}&w_{12}  \\ w_{21}&w_{22}\end{bmatrix}">
<div style="margin-top: 18px">일 때</div>
<img style="width: 84px; margin-right: 8px; margin-left: 10px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?y=xW">
<div style="margin-top: 18px">에서</div>
<img style="width: 26px; margin-right: 8px; margin-left: 10px; margin-top: -3px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?W">
<div style="margin-top: 18px">로 미분한다면</div>
</div>

<img style="width: 680px; margin-right: 8px; margin-left: 10px; margin-top: 30px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix} y_1&y_2  \\ \end{bmatrix}=\begin{bmatrix} x_1&x_2  \\ \end{bmatrix} \begin{bmatrix}w_{11}&w_{12}  \\ w_{21}&w_{22}\end{bmatrix}=\begin{bmatrix} x_1w_{11}+x_2w_{21}&x_1w_{12}+x_2w_{22}  \\ \end{bmatrix}">

<img style="width: 560px; margin-right: 8px; margin-left: 10px; margin-top: 30px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{\partial y}{\partial w^T}=\begin{bmatrix} \frac{\partial y_1}{\partial w_{11}}& \frac{\partial y_2}{\partial w_{11}} \\ \frac{\partial y_1}{\partial w_{12}}& \frac{\partial y_2}{\partial w_{12}} \\ \frac{\partial y_1}{\partial w_{21}} & \frac{\partial y_2}{\partial w_{21}} \\ \frac{\partial y_1}{\partial w_{22}}& \frac{\partial y_2}{\partial w_{22}} \\ \end{bmatrix}=\begin{bmatrix} x_1 & 0 \\ 0 & x_1 \\ x_2 & 0 \\ 0 & x_2 \\ \end{bmatrix} = x^T \bigotimes I_2">

---

<div id="11. 왜 그라디언트는 가장 가파른 방향을 향할까"></div>

# 11. 왜 그라디언트는 가장 가파른 방향을 향할까

<div style="display: flex; margin-top: 0px">
<div style="margin-top: 10px">loss 함수</div>
<img style="width: 50px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?L(w)">
<div style="margin-top: 10px">를</div>
<img style="width: 80px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?w=w_k">
<div style="margin-top: 10px">로 1차까지만 테일러 급수 전개를 하면</div>
</div>

<div style="display: flex; margin-top: 10px">
<img style="width: 360px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?L(w)\simeq L(w_k)+(w-w_k)\left.\begin{matrix}\frac{\partial L}{\partial w^T}\end{matrix}\right|_{w=w^k}">
</div>

<br>

<div style="display: flex; margin-top: 0px">
<div style="margin-top: 10px">윗 식에서</div>
<img style="width: 30px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?w_k">
<div style="margin-top: 10px">을 업데이트할</div>
<img style="width: 176px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?w_{k+1}=w_k+\Delta">
<div style="margin-top: 10px">(변경해야할 값)을 대입하면</div>
</div>

<div style="display: flex; margin-top: 10px">
<img style="width: 360px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?L(w+1)\simeq L(w_k)+\Delta\left.\begin{matrix}\frac{\partial L}{\partial w^T}\end{matrix}\right|_{w=w^k}">
</div>

<br>
따라서

<div style="display: flex; margin-top: 20px">
<img style="width: 360px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?L(w+1)-L(w_k)\simeq \Delta\left.\begin{matrix}\frac{\partial L}{\partial w^T}\end{matrix}\right|_{w=w^k}">
<div style="margin-top: 10px">(</div>
<img style="width: 20px; margin-right: 8px; margin-left: 4px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\Delta">
<div style="margin-top: 10px">와 그라디언트의 내적)</div>
</div>

<br>
<div style="display: flex; margin-top: 0px; margin-bottom: 10px">
<div style="margin-top: 10px">이므로 가장 업데이트를 많이 하기 위해서는 </div>
<img style="width: 20px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\Delta">
<div style="margin-top: 10px">를 그라디언트 방향으로 변경해야한다.</div>
</div>

*내적값이 크다는 것은 같은 방향이라는 것을 뜻한다.

---

<div id="12. 랜덤 변수와 확률 분포"></div>

# 12. 랜덤 변수와 확률 분포

<div id="랜덤 변수"></div>

## 랜덤 변수

> 사건을 입력하면 실수의 값으로 출력해주는 함수

예) 동전 뒤집기 : 앞면 - 1, 뒷면 - 2

랜덤 변수는 대문자, 실수 값으로 변환된 값은 소문자로 표시

<div id="확률 분포"></div>

## 확률 분포

- 확률 질량 함수(PMF)

  - 동전, 주사위 등
  - 각각이 양수이면서 0과 1 사이의 값을 가진다.
  - 합이 1 이다.

- 확률 밀도(연속) 함수(PDF)
  - 키, 나이
  - 양수는 맞지만 0과 1 사이는 아니다.
  - 적분이 1이다.
  - 정적분을 통해 확률을 구한다. (그래프에서 범위 내의 밑넓이)
  - 적분을 통해 확률을 구하기 때문에 키를 예로 들었을 때 딱 165일 확률은 0이다.
  
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

<div id="조건부확률과 likehood"></div>

## 조건부확률과 likehood

<div style="margin-top: 10px;">
<img style="width: 440px;" id="output" src="test4Img/abBox.PNG">
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 140px; margin-right: 8px; margin-left: 26px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(cb|A)=\frac{1}{2}">
<img style="width: 140px; margin-right: 8px; margin-left: 94px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(cb|B)=\frac{2}{3}">
<img style="width: 100px; margin-right: 8px; margin-left: 40px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?likehood">
</div>

<div style="display: flex; margin-top: 10px">
<img style="width: 140px; margin-right: 8px; margin-left: 26px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(b|A)=\frac{1}{2}">
<img style="width: 140px; margin-right: 8px; margin-left: 94px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(b|B)=\frac{1}{3}">
<img style="width: 100px; margin-right: 8px; margin-left: 40px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?likehood">
</div>

<div style="display: flex; margin-top: 20px">
<div style="font-size: 24px; margin-left: 40px;">조건부확률</div>
<div style="font-size: 24px; margin-left: 124px;">조건부확률</div>
</div>

- likehood
  - 위의 그림에서는 동시에 A상자에서 어떤 공이 나올지와 B상자에서 어떤 공이 나올지에 대한 확률을 의미 
  - 조건부 확률 값, but 확률 분포는 아니다. (합이 1이 아님)

<div id="MLE란"></div>

## MLE란

<div style="display: flex; margin-top: 36px">
<div style="margin-top: 10px">정규 분포를 예시를 들어보면</div>
<img style="width: 400px; margin-right: 8px; margin-left: 10px; margin-top: 6px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?z_1=x+n_1, z_2=x+n_2, n \sim N(0,\sigma ^2)">
<div style="margin-top: 10px">일 때</div>
</div>

<div style="display: flex; margin-top: 36px">
<img style="width: 110px; margin-right: 8px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(z_1,z_2|x)">
<div style="margin-left: 10px"> <u>"x가 뭐였길 때 z가 저렇게 나와을까?"</u>에 대한 답을 찾고자 하는 것</div>
</div>

<div style="display: flex; margin-top: 36px">
<div style="margin-top: 10px">그렇다면</div>
<img style="width: 70px; margin-right: 0px; margin-left: 10px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?z_1, z_2">
<div style="margin-top: 10px; margin-left: 10px;">이 나올 확률이 높은 경우 즉, <u>likehood가 최대가 되도록 하는 x</u>를 구하면 된다. </div>
</div>

<div style="display: flex; margin-top: 36px">
<div style="margin-top: 10px">*예시에서는 정규분포 이므로 순간변화량(기울기)이 0이 되는 값이 최대</div>
</div>

<div style="display: flex; margin-top: 26px">
<div style="margin-top: 26px">구해보면</div>
<img style="width: 280px; margin-right: 0px; margin-left: 10px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(z_1,z_2|x)=p(z_1|x)p(z_2|x)">
<div style="margin-top: 24px; margin-left: 10px;">(독립시행 가정),</div>
<img style="width: 280px; margin-right: 0px; margin-left: 10px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(z_1|x)=\frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{(z_1-x)^2}{2\sigma^2}}">
</div>

<div style="display: flex; margin-top: 36px">
<img style="width: 130px; margin-right: 0px; margin-left: 0px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\hat{x}=arg max_x">
<img style="width: 600px; margin-right: 0px; margin-left: 8px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(z_1|x)p(z_2|x)=log p(z_1|x)p(z_2|x)=-\frac{(z_1-x)^2}{2\sigma^2}-\frac{(z_2-x)^2}{2\sigma^2}">
</div>

<div style="display: flex; margin-top: 26px">
<div style="margin-top: 24px">미분하면</div>
<img style="width: 340px; margin-right: 0px; margin-left: 18px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{z_1-\hat x}{\sigma^2}+\frac{z_1-\hat x}{\sigma^2}=\frac{z_1+z_2}{2}=0">
</div>

<div style="display: flex; margin-top: 26px">
<div style="margin-top: 24px">즉 여기서는 들어온 measurement의 평균을 구하는 것이 MLE이다.</div>
</div>

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

- 균등분포일 때 1

<div id="Cross-entropy"></div>

## Cross-entropy

- 실제로는 <img style="width: 20px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p_i">를 따르지만 <img style="width: 18px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q_i">로 구하는 경우
  
  (실제 값을 모르는 경우, 결과로 정수값이 필요한데 정수값으로 나오지 않는 경우 등)
- 딥러닝에서는 <img style="width: 20px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p_i">가 출력, 최대한 <img style="width: 18px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q_i">와 비슷하게 만드려고 노력

<div id="KL-divergence"></div>

## KL-divergence

- <img style="width: 15px; margin-right: 4px; margin-left: 0px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p">와 <img style="width: 13px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q">의 분포 차이 (<img style="width: 60px; margin-right: 4px; margin-left: 4px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q - p">)

<div id="Mutual information"></div>

## Mutual information

- x, y가 관련이는 정도(독립이면 0)