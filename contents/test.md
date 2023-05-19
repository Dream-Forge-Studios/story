---
date: '2023-02-14'
title: '딥러닝 필수 기초 수학 1탄'
categories: ['AI']
summary: '딥러닝을 위해 반드시 알아야할 필수 기초 수학을 총정리 하였습니다.'
thumbnail: './test.png'
---

<div id="1. 함수"></div>

# 1. 함수

![funMainImg.PNG](testImg/funMainImg.PNG)

<div id="함수의 표현"></div>

## 함수의 표현

#### 1. 한 개 입력(x), 한 개 출력인 경우

<br>
<img style="width: 120px" alt='TeX' src="https://latex.codecogs.com/svg.image?f(x)=x^{2}">
<br>

#### 2. 두 개 입력(x,y), 한 개 출력인 경우
<br>
<img style="width: 140px" id="output" src="https://latex.codecogs.com/svg.image?f(x,y)=yx^{2}">
<br>

#### 3. 한 개 입력(x), 두 개 출력인 경우 (벡터 한개 출력)

<br>
<img style="width: 130px" id="output" src="https://latex.codecogs.com/svg.image?f(x)=\begin{bmatrix}x \\ x^{2}\end{bmatrix}">
<br>

#### 4. 두 개 입력(x,y), 두 개 출력인 경우 (벡터 한개 출력)
<br>
<img style="width: 180px" id="output" src="https://latex.codecogs.com/svg.image?f(x,y)=\begin{bmatrix}x+y \\ xy^{2}\end{bmatrix}">
<br>

[//]: # (<p></p>)
<br>
*여러개 출력인 경우에 그래프는 그냥 각각 그려주면 된다.

- 3번인 경우
<div>
<img style="width: 180px" id="output" src="testImg/oneFgraph.PNG">  ㅤㅤㅤ<img style="width: 190px" id="output" src="testImg/twoFgraph.PNG">
</div>
<br>

---

<div id="2. 로그 함수"></div>

# 2. 로그 함수

#### <img style="width: 60px" id="output" src="https://latex.codecogs.com/svg.image?log_{a}x">는 a(밑)를 몇 승 해야 x(진수)냐라는 의미

로그 함수는 딥러닝에서 가장 많이 사용하는 함수이다. 

그 이유는 로그를 통해 큰 수를 작게 만들고 복잡한 계산을 간편하게 하기 때문이다.

<div id="로그 그래프"></div>

## 로그 그래프
<p style=" FLOAT: none; CLEAR: none"><span class="imageblock" style="display: inline-block; width: 267px;  height: auto; max-width: 100%;" href="https://t1.daumcdn.net/cfile/tistory/2245BF3B534EAAEF32?original" data-lightbox="lightbox" data-alt="null"><img src="https://t1.daumcdn.net/cfile/tistory/2245BF3B534EAAEF32" style="max-width: 100%; height: auto;" srcset="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&amp;fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F2245BF3B534EAAEF32" width="267" height="225" alt="로그함수 그래프" filename="logarithmic_function_graph_movement_1.png" filemime="image/png"></span>&nbsp; <span class="imageblock" style="display: inline-block; width: 265px;  height: auto; max-width: 100%;" href="https://t1.daumcdn.net/cfile/tistory/223EFE3B534EAAEF35?original" data-lightbox="lightbox" data-alt="null"><img src="https://t1.daumcdn.net/cfile/tistory/223EFE3B534EAAEF35" style="max-width: 100%; height: auto;" srcset="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&amp;fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F223EFE3B534EAAEF35" width="265" height="226" alt="로그함수 그래프의 평행이동 - x축 방향" filename="logarithmic_function_graph_movement_2.png" filemime="image/png"></span><br><span class="imageblock" style="display: inline-block; width: 266px;  height: auto; max-width: 100%;" href="https://t1.daumcdn.net/cfile/tistory/222A493B534EAAEF0F?original" data-lightbox="lightbox" data-alt="null"><img src="https://t1.daumcdn.net/cfile/tistory/222A493B534EAAEF0F" style="max-width: 100%; height: auto;" srcset="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&amp;fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F222A493B534EAAEF0F" width="266" height="225" alt="로그함수 그래프의 평행이동 - y축 방향" filename="logarithmic_function_graph_movement_3.png" filemime="image/png"></span>&nbsp;<span class="imageblock" style="display: inline-block; width: 266px;  height: auto; max-width: 100%;" href="https://t1.daumcdn.net/cfile/tistory/254F743B534EAAF006?original" data-lightbox="lightbox" data-alt="null"><img src="https://t1.daumcdn.net/cfile/tistory/254F743B534EAAF006" style="max-width: 100%; height: auto;" srcset="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&amp;fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F254F743B534EAAF006" width="266" height="226" alt="로그함수 그래프의 평행이동 - x, y축 방향" filename="logarithmic_function_graph_movement_4.png" filemime="image/png"></span></p>

<br>

<div id="중요한 성질"></div>

## 중요한 성질

<div style="margin-top: -30px"></div>

1. <img style="width: 220px" id="output" src="https://latex.codecogs.com/svg.image?log_{a}xy=log_{a}x+log_{a}y">
<div style="margin-bottom: -20px"></div>

2. <img style="width: 170px" id="output" src="https://latex.codecogs.com/svg.image?log_{a}x^n=nlog_{a}x">
<div style="margin-bottom: -35px"></div>

3. <img style="width: 190px" id="output" src="https://latex.codecogs.com/svg.image?log_{a^m}x=\frac{1}{m}log_{a}x">
<div style="margin-bottom: -40px"></div>

4. <img style="width: 130px" id="output" src="https://latex.codecogs.com/svg.image?log_{a}b=\frac{log_{c}b}{log_{c}a}">
<div style="margin-bottom: -30px"></div>

5. <img style="width: 130px" id="output" src="https://latex.codecogs.com/svg.image?log_{a}b=\frac{1}{log_{b}a}">
<div style="margin-bottom: -30px"></div>

6. <img style="width: 116px" id="output" src="https://latex.codecogs.com/svg.image?a^{log_{a}x}=x">
<div style="margin-bottom: -30px"></div>

7. <img style="width: 150px" id="output" src="https://latex.codecogs.com/svg.image?a^{log_{b}c}=c^{log_{b}a}">
<div style="margin-bottom: -30px"></div>

<div style="margin-top: 30px"></div>

<div id="자연 상수 e"></div>

## 자연 상수 e 

밑 없이 <img style="width: 26px; margin-right: 4px; margin-left: 4px;" id="output" src="https://latex.codecogs.com/svg.image?log">만 쓰여있으면 밑이 e라고 생각하면 된다.

자연상수 e는 연속 성장(어떤 시스템, 물체, 현상 등이 지수적으로 성장하는 것)을 표현하기 위해 고안된 상수이다.

구체적인 의미는 <u>**100%의 성장률**</u>을 가지고 <u>**1회 연속 성장할 때 얻게 되는 성장량**</u>이다.

<br>

만약 50% 성장률을 가지고 1회 연속 성장한다면 <img style="width: 20px" id="output" src="https://latex.codecogs.com/svg.image?e^\frac{1}{2}">

100% 성장률로 2회 연속 성장한다면 그 성장량은 <img style="width: 18px" id="output" src="https://latex.codecogs.com/svg.image?e^2">

즉, <img style="width: 20px" id="output" src="https://latex.codecogs.com/svg.image?e^x">라는 식에서 지수 x가 갖는 의미는 <u>성장횟수 x 성장률</u>이다.

---

<div id="3. 벡터와 행렬 (선형대수학)"></div>

# 3. 벡터와 행렬 (선형대수학)
<br>
<div style="display: flex"><div style="padding: 32px 0px;">열벡터 : </div>
<img style="width: 40px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}1\\2\\3\end{bmatrix}">
<div style="padding: 32px 20px;">행벡터 : </div>
<img style="width: 90px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}1&2&3\\\end{bmatrix}">
</div>

<div style="display: flex; margin-top: 10px">
<div style="padding: 32px 0px;">행렬 : </div><img style="width: 90px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}1&2&3\\4&5&6\\\end{bmatrix}">
</div>
위의 행렬에서 2행 3열(2x3)의 값은 6
<br></br>

- 딥러닝에서 중요한 이유 : 연립방정식을 쉽게 표현하기 위해

연립 1차 방정식
(n차 방정식은 최고차항을 기준)

<br>
<div style="display: flex">
<img style="width: 150px" id="output" src="https://latex.codecogs.com/svg.image?\left\{\begin{matrix}2x+5y=10\\9x+6y=30\end{matrix}\right.">
<div style="padding: 12px 12px;">을 행렬로 표현하면</div>
<img style="width: 170px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}2&5\\9&6\\\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}%20=\begin{bmatrix}10\\30\end{bmatrix}">
</div>

<br>
상수가 같은 세 개의 연립 1차 방정식
<br>

<br>
<div style="display: flex">
<img style="width: 170px" id="output" src="https://latex.codecogs.com/svg.image?\left\{\begin{matrix}2x_{1}+5y_{1}=10\\9x_{1}+6y_{1}=30\end{matrix}\right.">
<div style="margin-left: 20px"></div>
<img style="width: 170px" id="output" src="https://latex.codecogs.com/svg.image?\left\{\begin{matrix}2x_{2}+5y_{2}=12\\9x_{2}+6y_{2}=20\end{matrix}\right.">
<div style="margin-left: 20px"></div>
<img style="width: 170px" id="output" src="https://latex.codecogs.com/svg.image?\left\{\begin{matrix}2x_{3}+5y_{3}=15\\9x_{3}+6y_{3}=19\end{matrix}\right.">
<div style="padding: 12px 12px;">을 행렬로 표현하면</div>
</div>
<br>
<img style="width: 360px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}2&5\\9&6\\\end{bmatrix}\begin{bmatrix}x_{1}&x_{2}&x_{3}\\y_{1}&y_{2}&y_{3}\\\end{bmatrix}%20=%20\begin{bmatrix}10&12&15\\30&20&19\\\end{bmatrix}">
<br>

<div id="주요 성질"></div>

## 주요 성질
1. **짝이 맞아야 곱할 수 있다.**

2x<u>2</u> 행렬과 <u>2</u>x3 행렬은 곱할 수 있고 2x<u>2</u> 행렬과 <u>3</u>x2 행렬은 곱할 수 없다.

2. **결과 사이즈는 맨 앞 x 맨 뒤 이다.**

<u>2</u>x4 행렬 곱하기 4x2 행렬 곱하기 2x<u>3</u> 행렬은 2x3이다.

3. **교환 법칙은 성립되지 않는다.**

<div id="벡터의 방향과 크기"></div>

## 벡터의 방향과 크기

벡터 [2,1]를 좌표평면에 나타내면 

(화살표는 방향, 직선의 길이는 크기)

<br>
<div style="width: 400px; z-index: -1;" >
<img src="testImg/vertorP.png">
</div>
<br>

**크기와 방향이 같으면 같은 벡터이다.** 시점이 달라도 같은 벡터일 수 있다.

위의 벡터와 아래 벡터는 시점은 다르지만 같은 벡터이다.

<br>
<div style="width: 400px; z-index: -1;" >
<img src="testImg/vertorP2.png">
</div>
<br>

### 벡터의 크기(화살표의 길이) 구하는 방법

1. L2-norm : <img style="width: 80px" id="output" src="https://latex.codecogs.com/svg.image?\sqrt{x^{2}+y^{2}}"> (피타고라스 정리)
<pr></pr>
2. L1-norm : <img style="width: 70px" id="output" src="https://latex.codecogs.com/svg.image?\left|x\right|+\left|y\right|"> (절댓값)

<br></br>

### 벡터 표기 법 : <img style="width: 20px; margin: -5px 10px" id="output" src="https://latex.codecogs.com/svg.image?\vec{x}">

---

<div id="4. 전치와 내적"></div>

# 4. 전치와 내적

<div id="전치(Transpose)"></div>

## 전치(Transpose)

<img style="width: 110px; margin-bottom: 10px" id="output" src="https://latex.codecogs.com/svg.image?A_{ij}%20=%20A^{T}_{ji}">
전치하면 i행 j열이 j행 i열된다.

<div style="display: flex; margin-top: 60px">
<img style="width: 140px" id="output" src="	https://latex.codecogs.com/svg.image?A=\begin{bmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\\\end{bmatrix}">
<div style="padding: 12px 12px;">일 때 전치하면</div>
<img style="width: 140px" id="output" src="	https://latex.codecogs.com/svg.image?A^{T}=\begin{bmatrix}a_{11}&a_{21}\\a_{12}&a_{22}\\\end{bmatrix}">
</div>

<img style="width: 160px; margin-top: 64px" id="output" src="https://latex.codecogs.com/svg.image?(a\vec{x})^{T}=\vec{x}^{T}a^{T}">
<div style="display: flex; margin-top: 30px">
<img style="width: 220px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\\\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\\\end{bmatrix}%20=%20\begin{bmatrix}b_{1}\\b_{2}\\\end{bmatrix}">
<div style="padding: 12px 12px;">일 때 전치하면</div>
<img style="width: 280px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}x_{1}&x_{2}\\\end{bmatrix}%20\begin{bmatrix}a_{11}&a_{21}\\a_{12}&a_{22}\\\end{bmatrix}=%20\begin{bmatrix}b_{1}&b_{2}\\\end{bmatrix}">
</div>
<br>
<br>

<div id="내적 (dot product)"></div>

## 내적 (dot product)

<img style="width: 330px; margin-bottom: 20px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}A_{1}\\A_{2}\end{bmatrix}\cdot%20%20\begin{bmatrix}B_{1}\\B_{2}\end{bmatrix}=A_{1}B_{1}+A_{2}B_{2}=\vec{A}^{T}\vec{B}">

위에서 점 표시( <img style="width: 4px;" id="output" src="https://latex.codecogs.com/svg.image?\cdot"> ) 내적을 의미

두 벡터를 내적한 결과는 스칼라이다. (스칼라는 방향없이 크기만 가지는 값)

<br>

#### 내적은 닮은 정도를 나타낸다.

<img style="width: 200px; margin-top: 20px" id="output" src="https://latex.codecogs.com/svg.image?\vec{A}^{T}\vec{B}=\left\|\vec{A}\right\|\left\|\vec{B}\right\|cos\theta">

1. <img style="width: 32px; margin-right: 5px;" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{A}\right\|"> 밑에 아무 것도 쓰여있지 않으면 L2-norm을 나타낸다. <img style="width: 40px;" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{A}\right\|_{1}">는 L1-norm

2. <img style="width: 11px; margin-right: 5px;" id="output" src="https://latex.codecogs.com/svg.image?\theta">는 두 벡터의 사잇각

<div style="margin-top: -10px; margin-bottom: 10px">
<u>내적이 닮은 정도를 나타내는 이유</u>는 빗변의 길이에 <img style="width: 46px; margin-left: 3px;" id="output" src="https://latex.codecogs.com/svg.image?cos\theta">
을 곱하면 밑변의 길이가 나오기 때문이다.
</div>

[cos(코사인)에 대해 이해가 되지 않는다면](https://hub1234.tistory.com/m/42)

<br>

두 벡터를 그래프로 그려보았을 때

<div style="width:500px; margin-top: 10px;">
<img id="output" src="testImg/dotProduct.png">
</div>

<div style="display: flex; margin-top: 20px;">
<div style="margin-top: -10px">위의 그림과 같이 
<img style="width: 100px; margin-right: 4px" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{A}\right\|\left\|\vec{B}\right\|cos\theta">에서
<img style="width: 70px; margin-left: 8px" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{B}\right\|cos\theta">
를 통해 방향이 다른 
<img style="width: 34px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{B}\right\|">
을 
<img style="width: 34px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{A}\right\|">
와 같은 방향에서 비교할 수 있게 된다.</div>
</div>

---

<div id="5. 극한과 입실론 - 델타 논법"></div>

# 5. 극한과 입실론-델타 논법

극한은 제대로 정의하는데는 100년이 걸렸을 만큼 쉬운 개념이 아니다.

우리가 고등학교 교육과정에서는


> 함수 <img style="width: 34px; margin-right: 4px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">에서 x가 a에 <u>한 없이 가까워질 때</u>, <img style="width: 34px; margin-right: 4px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">의 값이 일정한 값 L에 <u>한 없이 가까워지면</u> 함수 <img style="width: 34px; margin-right: 4px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">는 L에 수렴한다고 하고, L을 <img style="width: 34px; margin-right: 4px; margin-top: 10px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">의 극한값 또는 극한

이라 한다고 배웠다.
<br><br>

하지만 이는 극한을 직관적으로 이해시키기 위한 개념일 뿐 <u>오개념(계속 변화하는 상태가 아님)</u>을 만든다. 극한의 정의를 완전하게 이해하는 것은 너무 어려워 고등학교에서는 어쩔 수 없이 위와 같이 정의했다.
<br><br>

극한은 <u>입실론-델타 논법</u>을 통해 제대로 정의 된다.

> 입실론( <img style="width: 8px" id="output" src="https://latex.codecogs.com/svg.image?\epsilon "> )는 <img style="width: 36px; margin-left: 4px; margin-right: 6px;" id="output" src="https://latex.codecogs.com/svg.image?f(x)">의 오차를 표현하고, 델타( <img style="width: 9px" id="output" src="https://latex.codecogs.com/svg.image?\delta "> )는 <img style="width: 12px; margin-left: 5px;" id="output" src="https://latex.codecogs.com/svg.image?x "> 의 오차를 의미하는데,
> <br> <div style="display: flex; margin-top: 10px;"><div>임의의 </div><img  style="width: 50px; margin-left: 10px; margin-right: 6px; margin-top: -4px;" id="output" src="https://latex.codecogs.com/svg.image?\epsilon>0"> 에 대하여 <img style="width: 140px; margin-left: 10px; margin-right: 6px; margin-top: 4px;" id="output" src="https://latex.codecogs.com/svg.image?0<\left|x-a\right|<\delta ">의 범위에서 <img style="width: 140px; margin-left: 10px; margin-right: 6px; margin-top: 4px;" id="output" src="	https://latex.codecogs.com/svg.image?\left|f(x)-L\right|%3C\epsilon "> 이게 하는 <img style="width: 50px; margin-left: 10px; margin-right: 6px; margin-top: 0px;" id="output" src="	https://latex.codecogs.com/svg.image?\delta%3E0">가</div>
> <br> <div style="display: flex;"><div style="margin-top: 10px;">존재하면 </div><img style="width: 160px; margin-top: 7px; margin-left: 10px; margin-right: 10px;" id="output" src="https://latex.codecogs.com/svg.image?\displaystyle \lim_{x\to a}f(x)=L"><div style="margin-top: 10px;">로 정의한다.</div></div>

위의 글만 보면 이해하기가 정말 어렵다.

쉽게 이해하기 위해 극한이 존재하는 경우를 살펴보면

**임의의 <img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: 16px" id="output" src="https://latex.codecogs.com/svg.image?f(x)"> 오차 범위 안에 해당하는
<img style="width: 12px; margin-left: 5px;" id="output" src="https://latex.codecogs.com/svg.image?x "> 의 오차들에 해당하는
<img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: 16px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">가 모두 처음 <img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: 16px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">의 오차 범위 안에 있어야 한다.**

<br>

**1. 극한이 존재하는 경우**

<div style="display: flex">
<img style="width: 410px; margin-top: 20px; margin-right: 10px" id="output" src="testImg/g1.PNG">
<img style="width: 410px; margin-top: 20px" id="output" src="testImg/g2.PNG">
</div>
<div style="display: flex">
<img style="width: 410px; margin-top: 10px; margin-right: 10px" id="output" src="testImg/g3.PNG">
<img style="width: 410px; margin-top: 10px" id="output" src="testImg/g4.PNG">
</div>

<br>

**2. 극한이 존재하지 않는 경우**

<img style="width: 410px; margin-top: 20px; margin-right: 10px" id="output" src="testImg/gx.PNG">

<div style="display: flex; margin-top: 36px">
<div>
<img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: -4px" id="output" src="https://latex.codecogs.com/svg.image?f(x)"> 오차 범위 <img style="width: 10px; margin-left: 10px; margin-right: 6px; margin-top: -8px" id="output" src="https://latex.codecogs.com/svg.image?\frac{1}{2}"> 안에 해당하는
<img style="width: 12px; margin-left: 5px; margin-top: -5px; margin-right: 4px;" id="output" src="https://latex.codecogs.com/svg.image?x "> 의 오차들에 해당하는
<img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: 5px" id="output" src="https://latex.codecogs.com/svg.image?f(x)"> 중 처음 <img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: 5px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">의 오차 범위 안의 벗어나는 값이 있다.
</div>
</div>

<br>

[참고 자료](https://www.youtube.com/watch?v=JEe1rDCQ13E)

