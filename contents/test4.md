---
date: '2023-02-14'
title: '딥러닝 필수 기초 수학 2탄'
categories: ['AI']
summary: '딥러닝을 위해 반드시 알아야할 필수 기초 수학을 총정리 하였습니다.'
thumbnail: './test.png'
---

<div id="10. 스칼라를 벡터로 쉽게 미분하는 법"></div>

# 10. 스칼라를 벡터로 쉽게 미분하는 법

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

<div id="12. 쉽게 미분하는 법"></div>

# 12. 쉽게 미분하는 법

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