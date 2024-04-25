---
date: '2024-04-22'
title: 'vector DB 기반 판례 검색 시스템 제작기 1 - 데이터 만들기'
categories: ['LLM', 'Legal']
summary: 'LLM를 활용하여 판례 text embedding model을 만들어보자.'
thumbnail: './test.png'
---

<div id="개요"></div>

# 개요

 기존 판례 검색 시스템은 키워드 기반 단순 언어 매칭으로 원하는 판례를 찾기가 굉장히 어렵습니다. 
 
검색을 하면 수많은 검색 결과가 나타나며,

<img style="width: 40%; margin-top: 40px;" id="output" src="./precedentSerach1/figure1.PNG">