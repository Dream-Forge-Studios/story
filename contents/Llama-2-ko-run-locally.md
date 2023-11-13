---
date: '2023-11-01'
title: 'Llama 2 한국어 버전 window에서 실행하는 방법'
categories: ['Large Language']
summary: 'Llama 2 한국어 버전을 window에서 실행하는 방법을 알아봅시다.'
thumbnail: './test.png'
---

```
    conda create -n llama2Test python=3.10.9
    conda activate llama2Test
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    git clone https://github.com/oobabooga/text-generation-webui.git
```