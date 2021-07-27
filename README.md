# Welcome to Model Transformer

## Content
<!-- toc -->
- [Introduction](#introduction)
- [QAT Operators](#qat-operators)
<!-- tocstop -->

## Introduction

This repo contains a quantize package which includes all kinds of quantization algorithms. such as `Uniform Quant`, `DSQ`, `LSQ`, and etc. The transformer can automatically transform your float model into ***intx*** model, thus you don't need to change your model manually.
## QAT Operators

Our QAT ops includes two kinds of quantization method, namely `uniform quantization` and `non-uniform quantion`.

   Name     |  UniformQuant   | Cls Performance |  Detection Performance |
 -----------|-----------------|-----------------|------------------------|
LSQDPlus    | `Yes`           | [Link](https://git-core.megvii-inc.com/tanfeiyang/lowbit_classification_config/-/tree/tanfeiyang/dev2/LSQDPlus)    |[Link](https://git-core.megvii-inc.com/tanfeiyang/detection_configs/-/tree/tanfeiyang/dev/atss) |
LSQ         | `Yes`           | [Link]()    | [Link]() |
DSQ         | `Yes`           | [Link]()    | [Link]() |
APOT        | `NO`            | [Link]()    | [Link]() |