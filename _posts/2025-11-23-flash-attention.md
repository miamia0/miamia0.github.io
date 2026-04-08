---
layout: post
title:  "Flash Attention 学习记录"
date:   2025-11-23 15:19:17 +0800
categories: 
- LLM
---
# 背景

Transformer 的 Attention layer 涉及到 QKV 的矩阵乘法，在时间复杂度和内存复杂度上都是序列长度的平方级别。这导致在长序列场景下，计算量和内存占用都非常高，产生了性能瓶颈。

![alt text]({{ '/assets/images/posts/flash-attention/image-3.png' | relative_url }})
考虑到 GPU 的特点：SRAM 对比 HBM 要小的多，HBM 的带宽对比 SRAM 要低很多。因此尽量将计算放到 SRAM 中进行，减少 HBM 的读写次数。

Flash Attention 是经典的 Attention 优化方案，它的主要的思想是利用 GPU 硬件的特性最大化加速 Attention 的计算，通过1.算子融合减少内存搬迁和2. 分块计算使得单个块的计算都在 SRAM 来实现性能优化。

这篇文章是对 Flash Attention 的学习的记录。

# Flash Attention

简单的看一下 Attention 的公式
$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left( \frac{Q^\top K}{\sqrt{d_k}} \right) V$

一个最朴素的实现是先计算 QK 的矩阵乘法，然后计算 softmax，最后计算 dot product。
这样会导致每一步计算的中间结果需要存储到 GPU 的 HBM 中， 会有多次数据搬迁 IO 操作。
IO 包括 1. QK 的矩阵乘法计算一次 IO 2. softmax 计算的一次 IO 3. dot product 计算的IO。
写了个伪代码
``` python
# 计算 QK 的矩阵乘法
for i in range(N):
    for j in range(N):
        QK[i, j] = Q[i, :] @ K[j, :].T
# 计算 Softmax
for i in range(N):
    max_QK[i] = max(QK[i, :])
    sum_exp_QK[i] = sum(exp(QK[i, :] - max_QK[i]))

# 计算 dot product
for i in range(N):
    for j in range(N):
        P[i, j] = exp(QK[i, j] - max_QK[i]) / sum_exp_QK[i]
for j in range(N):
    O[i, :] += P[i, j] * V[j, :]
```

Flash Attention 的优化思路是将这三个算子融合在一起，减少中间结果的存储到 HBM 的次数，从而提升性能。

分析 Attention 算子的公式可知，如果不需要计算 softmax，只有QKV矩阵乘法计算，是可以无痛做 in SRAM 分块计算。 然而，普通的 Softmax 的公式是需要将 QK 的一行所有的值都计算完成才可以得到它们的 sum 和 max 值。这导致无法实现 in SRAM 的流式计算。

# Online Softmax
如何实现 Softmax 的流式计算呢？首先考虑一个普通的 Softmax 公式

![alt text]({{ '/assets/images/posts/flash-attention/image-4.png' | relative_url }})

为了让 Softmax的结果更稳定，避免数值多大导致数据溢出，一般会在计算均值之前，先把 对数的值减去最大值。
可见计算这里 safe softmax 需要三次循环。如果改用 online softmax可以将三次循环压缩到两次循环，如下列算法所示。

![alt text]({{ '/assets/images/posts/flash-attention/image-5.png' | relative_url }})
 
这样计算仍然有两次循环，不过因为 softmax 之后是 dot product，所以单行的结果最终需要做聚合，所以可以做进一步的公式推导。

\[
\begin{aligned}
\mathbf{o}_i'
&= \sum_{j=1}^i \frac{e^{x_j - m_i}}{d_i'} V[j,:] \\
&= \left( \sum_{j=1}^{i-1} \frac{e^{x_j - m_i}}{d_i'} V[j,:] \right)
+ \frac{e^{x_i - m_i}}{d_i'} V[i,:] \\
&= \left( \sum_{j=1}^{i-1}
\frac{e^{x_j - m_{i-1}}}{d_{i-1}'}
\frac{e^{x_j - m_i}}{e^{x_j - m_{i-1}}}
\frac{d_{i-1}'}{d_i'} V[j,:] \right)
+ \frac{e^{x_i - m_i}}{d_i'} V[i,:] \\
&= \left( \sum_{j=1}^{i-1}
\frac{e^{x_j - m_{i-1}}}{d_{i-1}'} V[j,:] \right)
\frac{d_{i-1}'}{d_i'} e^{m_{i-1} - m_i}
+ \frac{e^{x_i - m_i}}{d_i'} V[i,:] \\
&= \mathbf{o}_{i-1}'
\frac{d_{i-1}' e^{m_{i-1} - m_i}}{d_i'}
+ \frac{e^{x_i - m_i}}{d_i'} V[i,:]
\end{aligned}
\]

从这个公式可以看出，计算第 i 行的结果只需要第 i-1 行的结果，所以可以用迭代的方式计算得到 dot product 的结果。

在计算过程中，只需要保存分块的最大值和求和值，就可以计算出 dot product 的结果。

找到一个最小的 cuda 实现来帮助理解，https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu

```c++

```





# TBD FlashAttention2



Reference 
  - [FlashAttention paper](https://openreview.net/forum?id=H4DqfPSibmx)
  - [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691)
  - [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
  - [FlashAttention* V1/V2/V3](https://zhuanlan.zhihu.com/p/668888063)
