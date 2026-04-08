---
layout: post
title:  "SGLang 源码浅读"
date:   2025-11-23 15:19:17 +0800
categories: 
- LLM
---
# 背景

Q：为什么需要推理引擎？不能直接用 Pytorch 的 model.forward 吗？

A：

GPU 算力资源昂贵： 如 Kimi K2.5 参数量 1T，激活参数 32B，服务 Kimi K2.5 需要 128 TFLOPs 的算力资源（对应 A100 0.5秒的计算量）。需要推理引擎做 算力利用率优化，推理加速，降低服务成本。

低延迟，高并发的应用的需求：现代的大模型有多轮会话， Agent等需要低延迟，高并发的应用的需求。需要推理引擎提供 流式输出，请求调度等能力。

KV Cache 复用需求：多轮对话历史会产生大量 KV Cache，需要推理引擎实现 KV Cache 管理，共享与复用，减少重复计算与显存占用，进一步提升吞吐、降低延迟。

SGLang 和 Vllm 是目前开源的最火热的两个大模型推理框架，社区活跃。本文的目标是对 SGLang 推理引擎的源码进行阅读，来对当前的推理引擎的设计目标，设计思想，主要优化方案的进行学习。

## Scheduler

Scheduler 是 SGLang Runtime（SRT）里负责单个进程中推理的调度的组件，整体执行流程是：

- 接收已分词后的内部请求
- 把请求组织成 batch（`ScheduleBatch`）
- 在本进程内驱动 `tp_worker/model_runner` 做模型推理（forward/sample）
- 并把输出 token id/embedding 通过 IPC 发送给 detokenizer/tokenizer

在接收到一个请求时，会先将它放在等待队列里(waiting_queue/disagg_prefill_bootstrap_queue),之后会有独立的 event loop 从队列里取出 batch 进行推理执行。在接收到请求时，可以先进行 KVCache prefetch，让 IO 和推理进行 overlap。

- 这里需要用一个独立的 queue来管理等待队列的好处：一方面可以实现队列反压，控制请求量，另一方面可以方便搞优先级调度。我理解python 的性能比较垃圾，搞队列可以搞异步 IO。


## PD 分离调度

PD 分离调度最开始是在 [**DistServe](https://arxiv.org/abs/2401.09670)** 被提出来的。之后 https://github.com/kvcache-ai/Mooncake 提出了 KVCache-Centric 分离式架构https://arxiv.org/abs/2407.00079，

先简单介绍一下什么是 P(prefill)D(decode)，从用户给大模型发送 Promot 后，大模型输出的第一个词的过程就是 Prefill，在之后的周期性输出个词就是 Decode。

如果将 Prefill 和 Decode 使用同一个GPU资源，由于两者的 workload 差异较大，会导致互相影响：

- 延迟干扰：当 GPU 在处理 Prefill 请求时，其他的 Decode 请求会被阻塞，导致延迟 P99 较高。
- 资源闲置：Prefill阶段GPU算力需求高，但是内存带宽闲置；Decode 阶段显存带宽满载，但是 GPU算力闲置。
- 资源矛盾：Prefill需要短 burst 的高算力，Decode需要持续的低延迟，难以统一调度