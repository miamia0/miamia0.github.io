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
- 把请求组织成 batch（ScheduleBatch）
- 在本进程内驱动 tp_worker/model_runner 做模型推理（forward/sample）
- 并把输出 token id/embedding 通过 IPC 发送给 detokenizer/tokenizer

在接收到一个请求时，会先将它放在等待队列里(waiting_queue/disagg_prefill_bootstrap_queue),之后会有独立的 event loop 从队列里取出 batch 进行推理执行。在接收到请求时，可以先进行 KVCache prefetch，让 IO 和推理进行 overlap。

- 这里需要用一个独立的 queue来管理等待队列的好处：一方面可以实现队列反压，控制请求量，另一方面可以方便搞优先级调度。我理解python 的性能比较垃圾，搞队列可以搞异步 IO。

## KV Cache

KV Cache 的核心作用是避免重复计算历史 token 的 K/V。Decode 一个新 token 时，当前 token 的 Q 需要 attend 到之前所有 token 的 K/V；这些历史 token 的 K/V 在之前 step 已经算过，所以可以缓存下来。多个请求如果共享相同 prompt prefix，也可以复用同一段 prefix KV Cache。

在 SGLang 里，RadixTree 主要负责 prefix cache 的索引管理，真正的 K/V tensor 不存在 radix tree 里。整体可以分成三层：

1. KV 数据本体：存在 TokenToKVPool 这类内存池里。例如 MHA 路径会为每一层分配 k_buffer[layer] 和 v_buffer[layer]，形状类似 [size + page_size, head_num, head_dim]。
2. request 到 KV slot 的映射：ReqToTokenPool.req_to_token[req_pool_idx, token_pos] -> kv_index，表示某个请求第几个 token 的 KV 在 KV pool 的哪个位置。
3. prefix 到 KV indices 的索引：RadixCache.TreeNode 里存 key、value、children、lock_ref 等字段。其中 key 是 token prefix segment，value 是对应的 KV pool indices，不是 K/V tensor 本体。

所以一次请求的大致流程是：

token ids
-> RadixCache.match_prefix() 找最长已缓存 prefix
-> 返回 MatchResult.device_indices
-> 写入 req_to_token_pool
-> 新 token 分配新的 KV slot/page
-> attention 把新 K/V 写入 k_buffer/v_buffer
-> 请求结束或 chunk 结束时 RadixCache.insert() 写入 prefix -> kv_indices

SGLang 使用 radix tree 而不是普通 trie，一个重要原因是 radix tree 可以把一段 token prefix 压缩在同一个节点里。这样节点的 key 和 value 都是一段连续 segment，更适合和 paged KV cache、chunked prefill、chunk attention 这类按块处理的机制配合。

查询 prefix cache 时，SGLang 会把 token ids 包成 RadixKey，再从 root 根据 child_key 找子节点。命中压缩节点后，用 child.key.match(key) 比较最长公共前缀。完全匹配就继续往下走；如果只匹配节点的一部分，会触发 _split_node()，把公共前缀拆成新的中间节点。最后把一路命中的 node.value 拼起来，返回 MatchResult.device_indices。

插入时也类似：请求完成后从 req_to_token_pool 取出当前请求对应的 kv_indices，再用 RadixCache.insert() 把 token prefix -> kv_indices 写进树。如果新 key 和已有节点只共享一部分 prefix，也会 split 节点。也就是说 radix tree 的本质索引是：

(token prefix segment, extra_key namespace) -> KV pool indices segment

extra_key 用来隔离不能共享 KV 的 namespace，比如 LoRA/adaptor 等配置不同的请求。开启 paged KV cache 时，RadixKey 还会按 page_size 做 page alignment，非对齐的尾部通常不会进 radix cache。

### 缓存淘汰

RadixCache 的淘汰不是删除 K/V tensor 对象，而是从 leaf node 拿到 node.value，再调用 allocator 把对应 KV slot/page 还回 free list。公开参数 --radix-eviction-policy 支持几种策略：

- lru：按 last_access_time 淘汰最久未访问的 prefix。
- lfu：按 (hit_count, last_access_time)，优先淘汰命中少的 prefix。
- slru：命中次数达到阈值后进入 protected，否则优先淘汰 probationary。
- priority：按 (priority, last_access_time)，低 priority 先淘汰。

如果开启 HiCache，SGLang 会在普通 GPU KV prefix cache 之外增加 host 侧 L2 KV cache。--enable-hierarchical-cache 开启层级缓存，--hicache-size 或 --hicache-ratio 控制 host cache 大小，--hicache-write-policy 控制 GPU KV 写回 host 的时机：
- write_through：插入或命中后更积极写到 host。
- write_through_selective：更保守，通常需要更高命中热度。
- write_back：GPU eviction 时再写回 host，然后释放 GPU KV。

SGLang 的 KV tensor 存在 GPU/host KV pool 的连续或分页 buffer 里；radix tree 只存 token prefix 到这些 KV slot/page indices 的映射，用来做 prefix reuse、引用保护和 eviction 管理。


## PD 分离调度

PD 分离调度最开始是在 [DistServe](https://arxiv.org/abs/2401.09670) 被提出来的。之后 https://github.com/kvcache-ai/Mooncake 提出了 KVCache-Centric 分离式架构https://arxiv.org/abs/2407.00079，

先简单介绍一下什么是 P(prefill)D(decode)，从用户给大模型发送 Promot 后，大模型输出的第一个词的过程就是 Prefill，在之后的周期性输出个词就是 Decode。

如果将 Prefill 和 Decode 使用同一个GPU资源，由于两者的 workload 差异较大，会导致互相影响：

- 延迟干扰：当 GPU 在处理 Prefill 请求时，其他的 Decode 请求会被阻塞，导致延迟 P99 较高。
- 资源闲置：Prefill阶段GPU算力需求高，但是内存带宽闲置；Decode 阶段显存带宽满载，但是 GPU算力闲置。
- 资源矛盾：Prefill需要短 burst 的高算力，Decode需要持续的低延迟，难以统一调度
