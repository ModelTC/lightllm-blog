---
title: ""
tags:
- By MTC Team
- New Feature
excerpt: |
  dp rankers之间的前缀kv cache传输
mathjax: true
---


### 问题描述

在DP部署的过程中，每个dp_rank维护着自己的radix_cache，在dp_rank数较多的情况下，比如DP+EP H200单机部署DeepSeek-R1，会有8个dp_rank，cache命中的概率仅为1/8，做了很多重复的kv cache计算。为了减少这种重复计算，我们做了dp rankers之间的前缀kv cache传输。

### 具体实现

1. 我们给Req增加了`dp_max_kv_len`、`dp_max_kv_rank`、`dp_origin_kv_len`三个变量，分别代表所有dp_rank对该请求的最大匹配长度、最大长度所在dp_rank、本dp_rank匹配长度。由于`Req`本就使用shared_memory实现，所以所有dp_rank都可任意访问。在初始化请求的时候，每个dp_rank会将所有请求遍历，得到结果。
2. 每个dp_rank用shared_memory存储序列化后的`mem_manager`，其他dp_rank获取共享内存并且反序列化为`mem_manager`。由于torch.tensor在反序列化时，其c++源码中会将当前显卡切换到storage_device再做操作，这样得到的指针在进程当前上下文设备使用triton算子操作会出错。因此我们修改了tensor的rebuild函数`p2p_fix_rebuild_cuda_tensor`添加了`storage_device = torch.cuda.current_device()`，使得之后的`kv_trans_for_dp`kernel可以同时访问几张显卡的数据。
3. 每个dp_rank根据上述Req的三个变量，找出需要给其他dp_rank请求传输的请求，把请求相应的`kv_indexes`利用shared_memory进行共享。
4. 每个dp_rank根据上述Req的三个变量，找出需要读取的请求，`mem_manager`分配用来存放kv cache的索引，之后使用`kv_trans_for_dp`算子，根据`kv_indexes`和`dp_max_kv_rank`来将其他dp_rank`mem_manager.kv_buffer`中的kv cache读取到自己的`mem_manager.kv_buffer`。

### 性能测试

测试命令：
```
python test/benchmark/service/benchmark_sharegpt.py \
        --use_openai_api \
        --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
        --tokenizer /data/DeepSeek-R1 \
        --num-prompts 500 \
        --concurrency 32 \
        --history-turns 20
```
因为面向kv cache，所以测两遍。

由下方的测试结果可以看出，在第二次测试时，首字延迟小了0.04s, 大概17.1%。

| 指标 (Metric) | 第一次测试 | 第一次测试 (开启功能) | 第二次测试 | 第二次测试 (开启功能) |
| :--- | :--- | :--- | :--- | :--- |
| **total tokens** | 759336 | 759336 | 759336 | 759336 |
| **Total time** | 118.23 s | 119.16 s | 118.46 s | 115.16 s |
| **Throughput (qps)** | 4.23  | 4.20 | 4.22 | 4.34 |
| **Average latency** | 7.43 s | 7.49 s | 7.44 s | 7.23 s |
| **Average time to first token** | 0.41 s | 0.41 s | 0.41 s | 0.34 s |
| **Average latency per token** | 5.1 ms | 5.2 ms | 5.1 ms | 5.0 ms |
| **Average inter-token latency** | 69.3 ms | 69.8 ms | 69.3 ms | 68.0 ms |
