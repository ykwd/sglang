# 当SGLang 遇上 Mooncake：KV Cache 的三重奏

在大语言模型推理中，Prefill阶段需要将输入序列转化为Key-Value缓存（KV Cache）供后续解码使用。如果多个请求共享相同的前缀内容，那么这些前缀部分的KV Cache其实是完全一致的。通过将相同前缀的KV Cache缓存并复用，就可以避免重复计算。这样做的动机很直接：减少无效的算力浪费，降低延迟，并在并发场景下显著提升整体吞吐量。换句话说，**“一次计算，多次使用”** 让模型在处理相似输入时更高效、更省资源。因此，SGLang此前提出了RadixAttention，利用GPU中的闲置内存来缓存和重用相同前缀的KV Cache。

然而，随着上下文长度增长和并发请求增加，KV Cache的容量瓶颈问题日益凸显：GPU内存容量是很有限的，但请求的上下文长度和SLO要求是无限的。于是，SGLang干脆把现代CPU的"三级缓存"这一经典设计搬到了大模型里。这就是 HiCache：GPU显存当L1，Host内存当L2，Mooncake、3FS、NIXL等分布式缓存当L3。思路简单粗暴，却效果惊艳——既缓解了KV Cache的容量焦虑，又把性能拉到了新高度。

经过大家近3个月的努力，目前HiCache已经[成功发布啦](https://lmsys.org/blog/2024-01-17-sglang/)！我们很开心能在这里做一个HiCache相关的技术分享，抛砖引玉。下面我们会首先介绍SGLang HiCache的背景和整体架构，然后详细介绍HiCache的一些实现细节和遇到的挑战，最后介绍我们今后的规划。

**同时也欢迎大家来使用和贡献代码! 使用我们的Mooncake作为HiCache后端的方法可以参见[这篇文档](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/hicache-integration-v1)。**

## SGLang HiCache 简介

SGLang是一个高性能的大语言模型推理服务框架，专为大规模部署和优化推理性能而设计。HiCache（Hierarchical KV Cache）是SGLang中引入的一项关键技术创新，旨在解决现有KV Cache系统面临的容量瓶颈问题。在长上下文和多轮对话场景中，现有的RadixAttention虽然能够有效复用GPU内存中的KV Cache，但随着上下文长度增长和并发客户端增加，缓存命中率会显著下降，因为大部分历史KV Cache必须被驱逐，为新数据腾出空间。

针对上述挑战，HiCache应运而生。HiCache通过引入HiRadixTree作为页表来引用位于本地GPU和CPU内存中的KV Cache，并配备一个缓存控制器自动管理跨层级的KV缓存数据加载和备份，以及远程KV Cache的读取和写入，从而将GPU、CPU、SSD的“闲置”存储空间都利用起来，并依赖Mooncake、3FS、NIXL等分布式存储系统对全局的KV Cache进行存储和管理，在保障读取性能的同时大大提升了KV Cache的容量

![multiturn-per-turn](./resources/hicache_multi_turn_per_turn.png)


上面这幅图展示了我们在8卡H800的服务器上测试多轮对话的实验结果。其中GPU only, +L2, +Mooncake分别表示KV Cache包含了L1, L1+L2, L1+L2+Mooncake。可以看到KV Cache命中率对prefill性能有显著影响。当缓存命中时，TTFT明显低于缓存未命中的情况。在前三轮对话中，+Mooncake和+L2表现出相同的缓存命中率。在这个阶段，+L2略快于+Mooncake，因为它不需要从远程存储获取数据。然而，随着轮数增加，当KV缓存大小超过+L2的内存容量时，+L2的命中率逐渐下降，导致TTFT显著增加。相比之下，Mooncake保持了高命中率，其TTFT增长非常缓慢。

在实际部署中，Mooncake 将整个集群的内存聚合成一个大型分布式内存池，从而能够缓存大量的 KV Cache。每个缓存的 KV 都可以被所有 SGLang 实例共享，这在相同的内存预算下显著提升了缓存命中率。因此，在大规模SGLang集群中，Mooncake 能够降低推理的延迟，并提升吞吐量。

上述实验的细节详见我们的[blog](https://kvcache-ai.github.io/Mooncake/performance/sglang-hicache-benchmark-results-v1.html)。对HiCache的介绍可以参见SGLang的[blog](https://lmsys.org/blog/2025-09-10-sglang-hicache/)和[Strata](https://arxiv.org/abs/2508.18572)这篇论文。

## 技术实现与优化

### 整体架构

在现在的不少CPU架构中，速度快、容量小的L1和L2 cache是每个核心私有的，用于快速存取最热点的数据，而最大的L3 cache则是所有核心共享的，这可以显著降低cache内的数据冗余程度。与此类似，HiCache中L1和L2为每个推理实例的私有KV cache，而L3的KV cache则为整个集群内所有推理实例共享，

![HiCache Architecture](./resources/HiCache_Arch.png)

### 数据组织：HiRadixTree

在KV cache的数据组织方面，HiCache延续了RadixAttention的RadixTree，提出了HiRadixTree。RadixTree中每一个节点对应于GPU中的若干连续token的kv cache。每一条从根到叶子节点的路径就代表了一条请求的前缀，多个请求的公共前缀可以共享节点，从而避免数据的冗余。关于RadixAttention，更多细节可参见SGLang的[blog](https://lmsys.org/blog/2024-01-17-sglang/)。

![Radix Attention](./resources/radix_attn.jpg)

与此类似，HiRadixTree中，每一个节点对应于若干连续token的kv cache，并记录了这个节点的kv cache存储在本地GPU内存、Host内存、L3存储、亦或是其中的多个。如果kv cache存储在本地，则会详细记录其对应的存储地址。而处于性能的考虑，HiRadixTree不会存储或实时同步L3 kv cache的元数据信息，在需要读取L3数据时，会通过RPC向L3的后端实时查询所需读取数据的元数据信息，如是否存在、存储在哪台服务器的什么位置等。在我们的性能测试中，向Mooncake查询元数据信息通常只需要零点几到2毫秒的时间。

### Workflow

HiCache的工作流程主要涉及两个关键操作：预取和写回。当系统收到新的请求时，会首先本地的L1和L2中查找匹配的KV缓存。对于本地未命中的部分，则会尝试从L3中取。在Prefill计算结束后，则会考虑将新的数据存入L2或L3如果L1未命中，系统会依次检查L2（主机内存）和L3（Mooncake分布式存储）。当在L3中发现匹配的KV缓存时，系统会启动预取操作，将数据从Mooncake异步加载到L2，然后进一步加载到L1供计算使用。这个过程通过层间重叠机制实现，即在当前层执行计算的同时，并发地预取下一层的数据，有效隐藏了数据传输延迟。

在写回方面，HiCache采用智能的缓存管理策略。当L1中的KV缓存达到一定访问频率阈值时，系统会将其写回到L2进行备份。同样，L2中频繁访问的KV缓存会被写回到L3（Mooncake）进行长期存储。这种分层写回机制不仅确保了数据的持久性，还实现了跨实例的KV缓存共享。通过预取和写回的协同工作，HiCache能够在保证高性能的同时，最大化利用整个集群的存储资源。


**优化的数据平面**：
- 开发了GPU辅助I/O内核，相比标准`cudaMemcpyAsync`提供高达3倍的CPU-GPU传输吞吐量
- 采用"页优先"布局优化CPU内存池，与GPU的"层优先"布局解耦，实现更大的单次传输大小
- 结合零拷贝机制，在典型部署中实现高达2倍的吞吐量提升

**灵活的控制平面**：
- 当GPU缓存未命中但CPU内存命中时，采用层间重叠机制，在层N执行时并发加载层N+1的KV缓存，有效隐藏数据传输延迟
- 对于外部存储，缓存控制器在检测到存储层缓存命中时机会性地将数据预取到主机内存
- 支持多种预取策略：尽力而为模式、超时终止模式和更积极的请求暂存模式

### 1.2.3 存储后端支持

HiCache的设计亮点在于其简洁的存储后端接口。通过实现三个核心功能（`get(key)`、`exist(key)`、`set(key, value)`），可以轻松集成新的存储后端。目前支持的后端包括：

- **Mooncake**：高性能分布式内存存储系统
- **3FS**：阿里云TairKVCache团队提供的存储后端
- **NIXL**：NVIDIA设计的传输库，用于桥接GPU直连存储和云对象存储
- **HiCacheFile**：简单的文件后端，用于演示和参考

### 1.2.4 性能表现

根据社区反馈和基准测试结果，HiCache在多个场景下都取得了显著的性能提升：

- **Novita AI**：在Qwen3-Coder-480B编码代理场景中，平均TTFT（Time To First Token）降低56%，推理吞吐量翻倍，缓存命中率从40%提升至80%
- **Ant Group**：在DeepSeek-R1-671B模型的通用QA场景测试中，缓存命中相比完全重计算实现了84%的TTFT降低
- **官方基准测试**：实现了高达6倍的吞吐量提升和80%的TTFT降低

### 1.2.5 实现细节

从代码实现来看，HiCache的核心组件包括：

1. **HiRadixCache**：继承自RadixCache，管理分层缓存逻辑
2. **HiCacheController**：负责缓存控制、数据传输和存储操作
3. **HiCacheStorage**：抽象存储接口，支持多种后端实现

系统支持多种写入策略（write-through、write-through-selective、write-back）和IO后端（kernel、direct），能够根据不同的部署场景和性能需求进行灵活配置。

HiCache的成功不仅体现在性能提升上，更重要的是它提供了一个可扩展的架构，使得社区可以轻松集成新的存储后端，推动了整个生态系统的发展。
