# 第四章：Agent 的记忆——上下文与 RAG

人类的记忆分为“瞬间的脑前额叶记忆”（短期运作，比如刚记住的电话号码）和“大脑皮层深处的长期记忆”（比如怎么骑自行车，或是小时候的故事）。

在 AI Agent 架构中，大模型的底层原理决定了它是**无状态的（Stateless）**——它本身不会记住你上一秒对它说了什么。如果你想让 Agent 表现得像一个能连贯思考的人，就必须为它搭建**记忆系统**。

## 1. 短期记忆（Short-term Memory）与上下文窗口

Agent 的短期记忆，其实就是你在一次 API 调用中所扔给它的**完整聊天记录（Context / History）**。

在最简单的结构中，当我们与 Agent 聊天时，程序会在后台把之前所有的提问和回答拼接起来，每次都整体发送给模型。在这个模型支持的“上下文窗口”（Context Window）之内（比如 128K 个 Token），这就是它的短期记忆。

这就像是**电脑的内存（RAM）**。它的读取速度最快，随时可以被 Agent 提取用来进行下一步推理，但它有严格的容量限制（尽管现在的窗口越来越大，但越长就越贵，而且信息注意力容易下降，产生所谓的 "Lost in the Middle" 现象）。

### 解决方案：记忆瘦身（Summarization）
当对话超过一定长度时，常见的做法是在后台启动另一个小的 LLM 任务：“请把前面的对话总结成一段摘要：用户想要去北京，已经定好了机票，现在正在找酒店”。这就是早期框架（如 LangChain 的 ConversationSummaryMemory）的原理。

但这仅仅解决了聊天长度的问题，如果 Agent 需要阅读公司过去一整年的财务报表呢？此时，我们需要长期记忆机制。

> **💡 拓展：情节记忆（Episodic Memory）**
> 在更高阶的 Agent 系统中，还存在第三种记忆形态——**情节记忆**，即让 Agent 记住自己过去完成某个任务的"经历"（包括中途遇到了什么错误、用了什么工具、最终怎么解决的）。这种"经验积累"能力，是第八章高阶反思模式（Reflection）的重要前置基础：Agent 可以在遇到类似问题时，主动回溯并借鉴自己过去的得与失。


## 2. 长期记忆（Long-term Memory）与向量数据库

Agent 的长期记忆就像是**电脑的硬盘（Hard Drive）**。这通常指的是外部数据库，特别是专门为大模型准备的**向量数据库（Vector Database）**。

为了帮助模型回忆起它未曾学过或者是极长的历史知识，我们要依靠一项名为 **RAG（Retrieval-Augmented Generation，检索增强生成）** 的技术。

### RAG 是如何工作的？

假设我们的 Agent 被赋予了一个长篇的“公司员工手册”（PDF 格式），并被问道：“今年的带薪年假有几天？”

1. **切块与嵌入（Chunking & Embedding）**
在系统初始化阶段，这本 PDF 手册会被切成一段一段的小文本块（Chunks）。然后通过文本嵌入模型（Embedding Model），每一段文本都被转换成了几百上千维的浮点数向量（Vector）。
这个向量就好像是文字在数学世界的**坐标位置**。

2. **存储（Storage）**
把这些代表着文字特征的浮点数向量存入向量数据库中（如 Pinecone, Milvus 或 Chroma）。

3. **检索（Retrieval）**
当用户提问“带薪年假有几天？”时，我们的程序先把这句话也变成一个查询向量（Query Vector）。
然后在向量数据库中寻找**距离最近（含义最相似）**的文本块。
数据库返回了一段内容：“根据2024年新规，所有正式员工的带薪年假为10天”。

4. **增强生成（Augmented Generation）**
到这一步，刚才由数据库找到的长短句，作为**附加的上下文内容**，被拼接进原始提示词里，最后交给大模型。
"你是一名为公司解答年假问题的助手。根据以下事实：【2024年新规，年假为10天】。请回答用户提问：今年的带薪年假有几天？"

大模型最后自信地生成了结果：“根据员工手册规定，今年有10天带薪年假！”

> **💻 RAG 原理的极简 Python 代码演示：**
> ```python
> from sentence_transformers import SentenceTransformer
> import numpy as np
> 
> # 1. 嵌入模型（将文字变成数字坐标 / 向量）
> embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
> 
> # 2. 构建"长期记忆"数据库
> knowledge_base = [
>     "公司规定每天早上 9 点上班。",
>     "2024年新规，所有正式员工的带薪年假为10天。"
> ]
> kb_vectors = embed_model.encode(knowledge_base)
> 
> # 3. 当遇到用户提问时，也把它转换成向量进行搜索 (检索 Retrieval)
> query = "请问今年的年假有几天？"
> query_vector = embed_model.encode([query])
> 
> # 4. 向量在数学空间中比较距离，寻找最相似的知识块
> similarities = np.dot(query_vector, kb_vectors.T) 
> best_match_idx = np.argmax(similarities)
> 
> print(f"从记忆中苏醒的上下文: {knowledge_base[best_match_idx]}")
> # 最后，把这个提取出来的字句交给带有大模型的 Prompt：
> # prompt = f"根据上下文：{knowledge_base[best_match_idx]}，\n 请回答：{query}"
> ```

## 3. RAG 不仅仅是对答如流

在 Agent 系统里，RAG 的用途远超简单的 PDF 问答。
对于复杂的 Agent，RAG 可以被注册为 Agent 的**一个可用“工具（Tool）”**。

当遇到特定的未知情况时，Agent 的大脑可以主动选择调用 `Search_Knowledge_Base(query="年假")` 甚至 `Search_Past_Dialogues`。这使得 Agent 具备了在海量硬盘里主动翻找线索、复盘过去犯错经验的能力，真正具备了进化的潜力。

---

走到这里，我们已经剖析了 Agent 的各个核心器官：大脑、双手和记忆系统。如果我们要用手写 Python 代码把这些拼凑起来，会是一件极其痛苦和缺乏复用性的工作。

因此，站在巨人的肩膀上，我们需要强大的 Agent 开发框架出场。下一章，我们将重点剖析两大主流实战框架的代表作：专注于可控循环的 **LangGraph**，与 Camel-AI 团队打造的面向开放世界的强大智能体 **OWL**。

**下一章**：[第五章：主流框架实战——LangGraph 与 OWL →](./05-agent-frameworks.md)
