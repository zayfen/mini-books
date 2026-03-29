# 第八章：高阶推理流——让 Agent 学会反思 (Reflection & ToT)

> **⚠️ 高阶章节预警**：本章探讨的技术更关注提升 Agent 在极高难度任务下的“良品率”，适用于学术探索或硬核商业落地。如果你的 Agent 只是用来发发邮件查询天气，阅读本章可能会让你觉得牛刀杀鸡。

在我们第二章讨论过的大脑直觉推理模式（如 CoT 或 ReAct）中，Agent 大多是“一条道走到黑”。它顺着逻辑一直往下推倒，如果最后输出了错误的代码，程序就直接抛出异常崩溃了。

真正的专家在工作中会怎么做？人类不仅会犯错，而且人类特别擅长**自我反思（Self-Correction / Reflection）**。当你的代码运行报错时，你会看着 Error Log 说：“噢，原来这里由于拼写错误导致变量未定义，我改一下。”

这种机制，同样能通过巧妙的架构设计赋予 Agent。

## 1. 基础反思模式：打分员与演员 (Actor-Critic)

这是一种极为有效且容易落地的设计模式。我们不再期翼一个 Agent 能够一次性输出完美的答案，而是把它切分为两个角色：
- **生成者（Actor/Generator）**：只管按任务生成草稿。
- **点评员（Critic/Reflector）**：拿到草稿和任务原意，尖锐地指出问题所在。

代码在循环时，发生的是：“生成草稿 -> 批评指出不足 -> 拿着批评意见回去重新生成 -> 再次批评...” 
直到 Critic 认为代码完美或者超过最大循环限制。

> 💡 **实战代码：简易的反思自驱网络**
> 
> ```python
> def reflection_loop(task_prompt, max_retries=3):
>     # 1. 尝试第一次生成
>     current_draft = call_llm(prompt=task_prompt, role="Actor")
>     
>     for attempt in range(max_retries):
>         # 2. 调用专门点评的 Agent 寻找缺陷
>         critic_prompt = f"任务要求：{task_prompt}\n当前草稿：{current_draft}\n请严肃批评漏洞并提供具体修改意见。如果完美无缺，请仅输出 'PERFECT'"
>         feedback = call_llm(prompt=critic_prompt, role="Critic")
>         
>         if "PERFECT" in feedback:
>             print("考核通过！")
>             return current_draft
>         
>         print(f"收到评审反馈：{feedback}")
>         # 3. 带着反馈，把稿件打回去重修！
>         revision_prompt = f"之前你的草稿：{current_draft}\n这是审核员的苛刻批评：{feedback}\n请吸取教训，改进并输出最新草稿。"
>         current_draft = call_llm(prompt=revision_prompt, role="Actor")
>         
>     return current_draft # 实在修不好了，返回最后一稿
> ```

通过 LangGraph 这类流转控制框架，你可以轻易地画出一条“Actor -> Critic -> (if fail) -> Actor -> (if pass) -> END”的回环有向边，极大拔高 Agent 写长代码和执行长线思维链的成功率。

## 2. 从一条直线到大树繁枝：思维树 (Tree of Thoughts, ToT)

如果碰到了真正连人类都容易迷糊的逻辑推理题（比如算24点、规划复杂的旅行路线），简单的 CoT 或者是线性反思就不够了，因为错误的根本原因可能在第一步“方向就走错了”。

这就引出了学术界赫赫有名的 **思维树 (ToT)**。
在这个模式下，大模型不急着一路走到黑，而是把它变成了一个寻路游戏（类似于传统的 BFS 广度优先搜索/ DFS 深度优先算法）：
1. **生成分支**：在当前步骤，模型不产出一个决定，而是产出 3 个可能的尝试方向。
2. **评估打分**：用另一个模型（或者内部 Prompt）对这三个方向进行打分（比如：1方向是死胡同得0分，2方向似乎有戏得5分，3方向极有可能成功得9分）。
3. **剪枝与探索**：放弃低分的分支，沿着高分分支继续衍生下一个层级的新节点，这就是一棵不断生长的“决策树”。如果在树的深处发现不对劲，可以随时向上**回溯（Backtracking）**到没有走过的平行岔路口。

### 为什么我们不常用 ToT？
ToT 展现出了最惊艳的逻辑上限，但代价极其高昂：它消耗巨大的计算开销（Token 和延迟）。解决一道算法题可能要发几十上百次 LLM API 请求。
因此，虽然你可能不需要亲手实现一个 ToT 系统，但理解“多路生成并打分回退”的美妙哲学，能让你的 Agent 系统上限深不可测。

---

通过巧妙的设计模式，我们能在软件层面拉高模型的智力体验。但这也伴随着高昂的延迟与 Prompt Context 爆炸的体积费用。有没有办法，不靠每次在输入框里嘱咐一长串咒语，就能让它成为某个领域的超级熟手呢？

这就是我们要谈论的终极路线。
**下一章**：[第九章：模型微调与专职化 (Agent Fine-Tuning) →](./09-agent-fine-tuning.md)
