# 第二章：Agent 的大脑——推理与规划

很多人误以为，给模型发送一条带有工具指令的 prompt，模型就会像变魔术一样自动调出网页查资料。事实上，能够决定**“什么时候查资料，需要查什么资料，查完资料该怎么用”**，才真正体现 Agent 大脑的智慧。

这就是大模型在 Agent 中所负责的最核心工作：**推理（Reasoning）与规划（Planning）**。

## 1. 为什么大模型需要规划？

如果你问大模型一个直白的问题：
> "法国的首都是哪？"

它可以几乎零延迟地脱口而出："巴黎"。

但如果你问它一个复杂的、具有多重逻辑链路的问题：
> "在2024年奥运会主办城市的市长是谁？他/她今年多少岁？"

如果是一个单纯的生成模型，它可能会因为缺乏足够的内部信息和计算时间而产生"幻觉"（胡说八道）。但在 Agent 系统中，我们会让大模型“慢下来”，或者说，赋予它**思考的时间**。

在这里，我们引入了几个非常著名的框架和思维模式：

## 2. 思维链（Chain of Thought, CoT）

这是一种非常简单但也极其有效的技巧。在 Agent 开发中，我们会在指令里往往要求大模型："Let's think step by step"（我们需要一步一步地思考）。

这能强制 LLM 把思考过程“显式”地输出出来，而不是直接跳到答案。在每输出一个思考步骤的过程中，模型都会积累额外的上下文，从而大幅度提高最终推理的准确率。

> `CoT` 不再只是一个 Prompt 开发技巧，更是很多 Agent 底层运行时的前置条件。

## 3. ReAct 模式（推理与行动）

ReAct 是 **Reasoning（推理）** + **Acting（行动）** 的缩写，这是早期也是最经典的 Agent 骨架模式，由普林斯顿大学和 Google 研究人员提出。

ReAct 要求 Agent 以一种非常结构化的方式来解决问题，它通常循环执行以下三个步骤：
1. **Thought（思考）**：我现在该做什么？（比如：我需要先知道2024年奥运会主办城市）
2. **Action（行动）**：我要调用什么工具？（比如：使用 `Search[2024 Olympics Host City]`）
3. **Observation（观察）**：工具链返回的结果是什么？（比如：搜索结果返回"巴黎"）

然后 Agent 带着这个观察结果，重新进入（Thought）环节：
1. **Thought**：我已经知道是巴黎。我现在需要去查巴黎市长是谁。
2. **Action**：使用 `Search[Mayor of Paris]`。
3. **Observation**：搜索返回"安妮·伊达尔戈 (Anne Hidalgo)"。
4. **Thought**：接着查她的出生年份...

如此循环往复，直到 Agent 得出最终答案。它完美诠释了“一边推理，一边行动，再基于反馈继续推理”。

## 4. Plan-and-Solve（计划与解决）模式

ReAct 虽然经典，但有个缺点：**容易陷入死循环**。如果一个 Action 报错或者没查到有效信息，Agent 可能因为只盯着眼前这一步，而在原地无休止地尝试，彻底忘记了自己的最终目标。

这就引出了 **Plan-and-Solve（计划与执行）** 模式。
在采取任何行动之前，Agent 会先制定一个全局的执行计划清单。

```python
# 实战代码：一个极简的 ReAct 推理循环 (Reasoning Loop)
def simple_react_agent(task, max_steps=5):
    memory = []  # Agent 的工作记忆
    memory.append(f"任务目标: {task}")
    
    for step in range(max_steps):
        # 1. 思考 (Thought)
        thought_prompt = f"根据当前记忆：{memory}。你下一步该做什么？请输出 'THOUGHT: 你的思考' 然后输出 'ACTION: 工具名[参数]'，或者 'FINISH: 最终答案'。"
        agent_response = call_llm(thought_prompt) 
        
        # 2. 从回答中解析它的意图
        if "FINISH:" in agent_response:
            return agent_response.split("FINISH:")[1]
            
        elif "ACTION:" in agent_response:
            action_str = extract_action(agent_response) # 解析出行动
            print(f"[Step {step}] System: {agent_response}")
            
            # 3. 观察 (Observation)
            observation = run_tool(action_str)
            print(f"[Step {step}] 观察结果: {observation}")
            
            # 把经验加回记忆，供下一步循环
            memory.append(f"执行了 {action_str}，结果是 {observation}")
            
    return "达到最大循环次数，任务失败。"
```
这段代码虽然简陋，但它展示了 Agent 循环最本质的内核：**每遇到一个障碍，就根据最新现状重新思考（Thought），采取对策（Action），收集反馈（Observation），永不放弃直到完成最终目标。**

---

如果说这一章讲解的是 Agent 的大脑，那么下一章，我们将赋予大脑可以操控真实世界的手术刀：**工具调用（Tool Calling）**。

**下一章**：[第三章：Agent 的双手——工具调用 →](./03-hands-tool-calling.md)
