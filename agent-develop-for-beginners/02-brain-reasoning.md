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
# 实战代码：Plan-and-Solve 模式的基本结构
def plan_and_solve_agent(task):
    # ===== 第一阶段：Plan（规划）=====
    # 只让大模型做一件事：不允许调用任何工具，只输出计划清单
    plan_prompt = f"""
    你是一名项目经理。请为以下任务制定一个分步执行计划，
    仅输出步骤列表，不要执行任何操作：\n任务：{task}
    """
    plan_text = call_llm(plan_prompt)
    steps = parse_steps(plan_text)  # 解析成列表，例如：["Step1: 搜索...", "Step2: 计算..."]
    print(f"[Plan] 制定了 {len(steps)} 个执行步骤：\n{plan_text}\n")

    results = []
    # ===== 第二阶段：Solve（执行，每一步内部使用 ReAct 循环）=====
    for i, step in enumerate(steps):
        print(f"\n--- 开始执行 Step {i+1}: {step} ---")
        # 将每个子步骤交给 ReAct 执行者，并附带已有结果作为上下文
        context = f"全局任务：{task}\n已完成步骤结果：{results}\n当前步骤：{step}"
        # 这里的 simple_react_agent 就是第二章里的那个 ReAct 函数！
        step_result = simple_react_agent(task=context, max_steps=5)
        results.append(f"Step {i+1} 结果: {step_result}")

    return "\n".join(results)
```

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

## 5. 三种模式的对比与总结

为了更直观地理解，我们将这三种模式的区别和各自的优劣势总结如下：

### 5.1 思维链（CoT）
- **特点**：纯粹的内部推理过程。通过 Prompt 强制大模型把“内心的思考步骤”写出来，不涉及工具调用。
- **优势**：简单高效，不需要复杂的代码框架，响应速度快，成本低。
- **劣势**：无法获取外部实时信息或跨越自身的知识盲区；如果推导链条中某一步出错，后续很容易“一错全错”。
- **适用场景**：逻辑推理题、数学运算等不需要外部知识辅助的基础任务。

### 5.2 ReAct 模式（推理与行动）
- **特点**：“走一步看一步”的动态循环。允许大脑（推理）和双手（调用工具）在收到反馈（Observation）后交替进行。
- **优势**：能调用外部工具与真实世界交互；具备即时纠错能力，能根据最新鲜的反馈灵活调整下一步思路。
- **劣势**：缺乏大局观，容易因为某个 Action 的失败而在原地反复尝试钻牛角尖，陷入死循环；整体链路的 Token 消耗较大。
- **适用场景**：需要搜索实时信息、查询知识库、执行短平快的动态问答等。

### 5.3 Plan-and-Solve（计划与执行）
- **特点**：谋定而后动。在开展任何行动之前，必须先要求大模型输出一份包含明确阶段划分的全局执行清单（Plan），然后再逐一执行（Solve）。
- **优势**：目标感极强，有全局清单兜底指引，大幅度降低迷失方向的风险，极少出现死循环。
- **劣势**：灵活性差，一旦早期的计划存在致命假设错误，后期很难扭转局面；对模型本身的前瞻规划能力要求极高。
- **适用场景**：跨度长、流程清晰的复杂任务，如自主研发一整个软件项目、撰写万字研报等。

## 6. 进阶思考：大规划与小执行（Planner-Executor 架构）

学习到这里，你可能会产生一个极其敏锐的疑问：**“既然 Plan 之后还是要由 Agent 执行任务，执行任务时如果遇到阻碍（比如查不到资料）依然需要重新思考和调用工具，那是不是意味着 Plan-and-Solve 模式，其实在内部是包含了 ReAct 模式的？”**

答案是**绝对肯定**的。在现代主流的复杂 Agent 框架（如 AutoGPT, LangChain 的 Plan-and-Execute 等）中，**Plan-and-Solve 和 ReAct 绝不是非此即彼的互斥选项，而是“宏观与微观”的完美嵌套。**业界通常将这种组合称为 **Planner-Executor（规划者-执行者）架构**：

1. **宏观层面：大模型作为 Planner（项目经理）**
   单纯的 ReAct 直接上阵容易“见树不见林”，所以我们在最外层使用 Plan-and-Solve。在这个阶段，大模型是不被允许立刻调用工具的，它只负责一件事：**根据终极目标，拆解出一个结构化的宏观 TODO 计划清单。**
2. **微观层面：大模型作为 Executor（打工人，本质在使用 ReAct）**
   有了清晰的清单后，框架会依次把 Plan 中的具体子任务丢给底层的“执行者 Agent”。这个 Executor 拿到小任务后，就会在这个具体任务的边界内**开启你所熟悉的 ReAct 循环**（Thought -> Action -> Observation），去不断地调用工具、试错、纠正，直到攻克这个小任务打卡下班。

**总结来说：**
单一的 ReAct 犹如一个不停试错的实干家，走得快但容易原地打转；单一的 Plan-and-Solve 犹如按图索骥的书呆子，工具一崩就卡死。而**将两者结合（宏观大目标 Plan，微观小目标 ReAct）**，能让 Agent 既拥有把控全局的心智，又有见招拆招的灵活手腕，这正是当下构建复杂 AI 应用的**最佳实践**！

---

如果说这一章讲解的是 Agent 的大脑，那么下一章，我们将赋予大脑可以操控真实世界的手术刀：**工具调用（Tool Calling）**。

**下一章**：[第三章：Agent 的双手——工具调用 →](./03-hands-tool-calling.md)
