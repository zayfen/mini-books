# 第三章：Agent 的双手——工具调用（Tool Calling）

我们在前两章探讨了 Agent 为什么需要有一颗聪明的“大脑”来思考和规划。可是，无论思考得多么缜密，**如果不能采取实际行动，它永远只能纸上谈兵。**

在纯聊天的大语言模型时代，我们只能得到一行行干瘪的回答。
而在 Agent 时代，大模型可以“操作物理世界”，其中最关键的技术叫做 **工具调用（Tool Calling / Function Calling）**。

## 1. 什么是工具调用？

简单理解，**工具就是任何一个拥有特定输入和输出的函数（Function）。**
在传统编程中，是你的 Python 脚本决定什么时候去调用 `get_weather(city)`。
在 Agent 开发中，**“大脑（LLM）”会自己决定是否要调用这个函数，以及给这个函数传入什么参数。**

### OpenAI 的 Function Calling 革命
2023 年，OpenAI 首次发布了 Function Calling 能力。在这之前，让 LLM 输出一段 JSON 格式的数据去传给后端函数是一项极具挑战性的任务，因为它总是喜欢“废话连篇”（比如回复："好的！这是您的JSON..."），从而导致解析失败。

Function Calling 从模型底层结构上解决了这个问题：它被专门微调过，能够在需要行动时，**只输出一段结构严密的 JSON 数据。**

## 2. 工具调用的工作流（Workflow）

假设我们要构建一个能查天气并给老板发邮件的贴身管家 Agent。

### 步骤一：定义并声明工具
在现代 Agent 开发中，你不再需要手写繁琐的 JSON 格式，大多数框架允许你直接把一个普通的 Python 函数转换为工具，只需要你写好**类型注解**和**文档字符串（Docstring）**：

```python
import json

def get_weather(location: str) -> str:
    """
    获取指定城市的当前天气情况。
    
    参数:
        location: 城市名称，例如：北京, 巴黎
    """
    # 这一步在真实世界中会调用类似 weather.com 的 API
    # 这里我们返回一个模拟的天气 JSON 字符串
    simulated_weather = {"location": location, "condition": "Rainy", "temperature": 15}
    return json.dumps(simulated_weather)

# 很多框架（如 OpenAI SDK 或 LangChain）会自动读取上方的注释和类型
# 并把它转化为 LLM 能够读懂的格式（Tool Schema）发送给云端。
```

### 步骤二：大脑做出请求调用的决定
用户输入指令："查查明天巴黎会下雨吗？如果下雨，给老板发邮件说把会议改到线上。"

大脑（LLM）收到指令和工具菜单后，会进行分析：
"这个问题我本身不知道（因为超出我训练数据的时效），但我发现 `get_weather` 这个工具可以帮我。所以我现在暂停回答，并发起一个工具调用请求！"

此时，LLM 会中断生成，抛出一个特殊的信号，内容是：
**"我想调用 `get_weather`，参数是 `{'location': '巴黎'}`。你等会查完了告诉我结果。"**

### 步骤三：我们在本地执行真实代码
作为开发者，我们的 Python 脚本捕获到了这个调用请求，然后去执行真正的网络请求：
```python
# 脚本在本地真正地访问外部天气 API
result = requests.get("https://api.weather.com/v1/forecast?city=巴黎").json()
# 返回结果比如说是：{"condition": "Rainy", "temperature": 15}
```

### 步骤四：将结果返回给大脑
我们将刚刚查到的真实数据组装起来，重新丢给 LLM 的上下文中：
**"这是你刚才要求调用的工具的结果：`{'condition': 'Rainy', 'temperature': 15}`。"**

### 步骤五：大脑继续推理甚至调用下一个工具
LLM 收到数据后，继续它的 ReAct（我们在第二章讲过的观察与行动）循环：
"哦，巴黎明天是雨天（Rainy）。用户说如果下雨要发邮件。那么接下来，我要调用 `send_email` 工具..."

这个循环不断持续，直到任务完成。

## 3. 工具的种类与想象空间

有了 Tool Calling，Agent 的能力边界被无限拓宽了。工具不仅限于简单的数学运算或天气查询：

- **网页检索工具**：像 Google Search 或 Bing API，让 Agent 打破知识停滞的禁锢，拥有联网能力。
- **文件操作工具**：读取本地的 PDF、Excel，写入报告并保存。
- **代码执行工具**：允许 Agent 编写一段 Python 脚本，并在安全的沙盒（Sandbox）里执行以获得运算结果。
- **多模态感知工具**：例如 Camel-AI 团队的 **OWL** 框架所展示的强大功能——通过 `Playwright` 自动化操控浏览器，像真人一样在网页上浏览、点击、滚动，并截取屏幕发送回多模态大模型进行验证。

---

如果“工具调用”让你的 Agent 拥有了千变万化的双手，那要如何让这双手避免“每次遇到相同问题都要重头查一遍资料”的尴尬呢？

接下来我们进入第四部分，赋予 Agent 过目不忘的能力：**记忆机制与 RAG。**

**下一章**：[第四章：Agent 的记忆——上下文与 RAG →](./04-memory-and-rag.md)
