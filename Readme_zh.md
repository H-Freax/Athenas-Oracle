# 🌌 Athenas-Oracle：AI 驱动的文档洞察引擎 📚

欢迎踏入**Athenas-Oracle**的殿堂，这是一个创新的AI伴侣，旨在助您驾驭庞大的文档海洋，借助古代智慧女神雅典娜之智慧和洞察力。🌟 利用人工智能和自然语言处理的最新进展，雅典娜神谕能让用户轻松、精确地深入探索学术论文、报告及任何形式的文本内容，发掘前所未有的深层见解。

## 📜 教程
- [![Medium](https://img.shields.io/badge/Medium-12100E?style=&logo=medium&logoColor=white)](https://medium.com/@limyoonaxi) : [RAG Tutorial: Start from Athenas-Oracle [1]](https://medium.com/@limyoonaxi/rag-tutorial-start-from-athenas-oracle-1-fb9c7b77b0f1)
- [New Article is out now!!!][![Medium](https://img.shields.io/badge/Medium-12100E?style=&logo=medium&logoColor=white)](https://medium.com/@limyoonaxi) : [RAG Tutorial: Start from Athenas-Oracle [2]](https://medium.com/@limyoonaxi/rag-tutorial-start-from-athenas-oracle-2-feda0b528588)

- [New Article is out now!!!][![Medium](https://img.shields.io/badge/Medium-12100E?style=&logo=medium&logoColor=white)](https://medium.com/@limyoonaxi) : [RAG Tutorial: Start from Athenas-Oracle [3]](https://medium.com/@limyoonaxi/rag-tutorial-start-from-athenas-oracle-3-1e48876f9c01)
- [New Article is out now!!!][![Zhihu](https://img.shields.io/badge/Zhihu-3982f7?style=&logo=zhihu&logoColor=white)](https://www.zhihu.com/people/freax-23/posts) [Athena's Oracle: 掌握 RAG 技术的理想跳板](https://zhuanlan.zhihu.com/p/686693403)
- [New Article is out now!!!][![Zhihu](https://img.shields.io/badge/Zhihu-3982f7?style=&logo=zhihu&logoColor=white)](https://www.zhihu.com/people/freax-23/posts) [RAG 教程: 从Athenas-Oracle开始 [1]](https://zhuanlan.zhihu.com/p/689013625)
- [New Article is out now!!!][![Zhihu](https://img.shields.io/badge/Zhihu-3982f7?style=&logo=zhihu&logoColor=white)](https://www.zhihu.com/people/freax-23/posts) [RAG 教程: 从Athenas-Oracle开始 [2]](https://zhuanlan.zhihu.com/p/689013764)

## 🎉 特色亮点
我们非常高兴地宣布推出一项全新特色，旨在提升您的文档分析和知识获取体验：🚀 从 Awesome Paper 列表批量下载 🚀。

📚 **批量下载**：自动从 GitHub 上的 Awesome Paper 列表中提取 arXiv 链接，并批量下载论文到您的本地系统。

🔧 **RAG 知识库构建**：利用批量下载的论文轻松构建或扩充您的 RAG（检索增强生成）知识库，为深入研究和机器学习模型提供坚实支持。

## 功能 🚀✨

- **直观的查询处理**：向Athenas-Oracle低语您的疑问，让它引导您发现所求的智慧，充分利用您整个文档库的深度。📖
- **多文档智能**：能够解析和理解跨多个文档的信息层次，Athenas-Oracle织出的答案不仅精确无比，还极为全面。🌐
- **互惠排名融合**：通过一系列复杂的算法操作，将来自不同来源的最相关文档聚集一处，确保为您提供最贴切且丰富的知识。🔍
- **Streamlit 集成**：通过用户友好的界面进入Athenas-Oracle的殿堂，让复杂的文档分析艺术变得触手可及。💻

## 如何工作 🔮

Athenas-Oracle通过融合多种尖端技术和方法论施展其魔法：

1. **文档检索**：采用先进的向量搜索技术，根据您的查询内容挖掘出最相关的文档。🗂️
2. **问答**：利用最新的语言模型智慧，从识别出的文档中生成对您问题的精准答案。💡
3. **互惠排名融合**：集合多个文档的搜索结果，确保赋予您的知识是最相关且丰富的。📊

## 开启您的探索之旅 🌟

要与Athenas-Oracle一同开启您多个文档的搜索结果，确保赐予您的知识是最相关和丰富的。📊

## 开启您的探索之旅 🌟

要与Athenas-Oracle一同踏上旅程，请遵循这些神秘步骤：

1. **克隆仓库**：从我们的GitHub圣地召唤Athenas-Oracle的最新版本。
   ```
   git clone https://github.com/H-Freax/Athenas-Oracle.git
   ```
2. **安装依赖**：进入神圣的项目目录，召唤所需的Python药剂。
   ```
   cd Athenas-Oracle
   pip install -r requirements.txt
   ```

## 运行应用 🚴‍♂️

在您设置好所有要求之后，还有一个关键步骤，然后您就可以释放Athenas-Oracle的全部力量：添加您的OpenAI API密钥。这个密钥就像是允许您访问Athenas-Oracle所依赖的广泛智能的秘密密码。

### 获取OpenAI API密钥 🔑

您有两种管理OpenAI API密钥的选项：

1. **添加到Streamlit秘密文件**：您可以通过将API密钥添加到`.streamlit/secrets.toml`文件中来安全存储您的API密钥。这样，它就会被应用程序自动捕捉到，您就不必再担心它了。以下是文件中的格式：

   ```toml
   OPENAI_API_KEY = "sk-yourapikeyhere"
   ```

   确保将`"sk-yourapikeyhere"`替换为您实际的OpenAI API密钥。

2. **用户在侧边栏中输入**：如果您愿意，您也可以设置应用程序，让用户每次访问页面时在侧边栏中输入他们的OpenAI API密钥。这种方法更灵活，适合于您想与他人共享您的创作，但又不想直接分享您的API密钥时使用。

### 唤醒Athenas-Oracle 🌞

有了安全放置的API密钥，您就准备好激活Athenas-Oracle了。打开您的终端或命令提示符，导航到您存储Athenas-Oracle的文件夹，并运行以下命令：

```bash
streamlit run app.py
```

## 与Athenas-Oracle共启探索之旅 🌍✨
一旦Athenas-Oracle苏醒，您就站在未知知识的门槛上。遵循以下步骤，通过古老智慧导航：

**输入神圣的arXiv链接** 🔗
通过在指定字段中输入神圣的arXiv链接开始您的旅程。这个链接是您解锁存储在arXiv图书馆中的广泛学术知识的钥匙。不用害怕，Athenas-Oracle旨在通过这一过程轻松引导您。

**下载仪式** 📥
提交您的arXiv链接后，Athenas-Oracle将开始下载仪式。观察文档如何从数字以太中被召唤到神谕的领域。"下载成功"的消息将预示着这一步骤的完成，标志着文档准备好进入下一个启示阶段。

**生成神圣的嵌入** 🔮
现在文档已经处于Athenas的掌握之中，Oracle将开始神秘的嵌入生成过程。这些嵌入是文档知识的精华，转化为Athenas-Oracle能够理解和利用于其智慧赋予努力的形式。

**选择羊皮纸** 📜
随着嵌入的创建，您现在必须选择文件——您希望查询的特定知识羊皮纸。这一步骤至关重要，因为它决定了Athenas-Oracle将提供的智慧来源。明智地选择，并让您的直觉引导您。

**与Athenas开始对话** 🗣️
最后，您准备好与Athenas-Oracle进行对话了。将您的问题、询问和思考输入到Oracle的界面中。Athenas-Oracle，现已完全调整到所选文档的知识上，将以从文本深处汲取的洞察、答案和指导回应。

## 依赖项 📜🔗

携带这些魔法工具开始您的冒险：

- **LangChain**：无缝将语言模型编织到应用的结构中。
- **FAISS**：快速高效地搜索和聚类密集向量，就像在宇宙的干草堆中找到针一样。
- **OpenAI**：调用强大的AI模型中的Oracle。
- **arxiv-downloader**：一个备受尊敬的工具，用于从arXiv的大厅召唤文档，展示了文档检索的无界领域。
- **LLMStreamlitDemoBasic**：展示如何使用Streamlit和大型语言模型创建交互式演示的灵感之源，提升凡人在Athenas-Oracle内的体验。

## 贡献与反馈 📬🤝

Athenas-Oracle随着社区的贡献和洞察而茁壮成长。无论是精炼算法、增强神圣界面还是扩展文档语料库，我们对您的智慧贡献非常珍视。随时克隆仓库，用您的增强注入其中，并提交一个拉取请求。

## 致谢 🌺

- **arXiv-downloader**：衷心感谢[arXiv-downloader](https://github.com/braun-steven/arxiv-downloader)提供轻松访问学术论文的卷轴，帮助Athenas-Oracle的知识库持续增长。
- **LLMStreamlitDemoBasic**：我们的感激之情延伸至[LLMStreamlitDemoBasic](https://github.com/nimamahmoudi/LLMStreamlitDemoBasic)的圣贤们，他们为集成伟大的语言模型与Streamlit奠定了神秘的基础，从而增强了通过Athenas-Oracle的互动朝圣之旅。

## 许可证 📖

Athenas-Oracle在MIT许可证下赠予世界。有关更多细节，请查阅许可证文件。
