---
author: ["Yan Liang"]
title: "什么是向量数据库？"
date: "2024-11-29"
description: "向量数据库的简介，包括它们如何利用嵌入处理非结构化数据，以及在人工智能驱动任务中的应用。"
CNtags: ["机器学习", "向量数据库"]
CNcategories: ["机器学习", "数据工程"]
CNseries: ["机器学习"]
ShowToc: true
TocOpen: true
---
# **1 - 引言**

## **1.1 - 什么是向量数据库？**

向量数据库是一种将信息存储为向量的数据库，能够高效且灵活地处理数据。从计算的角度来看，"向量"就是一个有序的数字序列，有点像列表。通过将文本、图像或其他类型的数据转换为向量，计算机可以快速执行例如比较两段信息相似度的任务。如果你想深入了解嵌入的概念，可以查看[文本嵌入初学者指南](https://www.deepset.ai/blog/the-beginners-guide-to-text-embeddings)。

在向量数据库中，数据被表示为**高维向量**——这是一种数学结构，用于捕捉数据的重要特征或属性。根据复杂程度，这些向量可能有几十、几百甚至上千个维度。这些向量通常是通过**嵌入函数**将原始数据（如文本、图像或音频）转换为数值格式而生成的。这种转换通常涉及机器学习模型或其他特征提取技术。

## **1.2 - 向量数据库是如何工作的？**

### **1.2.1 - 理解嵌入（Embeddings）**

为了理解向量数据库的工作原理，首先需要掌握**嵌入**的概念。

非结构化数据，如文本、图像和音频，缺乏固定结构，这使得传统数据库难以处理这些数据。为了使这些数据更易于人工智能和机器学习使用，需要将其通过嵌入转换为数字。

**嵌入**就像给每个数据打上独特的“指纹”，以捕捉其核心含义。这个“指纹”帮助计算机更有效地比较和理解这些数据。想象一下，把整本书转换为一组关键数字，但仍然能够传达它的精髓。

嵌入通常由专门设计用于特定任务的神经网络生成。例如，**词嵌入**（word embeddings）将单词转换为向量，使得含义相近的单词在虚拟空间中靠得更近。这种转换让计算机可以更容易识别模式和关系。

简而言之，嵌入将非结构化数据转换为机器学习模型能够轻松处理的形式，帮助它们更有效地找到关系和模式。

**文本嵌入示例**

例如，嵌入在语义搜索中非常有用，帮助找到语义相似的文档。一种简单的文本嵌入形式是**独热编码（one-hot encoding）**，它将单词转换为向量。如果我们有四个唯一的单词，每个单词由长度为四的向量表示，其中大部分是零，只有一个位置的值为“1”，表示其独特的位置。这种技术称为“稀疏”嵌入。

![Text Embedding Example](assets/images/cn_attachments/text_embedding.png)

以下是使用 **BERT**（一种流行的自然语言处理技术）将一段文本转换为向量嵌入的示例：

```python
import torch
from transformers import BertTokenizer, BertModel

# 初始化tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 准备文本
text = "Hello, how are you?"
encoded_input = tokenizer(text, return_tensors='pt')

# 生成嵌入
with torch.no_grad():
    output = model(**encoded_input)
embeddings = output.last_hidden_state

# 输出嵌入
print(embeddings)
```

输出是一个**张量（tensor）**，本质上是一个多维数组。张量用于表示更复杂的数据结构，如句子中的单词及其多个关联特征。更多关于张量的详细信息可以参见此[参考](https://www.linkedin.com/pulse/machine-learning-basics-scalars-vectors-matrices-tensors-prasad)。

### **1.2.2 - 向量数据库的应用场景**

向量数据库广泛应用于**自然语言处理（NLP）**、**计算机视觉（CV）**和**推荐系统（RS）**等领域。

以下是一些常见的应用示例：

- **图像相似搜索**：找到与给定图像视觉上相似的图像。
- **文档匹配**：检索在主题或情感上相似的文档。
- **产品推荐**：根据特征或评分推荐类似的产品。
- **音乐搜索**：找到符合某特定旋律的歌曲。
- **内容发现**：发现与某一主题或视角相关的文章。

### **1.2.3 - 使用向量数据库的好处**

- **快速且准确的相似度搜索**：与传统数据库不同，向量数据库使用相似性度量来查找相关数据，这对于像文本或图像这样的非结构化数据特别有用。
- **处理非结构化数据**：它们将非结构化数据转换为向量，从而更容易获得洞察。
- **能够扩展到大数据集**：无论数据量是几GB还是几PB，向量数据库都能有效扩展。
- **速度优化**：这些数据库为快速查询而设计，对于人工智能和机器学习模型中的实时应用至关重要。

与依赖于精确匹配的传统关系型数据库相比，向量数据库擅长通过相似性搜索找到“最接近的匹配”。

## **1.3 - 向量数据库的搜索技术**

向量数据库使用近似最近邻(ANN)搜索技术，这些技术包括哈希(hashing)和基于图的搜索方法，以快速定位相似的向量S。

# **2 - 向量数据库在实践中的应用示例**

![向量数据库在LLM应用中的使用案例](https://images.datacamp.com/image/upload/v1694511771/image5_fd4c7efb1b.png)

当向量数据库与**大型语言模型(LLM)**结合时，其威力非常强大。以下是它们增强LLM能力的三种方式：

1. **从知识库中检索相似信息**：
    
    - **步骤**：将一个问题转换为特定的LLM嵌入，从向量数据库中检索相关的上下文以构建LLM提示。
    - **应用场景**：文档发现、聊天机器人、问答系统。
    - **主要好处**：这种方法避免了将敏感数据用于训练，同时能以较低的成本、几乎实时地更新知识库。
2. **从聊天记录中检索上下文**：
    
    - **步骤**：将用户的查询嵌入并与向量数据库中的存储嵌入进行匹配。系统提供相关响应并更新对话历史。
    - **应用场景**：客户支持机器人、持续且一致的用户交互。
    - **主要好处**：有助于克服LLM的token长度限制，并保持对话的连续性。
3. **缓存之前的查询和响应**：
    
    - **步骤**：如果一个问题之前已经被问过，系统可以快速从缓存中响应，而无需再次调用LLM。
    - **应用场景**：提高信息检索效率、常见问题解答和聊天机器人操作。
    - **主要好处**：节省计算资源，加快响应时间，降低LLM调用成本。

# **3 - 常见的向量数据库**

如果你想深入了解并使用向量数据库，以下是一些著名的选择：

- **Chroma**：以开发者友好、易于集成为特色。
- **Qdrant**：开源，针对高性能相似性搜索进行了优化。
- **Milvus**：一种流行的选择，专为非结构化数据而设计，具有良好的扩展性。
- **Azure AI Search**：基于云的向量搜索服务，能够与微软的Azure生态系统进行强大集成。

市面上有许多向量数据库，每个都有自己的优势和特定的使用场景。想了解更全面和最新的比较，可以查看 [这篇详细的向量数据库比较](https://superlinked.com/vector-db-comparison)。

# **结论**

向量数据库正在改变我们存储、处理和检索非结构化数据的方式，为增强人工智能和机器学习应用提供了强大的工具。它们将复杂的信息转换为向量，使数据发现更直观，实现实时交互能力，成为现代AI模型（如LLM）的完美补充。从语义搜索到推荐系统，向量数据库提供了一种传统数据库难以企及的灵活性和可扩展性。

# **参考文献**

- [文本嵌入初学者指南 | deepset](https://www.deepset.ai/blog/the-beginners-guide-to-text-embeddings)
- [5种最佳向量数据库 | 列表及示例 | DataCamp](https://www.datacamp.com/blog/the-top-5-vector-databases)
- [3种方法让向量数据库提升LLM用例 (linkedin.com)](https://www.linkedin.com/pulse/3-ways-vector-databases-take-your-llm-use-cases-next-level-mishra/)
- [什么是向量数据库？| 向量数据库综合指南 | Elastic](https://www.elastic.co/what-is/vector-database)