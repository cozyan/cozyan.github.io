---
author: ["Yan Liang"]
title: "What is Vector DB?"
date: "2024-11-29"
description: "An introduction to vector databases, their use of embeddings to handle unstructured data, and their applications in AI-driven tasks."
tags: ["Machine Learning", "Vector DB"]
categories: ["Machine Learning", "Data Engineering"]
series: ["Machine Learning"]
ShowToc: true
TocOpen: true
---
# **1 - Introduction**

## **1.1 - What is a vector database?**

A vector database is a type of database that stores information as vectors, allowing for efficient and flexible data processing. In computational terms, a "vector" is simply an ordered sequence of numbers—kind of like a list. By converting text, images, or other types of data into vectors, computers can quickly perform tasks like finding how similar two pieces of information are. If you’d like a deeper dive into the concept of embeddings, check out [The Beginner's Guide to Text Embeddings](https://www.deepset.ai/blog/the-beginners-guide-to-text-embeddings).

In a vector database, data is represented as **high-dimensional vectors**—mathematical structures that capture important features or attributes of the data. Depending on the complexity, these vectors may have dozens, hundreds, or even thousands of dimensions. The vectors are usually generated using an **embedding function** that transforms raw data—such as text, images, or audio—into a numerical format. This transformation often involves machine learning models or other feature extraction techniques.

## **1.2 - How does a vector database work?**

### **1.2.1 - Understanding Embeddings**

To understand how a vector database works, it's important to grasp the concept of **embeddings**.

Unstructured data, such as text, images, and audio, doesn't have a fixed structure, which makes it difficult to use with traditional databases. To make this data more usable for artificial intelligence and machine learning, it's transformed into numbers using embeddings.

**Embedding** is like giving each piece of data a unique "fingerprint" that captures its core meaning. This fingerprint helps computers compare and understand these data pieces more effectively. Think of it like turning an entire book into a set of key numbers that still conveys its essence.

Embeddings are often created using special neural networks designed for a specific task. For example, **word embeddings** convert words into vectors, placing words with similar meanings close together in a virtual space. This transformation allows computers to identify patterns and relationships.

In short, embeddings convert unstructured data into something that machine learning models can easily work with, helping them find relationships and patterns more efficiently.

**An Example of Text Embedding**

For instance, embeddings are helpful for tasks like semantic search, where you want to find documents similar in meaning. A simple form of text embedding is **one-hot encoding**, which turns words into vectors. If we have four unique words, each word is represented by a vector of length four, containing mostly zeros except for a single "1" that indicates its unique position. This technique is known as a "sparse" embedding.

![Text Embedding Example](assets/images/attachments/text_embedding.png)

Here's an example of transforming a piece of text into a vector embedding using **BERT**, a popular technique in natural language processing:

```python
import torch
from transformers import BertTokenizer, BertModel

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Prepare the text
text = "Hello, how are you?"
encoded_input = tokenizer(text, return_tensors='pt')

# Generate embeddings
with torch.no_grad():
    output = model(**encoded_input)
embeddings = output.last_hidden_state

# Output the embeddings
print(embeddings)
```

The output is a **tensor**, which is essentially a multi-dimensional array. Tensors are used to represent more complex data structures, like words in a sentence with multiple associated features. For more details about tensors, see this useful [reference](https://www.linkedin.com/pulse/machine-learning-basics-scalars-vectors-matrices-tensors-prasad).

### **1.2.2 - Use Cases for Vector Databases**

Vector databases are used in a variety of applications involving **natural language processing (NLP)**, **computer vision (CV)**, and **recommendation systems (RS)**.

Here are some common examples of their use:

- **Image Similarity Search**: Finding images visually similar to a given image.
- **Document Matching**: Retrieving documents similar in theme or sentiment.
- **Product Recommendations**: Suggesting products similar to another based on features or ratings.
- **Music Search**: Finding songs that match a particular tune.
- **Content Discovery**: Discovering articles related in theme or perspective to a given one.

### **1.2.3 - Benefits of Using a Vector Database**

- **Fast and Accurate Similarity Search**: Unlike traditional databases, vector databases use similarity measures to find relevant data. This is especially useful for unstructured data like text or images.
- **Handling Unstructured Data**: They transform unstructured data into vectors, making it easier to derive insights.
- **Scalable with Large Datasets**: Vector databases are built to scale effectively with large datasets, whether gigabytes or petabytes of data.
- **Optimized for Speed**: These databases are designed for rapid querying, which is essential for real-time applications such as AI and machine learning models.

Compared to traditional relational databases that rely on exact matches, vector databases excel at finding "closest matches" through similarity searches.

## **1.3 - Search Techniques for Vector Databases**

Vector databases use **Approximate Nearest Neighbor (ANN)** search techniques, which can include methods like **hashing** and **graph-based searches** to quickly locate similar vectors.

# **2 - Examples of Vector Databases in Practice**

![Use-cases of vector database in LLM applications](https://images.datacamp.com/image/upload/v1694511771/image5_fd4c7efb1b.png)

Vector databases are powerful when combined with **Large Language Models (LLMs)**. Here are three ways they can enhance LLM capabilities:

1. **Retrieving Similar Information from a Knowledge Base**:
    
    - **Steps**: A question is transformed into an LLM-specific embedding, which retrieves relevant context from a vector database to construct an LLM prompt.
    - **Use Cases**: Document discovery, Chatbots, Q&A.
    - **Key Benefits**: This approach avoids using sensitive data for training while providing a cheaper, almost real-time update to the knowledge base.
2. **Contextual Retrieval from Chat History**:
    
    - **Steps**: A user's query is embedded and matched against stored embeddings in the vector database. The system then provides relevant responses while updating the conversation history.
    - **Use Cases**: Customer support bots, persistent and consistent user interactions.
    - **Key Benefits**: Helps overcome LLM token length limits and maintains conversation continuity.
3. **Caching Previous Queries and Responses**:
    
    - **Steps**: If a question has been previously asked, the system can quickly respond from cache without needing to invoke the LLM again.
    - **Use Cases**: Improves efficiency in information retrieval, FAQs, and chatbot operations.
    - **Key Benefits**: Saves computation resources and speeds up response time, reducing LLM invocation costs.
# **3 - Popular Vector Databases**

If you're looking to dive into vector databases yourself, here are some of the well-known options:

- **Chroma**: Known for being developer-friendly with easy integration.
- **Qdrant**: Open-source and optimized for high-performance similarity search.
- **Milvus**: A popular option, scalable and designed specifically for unstructured data.
- **Azure AI Search**: A cloud-based vector search service with powerful integration into Microsoft’s Azure ecosystem.

There are many vector databases out there, each with its strengths and particular use cases. For a more comprehensive and up-to-date comparison, you can check out [this detailed vector database comparison](https://superlinked.com/vector-db-comparison).

# **Conclusion**

Vector databases are transforming how we store, process, and retrieve unstructured data, providing a powerful tool for enhancing AI and machine learning applications. Their ability to convert complex information into vectors allows for more intuitive data discovery and real-time interaction capabilities, making them a perfect complement to modern AI models like LLMs. From semantic searches to recommendation systems, vector databases offer a level of flexibility and scalability that traditional databases struggle to provide.

# **References**

- [The Beginner’s Guide to Text Embeddings | deepset](https://www.deepset.ai/blog/the-beginners-guide-to-text-embeddings)
- [The 5 Best Vector Databases | A List With Examples | DataCamp](https://www.datacamp.com/blog/the-top-5-vector-databases)
- [3 Ways Vector Databases Take Your LLM Use Cases to the Next Level (linkedin.com)](https://www.linkedin.com/pulse/3-ways-vector-databases-take-your-llm-use-cases-next-level-mishra/)
- [What is a Vector Database? | A Comprehensive Vector Database Guide | Elastic](https://www.elastic.co/what-is/vector-database)