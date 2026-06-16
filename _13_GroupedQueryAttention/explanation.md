# Grouped Query Attention

Grouped Query Attention (GQA) is a technique used in transformer models to make attention **faster and more memory-efficient**. Instead of giving every attention head its own keys and values, several query heads **share the same keys and values**. This reduces the amount of information the model needs to store and process while keeping most of the performance of standard attention.

In a normal **Multi-Head Attention (MHA)** layer, every head has its own **query (Q)**, **key (K)**, and **value (V)** matrices. Each head learns to focus on different relationships in the input, but storing separate keys and values for every head requires a large amount of memory, especially when generating long sequences of text.

Grouped Query Attention simplifies this by keeping separate **query heads** but grouping them so they share the same **key** and **value** heads. For example, instead of eight query heads each having their own keys and values, the eight query heads might share only two sets of keys and values. The queries can still look for different information, but they search using the same shared references.

You can think of it like a classroom. Imagine eight students (the query heads) are working on different questions. Instead of each student having their own textbook, every group of four students shares one textbook. Each student can still search for different answers, but fewer books are needed. This saves space and makes the process more efficient.

The main advantage of GQA is that it reduces the size of the **key-value cache**, which stores information from previously processed tokens during text generation. A smaller cache means the model uses less memory and can generate text faster, particularly for long conversations or documents.

Modern large language models often use Grouped Query Attention because it provides a good balance between speed and accuracy. It performs much better than sharing a single key and value across all heads (Multi-Query Attention) while requiring much less memory than giving every head its own separate keys and values (Multi-Head Attention). As a result, GQA has become a common design choice in efficient transformer architectures.
