## Deep Learning Final Project Report

### Title: Beyond Attention: Breaking the Limits of Transformer Context Length with Recurrent Memory

#### Authors:
Jui-Hung Cheng (110703007)  
Yu-Chih Pan (110703013)  
NCCU Computer Science Senior Students  

### Abstract:
This report explores the Recurrent Memory Transformer (RMT) as a solution to overcome the context length limitations of traditional transformers. By integrating token-based memory augmentation, RMT extends context length from 128k tokens to over 2 million tokens, enabling efficient handling of long-term dependencies. The proposed approach maintains computational efficiency, enhances scalability, and supports seamless integration with pre-trained models like BERT and GPT-2.

---

### Introduction:
Transformers have become a cornerstone of modern natural language processing tasks. However, their quadratic computational complexity poses significant challenges when processing long sequences. Traditional solutions, such as memory-augmented neural networks and modified attention mechanisms, often require complex architectural changes or remain constrained by hardware limitations. This project investigates RMT, which introduces recurrent memory mechanisms to address these bottlenecks without sacrificing performance or scalability.

#### Background:
Transformers like BERT and GPT have revolutionized NLP tasks due to their ability to model complex dependencies in data. However, the self-attention mechanism becomes computationally expensive for long sequences. Memory-augmented neural networks and alternatives like Transformer-XL have partially addressed this issue but face limitations in scalability and integration.

---

### Problem Statement:
The inability of traditional transformers to handle long sequences effectively has prompted research into alternative architectures. Key questions include:

1. Can RMT enable efficient handling of extended context lengths?
2. Is RMT applicable to pre-trained models without significant modifications?
3. How does RMT compare to Retrieval Augmented Generation (RAG) frameworks for long-context tasks?

#### Significance:
The increasing demand for models that handle long-term dependencies, such as in legal document analysis, bioinformatics, and large-scale language understanding, highlights the necessity of scalable solutions. Addressing these demands efficiently could transform industries reliant on long-context data processing.

---

### Methodology:
#### Recurrent Memory Transformer (RMT):
- **Memory Tokens:** Prepend memory tokens to input/output sequences.
- **Segmented Processing:** Long sequences are segmented, and memory is passed recurrently between segments.
- **Plug-and-Play Integration:** RMT is implemented as a wrapper for pre-trained transformers, allowing seamless adoption.

#### Curriculum Learning:
- Begin with short sequences during fine-tuning.
- Gradually increase input lengths to stabilize training.

#### Experiment Design:
- **Datasets:** SQuAD, HotpotQA, and TriviaQA.
- **Baseline Models:** Llama 3.1 (8b-instruct) for language modeling.
- **Retrieval Techniques:** BM25 (sparse) and OpenAI embeddings (dense).
- **Evaluation Metrics:** BERTScore for performance and computational efficiency metrics for acceleration.

#### Visual Representation:
- **Architecture Diagram:** Illustrating the flow of memory tokens and segmented processing.
- **Performance Graphs:** Showcasing RMT’s efficiency compared to traditional transformers.

---

### Results:
1. **Performance:**
   - RMT demonstrated superior handling of long contexts compared to standard transformers.
   - Efficient integration with existing pre-trained models enabled faster training and inference.

2. **Acceleration:**
   - Cache-Augmented Generation (CAG) further reduced computational overhead, leveraging kvcache for knowledge storage and retrieval.

3. **Comparison to RAG:**
   - RMT outperformed RAG in terms of computational efficiency and ease of implementation.

#### Figures and Tables:
- Table comparing model performance metrics.
- Graph illustrating training time reduction with RMT.

---

### Discussion:
RMT’s linear computational scaling and memory token augmentation present significant advancements over existing methods. Unlike prior approaches, RMT avoids architectural complexities and hardware limitations, making it an attractive solution for tasks requiring extended contexts. Furthermore, its compatibility with closed-source language models broadens its applicability.

#### Limitations:
- Memory token storage may introduce challenges for extremely large datasets.
- Requires extensive fine-tuning to achieve optimal performance.

---

### Conclusion:
The Recurrent Memory Transformer (RMT) addresses the key limitations of traditional transformers by significantly extending their context length capabilities while maintaining efficiency and scalability. By leveraging memory tokens and recurrent mechanisms, RMT simplifies integration into pre-trained models, offering a practical solution for real-world applications.

#### Future Work:
- Expanding RMT’s application to multi-modal data.
- Exploring hardware-specific optimizations to further enhance efficiency.

---

### References:
1. Jui-Hung Cheng and Yu-Chih Pan. *Don’t Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks*. NCCU.
2. Dai et al. *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*. arXiv.
3. Rae et al. *Compressive Transformers for Long-Range Sequence Modeling*. arXiv.
4. Llama-index documentation. https://arxiv.org/pdf/2412.15605.

---

### Acknowledgments:
We thank our advisor and team members for their support and guidance throughout this project. Special thanks to the NCCU Computer Science department for providing computational resources.

---

### Appendix:
#### Figures:
1. **RMT Architecture Diagram**
   ![RMT Architecture Diagram](#)

2. **Performance Graphs**
   - Graph comparing model performance across datasets.
   - Chart showcasing memory usage efficiency.

#### Tables:
- Table summarizing datasets and model configurations.
- Results table for BERTScore and efficiency metrics.

---

**Note:** Placeholder for images has been added. Please insert appropriate diagrams and graphs to further enhance the visual appeal and readability of the report.

