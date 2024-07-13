
# Article_Research_Tool Based_on_OpenSouceLLMs


## Overview
Article_Research_Tool is an advanced RAG application designed for efficient extraction and analysis of information from news articles using Python, Streamlit, LangChain, FAISS, and open-source language models.


## Problem It’s Trying to Solve

 - Many researchers, journalists, and everyday users face the challenge of extracting specific information from lengthy articles or websites. This process is time-consuming as they need to read through entire articles or web pages to find the details they seek. 
 - For instance, researchers might need to understand a specific topic within an article, while consumers may want to find particular policy rules or product features before making a purchase decision. 
 - The News Research Tool aims to streamline this process by providing precise and concise answers to user queries, saving valuable time and effort.




## Features

#### 1. User-Friendly Interface:
- A clean and intuitive Streamlit interface.
- Sidebar input for up to three news article URLs.

#### 2. Dynamic Data Loading:
- Utilizes WebBaseLoader for scraping and parsing articles.
- Supports dynamic web pages with customizable class names for precise data extraction.

#### 3. Advanced Embedding:
- Employs HuggingFaceEmbeddings with the sentence-transformers all-MiniLM-l6-v2 model for robust text embeddings.
- Configurable model parameters and encoding options for optimized performance.

#### 4. Vector Store Integration:
- Integrates FAISS for efficient vector storage and retrieval.

#### 5. Open-Source Language Models:
- Utilizes mixtral-8x7b-32768 open-source LLM for question answering and information retrieval, ensuring transparency and flexibility.

## Installation

#### 1. Clone the repository:

```
git clone https://github.com/theshubh007/Article_Research_Tool_Based_on_OpenSouceLLMs.git
```

#### 2. Create a virtual environment and activate it:

```
python -m venv venv
.\.venv\Scripts\activate
```

#### 3. Install the required packages:

```
pip install -r requirements.txt
```

#### 4. Create a .env file in the root directory and add your API key:

```
GROQ_API_KEY=your_groq_api_key

```

#### 5. Run the Streamlit application:

```
streamlit run app.py

```







## Components

- Streamlit: Provides the interactive web interface.
- LangChain: Handles document loading, text splitting, and prompt management.
- FAISS: Manages the vector storage and retrieval of document embeddings.
- HuggingFaceEmbeddings: Generates text embeddings using a pre-trained model.
- ChatGroq: Provides advanced language modeling for question answering.

## Future Enhancements
- Support for Additional Languages: Extend the tool to support news articles in multiple languages.
- Improved Scraping Capabilities: Enhance the scraping logic to handle a wider variety of news websites.
- Enhanced Visualization: Add more visualization options for the retrieved data and analysis results.

## Contribution
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.


![rag3](https://github.com/theshubh007/Article_Research_Tool_Based_on_OpenSouceLLMs/assets/100220928/76c52171-284f-4da2-b7d9-6a9512904104)
