# **Code Tutorial**

Creating Athenas-Oracle: A Practical Guide with a Tech Twist

Let's break down the making of Athenas-Oracle, a tool that's all about turning data into knowledge, with a touch of AI magic and a lot of tech savvy. Imagine it's like building a super smart assistant that can read through piles of academic papers, understand what's what, and even chat about it. Here's how we do it, step by step, ensuring we sprinkle in those professional terms without losing the plot.

### Step 1: GitHub Link Parsing

First off, we have our system look at GitHub links. It's like giving it a treasure map where X marks the spot for coding projects and software gems. The goal here is simple: find those links, understand them, and get ready to dive deep into the code and papers they point to. It's the first step in a journey of discovery and learning.

### Step 2: arXiv Download

Next up, Athenas-Oracle rolls up its digital sleeves and starts pulling in research papers from arXiv. Think of arXiv as a massive library that's all online, full of the latest research papers. By knowing which GitHub projects to look at, Athenas-Oracle can grab the right papers, making sure it's feeding its brain with all the right info.

### Step 3: Embedding PDF

With a stack of papers (well, digital ones), Athenas-Oracle then gets to work on turning these PDFs into something it can work with. This step is all about extracting text, which means pulling out all the words and numbers from these papers so the system can analyze and understand them. It's like translating a foreign language into one Athenas-Oracle speaks fluently.

### Step 4: GPT-4 API Integration

Now for the cool part—bringing in GPT-4, a state-of-the-art AI from OpenAI. This integration is like giving Athenas-Oracle a superpower. Suddenly, it can understand natural language way better, answer questions, and even come up with responses that sound eerily human. GPT-4 is the brain boost that makes all the difference, turning raw data into conversations.

### Step 5: LLM Helper via Langchain to Agent

To make sure Athenas-Oracle doesn't just understand data but can also use it smartly, we hook it up with a Large Language Model (LLM) helper via Langchain. This step is like having a wise advisor in the room, guiding the AI to better handle queries and sift through information. It's all about making the system smarter and more relevant to what users need.

### Step 6: Streamlit for UI

Finally, we want people to actually use this thing, right? That's where Streamlit comes in, letting us build a user interface that's not just functional but also kind of fun to use. It's the window into Athenas-Oracle's brain, where users can ask questions, see results, and interact with all that data it's been crunching. Imagine it as the friendly face of our super-smart system.

And there you have it—a whistle-stop tour of building Athenas-Oracle, packed with all the techy terms but hopefully still clear enough to see how all these pieces fit together. It's a bit like assembling a high-tech puzzle where each piece is crucial for the big picture, which, in this case, is about making sense of mountains of data and turning it into knowledge at our fingertips.

Sure, let's dive into the code. We'll walk through it step by step, explaining the key pieces as they align with the development steps we outlined earlier.

### Step 1: GitHub Link Parsing

In this initial phase, Athenas-Oracle scans through GitHub repository links to locate and earmark academic papers or relevant code repositories. This process is pivotal for identifying the source materials Athenas-Oracle will analyze and utilize.

### Import Statements

First, let's look at the import statements required for this step:

```python
import base64
import requests
import re

```

- `base64`: This module is essential for encoding and decoding operations in base64 format. GitHub's API returns content in base64 encoding, making this module crucial for decoding README files fetched from GitHub repositories.
- `requests`: A versatile HTTP library for Python, `requests` simplifies making HTTP requests to web servers, which is necessary for interacting with GitHub's API to fetch repository README files.
- `re`: The `re` module, standing for Regular Expressions, allows for efficient searching, matching, and manipulation of strings based on specific patterns. It's used here to identify arXiv links within the text of README files.

### Functions to Fetch and Parse Data

```python
def get_readme_contents(repo_url):
    """Fetch the README.md file's contents from a GitHub repository using GitHub's API."""
    user_repo = repo_url.replace("<https://github.com/>", "")
    api_url = f"<https://api.github.com/repos/{user_repo}/contents/README.md>"
    response = requests.get(api_url)
    if response.status_code == 200:
        content = response.json()['content']
        readme_contents = base64.b64decode(content).decode('utf-8')
        return readme_contents
    else:
        st.sidebar.error("Error: Unable to fetch README.md")
        return None

```

This function transforms the GitHub repository URL into a format compatible with GitHub's API endpoints. It then sends a request to fetch the content of the [README.md](http://readme.md/) file. If successful, it decodes the content from base64 to plain text and returns it.

```python
def extract_arxiv_links(readme_contents):
    """Extract all arXiv links from the README.md contents."""
    arxiv_links = re.findall(r'<https://arxiv.org/abs/[^\\s)]+>', readme_contents)
    return arxiv_links

```

`extract_arxiv_links` scans the text of the [README.md](http://readme.md/) file using a regular expression pattern to find all URLs that point to arXiv abstract pages. These links are then compiled into a list and returned.

### Step 2: arXiv Download

Having identified the relevant arXiv links, Athenas-Oracle proceeds to download the associated academic papers for further analysis.

### More Imports

```python
import arxiv_downloader.utils

```

- `arxiv_downloader.utils`: This module provides utility functions for interacting with arXiv. Specifically, it includes methods for converting arXiv URLs into paper IDs and for downloading the papers themselves.

### Download Function

```python
def download_arxiv_paper(link):
    """Download a paper from arXiv given its link."""
    arxiv_id = arxiv_downloader.utils.url_to_id(link)
    try:
        arxiv_downloader.utils.download(arxiv_id, "./pdf", False)
        st.sidebar.success(f"Downloaded: {link}")
    except Exception as e:
        st.sidebar.error(f"Failed to download {link}: {e}")

```

This function extracts the paper ID from the provided arXiv link, then attempts to download the paper using the `download` function from `arxiv_downloader.utils`. It saves the paper to a specified directory (here, "./pdf"). Success or failure feedback is provided through Streamlit's sidebar notifications.

### Step 3: Embedding PDFs

After downloading the desired arXiv papers, Athenas-Oracle proceeds to analyze and embed the documents, converting them into a format that facilitates efficient search and retrieval. This involves breaking down the PDFs into manageable chunks and generating embeddings for each segment. The embedding part is in `emded_pdf.py`

### Import Statements and Setup

```python
from langchain.document_loaders import PagedPDFSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

```

- `PagedPDFSplitter`: A class from `langchain` used to split PDF documents into individual pages or segments for easier handling.
- `RecursiveCharacterTextSplitter`: This utility helps to further break down the text into chunks of a specified size, managing overlaps for continuity in context.
- `OpenAIEmbeddings`: A feature from `langchain` that leverages OpenAI's API to create embeddings for text segments. Embeddings are essentially numerical representations that capture the essence of the text, making it searchable and comparable.
- `FAISS`: A highly efficient library for similarity search and clustering of dense vectors. `langchain` integrates FAISS to store and manage the generated embeddings, facilitating fast retrieval.
- `os`: A standard library module in Python, used here to interact with the file system, particularly for listing files in directories.

### Embedding Functions

```python
def embed_document(file_name, file_folder="pdf", embedding_folder="index"):
    file_path = f"{file_folder}/{file_name}"
    loader = PagedPDFSplitter(file_path)
    source_pages = loader.load_and_split()

    embedding_func = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        separators=["\\n\\n", "\\n", " ", ""],
    )
    source_chunks = text_splitter.split_documents(source_pages)
    search_index = FAISS.from_documents(source_chunks, embedding_func)
    search_index.save_local(folder_path=embedding_folder, index_name=file_name + ".index")

```

- `embed_document` function takes a PDF file, splits it into pages, then further into text chunks. It generates embeddings for these chunks using OpenAI's API and stores them in a FAISS index for fast retrieval. This index is saved locally, allowing the system to later search through the document efficiently.

```python
def embed_all_pdf_docs():
    pdf_directory = "pdf"
    if os.path.exists(pdf_directory):
        pdf_files = [file for file in os.listdir(pdf_directory) if file.endswith(".pdf")]
        if pdf_files:
            for pdf_file in pdf_files:
                print(f"Embedding {pdf_file}...")
                embed_document(file_name=pdf_file, file_folder=pdf_directory)
                print("Done!")
        else:
            raise Exception("No PDF files found in the directory.")
    else:
        raise Exception(f"Directory '{pdf_directory}' does not exist.")

```

- `embed_all_pdf_docs` scans a directory for PDF files and processes each one through `embed_document`. This batch process ensures all documents are ready for search and analysis.

```python
def get_all_index_files():
    index_directory = "index"
    if os.path.exists(index_directory):
        postfix = ".index.faiss"
        index_files = [file.replace(postfix, "") for file in os.listdir(index_directory) if file.endswith(postfix)]
        if index_files:
            return index_files
        else:
            raise Exception("No index files found in the directory.")
    else:
        raise Exception(f"Directory '{index_directory}' does not exist.")

```

- `get_all_index_files` lists all the index files in the `index` directory, effectively providing a catalog of processed documents. This function is essential for retrieving the list of documents available for searching.

Through these steps, Athenas-Oracle transforms raw PDFs into a searchable and analyzable format, significantly enhancing the platform's ability to deliver precise and relevant information based on user queries.

**Now you could imports for Embedding in app.py**

```python
import embed_pdf

```

### Step 4: GPT-4 API Integration

After preparing the documents for quick retrieval through embedding, the next step in Athenas-Oracle's development involves integrating OpenAI's GPT-4 to leverage its advanced natural language understanding and generation capabilities. This step is crucial for processing queries, generating responses, and enhancing the overall intelligence of the system.

### Import Statements and Setup

Before diving into the code, ensure you have access to the OpenAI API and have installed the necessary Python package:

```bash
pip install openai

```

For the code itself, you will need to import the OpenAI library. It's also good practice to ensure your API key is correctly configured in your environment or application settings:

```python
import openai
import os

```

- `openai`: The library provided by OpenAI for interacting with their API, including GPT-4.
- `os`: Used for accessing environment variables that store sensitive information like the OpenAI API key.

### Configuring the API Key

The API key for OpenAI is crucial for authenticating requests. It's recommended to store this key securely and load it into your application's environment. Here's a snippet on how you might set it up, assuming the key is stored in an environment variable:

```python
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key is not set in the environment variables.")

```

This code checks if the `OPENAI_API_KEY` environment variable is set and raises an error if not, ensuring your application does not proceed without the necessary API access.

### Using GPT-4 for Query Processing and Response Generation

With the API key configured, you can now use GPT-4 to process user inputs, queries, or any text data that requires understanding or response generation. Here's a basic example of how you might structure a call to the GPT-4 API to generate a response:

```python
def generate_response(prompt_text, max_tokens=100):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Specify the GPT-4 model, adjust as necessary
        prompt=prompt_text,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7
    )
    if response and response.choices:
        return response.choices[0].text.strip()
    else:
        return "Sorry, I couldn't generate a response."

```

- `prompt_text`: The input text or question for which you're generating a response.
- `max_tokens`: The maximum length of the generated response. Adjust this based on your application's needs.
- `engine`: The identifier for the OpenAI GPT model you're using. As of my last update, GPT-4's exact engine name might vary, so replace `"text-davinci-003"` with the correct GPT-4 engine identifier.
- The function sends a request to OpenAI's API with the provided prompt and settings, then formats and returns the response.

Integrating GPT-4 into Athenas-Oracle significantly boosts its ability to understand complex queries and generate informative, contextually relevant responses. This step is essential for transforming Athenas-Oracle into a powerful AI tool capable of engaging in meaningful interactions and providing valuable insights derived from vast amounts of data.

Securing the OpenAI API key is a critical aspect of integrating GPT-4 into Athenas-Oracle. Proper management of this key ensures secure and authorized access to OpenAI's API, maintaining the integrity and privacy of your application. Here are two secure methods to manage your OpenAI API key within your application:

### Option 1: Using Streamlit Secrets for API Key Management

Streamlit offers a built-in feature for managing secrets, such as API keys, which is both secure and convenient. To utilize this feature:

1. **Storing the API Key**: Place your OpenAI API key in the `.streamlit/secrets.toml` file within your application's directory. Here's the format you should use:
    
    ```toml
    OPENAI_API_KEY = "sk-yourapikeyhere"
    
    ```
    
    Ensure you replace `"sk-yourapikeyhere"` with your actual OpenAI API key. This file is encrypted by Streamlit when deployed, making it a secure place for sensitive information.
    
2. **Accessing the API Key in Your Application**: Streamlit automatically loads the contents of `secrets.toml` into `st.secrets`, making it accessible from your code like this:
    
    ```python
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    
    ```
    
    This method is particularly useful for deployment, as it keeps your key secure and hidden from the public.
    

### Option 2: Requesting API Key via User Input

For scenarios where you might be sharing your application with others and prefer not to store your API key in the application code or secrets file, you can opt to have users enter their API key:

1. **Implementing User Input for the API Key**: Add an input field in the Streamlit sidebar that allows users to input their OpenAI API key each time they use the application:
    
    ```python
    openai_api_key_input = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    
    ```
    
    This approach enhances flexibility and is ideal for shared applications. It's crucial to use the `type="password"` argument to ensure the key is obscured and treated securely.
    
2. **Setting the API Key from User Input**: Once the user provides their API key, set it for use in your application:
    
    ```python
    if openai_api_key_input:
        openai.api_key = openai_api_key_input
    else:
        st.sidebar.error("Please enter your OpenAI API key to proceed.")
    
    ```
    
    This conditional check ensures that the API key is set before attempting to use any of OpenAI's services.
    

By implementing one of these methods, you can effectively manage the OpenAI API key, maintaining the security of your application while leveraging the powerful capabilities of GPT-4 for natural language processing and generation tasks. This setup is a crucial part of integrating GPT-4 into Athenas-Oracle, ensuring that the application operates securely and efficiently.

## **Step 5: LLM Helper via Langchain to Agent**

Let's break down the integration of the LLM helper via Langchain into Athenas-Oracle, focusing on explaining each part in detail for beginners. We'll start with the foundational aspects and gradually work through the complex parts, ensuring clarity at each step. Given the complexity, we'll split this explanation across multiple parts, starting with the initial setup and the core functionalities.

### Initial Setup and Core Functions

### Import Statements

Before diving into the functionalities, we first need to import necessary libraries and modules. This is done at the top of your Python file (`llm_helper.py`).

```python
from typing import Optional, List

```

- `typing`: This module is used for type hinting, which helps with code readability and ensures that functions expect and return the correct types of arguments and values. Here, `Optional` and `List` are specific types from the `typing` module, used to annotate variable types.

```python
from langchain.agents import AgentExecutor

```

- `AgentExecutor`: A component from Langchain that manages the execution of AI agents. An AI agent, in this context, can be thought of as a piece of code that performs tasks or makes decisions based on input data.

```python
from langchain.chat_models import ChatOpenAI

```

- `ChatOpenAI`: This is a wrapper around OpenAI's ChatGPT model provided by Langchain. It allows you to easily integrate ChatGPT's conversational capabilities into your application.

```python
from langchain.prompts import ChatPromptTemplate

```

- `ChatPromptTemplate`: A utility for creating and managing prompt templates. Prompts are questions or statements that guide the AI in generating a response. Templates allow for dynamic insertion of content into these prompts based on the context of the conversation.

```python
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

```

- `RunnableMap` and `RunnablePassthrough`: These are part of Langchain's abstraction for creating chains of operations (or "runnables") that process data. `RunnableMap` maps input data through a series of functions or transformations, while `RunnablePassthrough` is a special kind of runnable that allows data to pass through it unchanged or with minimal processing.

```python
from langchain.schema.output_parser import StrOutputParser

```

- `StrOutputParser`: This parses the output from a runnable into a string. It's useful for converting complex data structures into a readable format.

```python
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage

```

- These classes represent different types of messages within a conversational context. `HumanMessage` for messages from the user, `AIMessage` for messages from the AI, and `SystemMessage` for messages generated by the system (like error messages or logs).

```python
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler

```

- `StreamlitCallbackHandler`: An integration between Langchain and Streamlit for handling callbacks. Callbacks are functions that get called in response to events (like a user clicking a button). This handler allows for updates to the Streamlit UI based on events in the Langchain processing pipeline.

```python
from operator import itemgetter

```

- `itemgetter`: A function from Python's `operator` module that's used for retrieving items from a collection (like a list or dictionary) based on a key or index. It's useful for extracting specific pieces of data from complex structures.

These imports are the building blocks for creating an interactive, AI-powered application with Athenas-Oracle. They enable the application to understand and process natural language queries, manage conversation flow, and integrate with powerful language models like GPT from OpenAI.

And then let's dive deeper into the `llm_helper.py` file, focusing on specific functions and their roles in the LLM helper setup. This continuation aims to provide a clear understanding of how Athenas-Oracle leverages Langchain and OpenAI's GPT models to enhance its conversational capabilities.

### Message Conversion Function

Located in `llm_helper.py`, the `convert_message` function transforms messages based on their roles in the conversation (user, assistant, or system) into corresponding Langchain message objects. This standardizes message handling within the system.

```python
def convert_message(m):
    if m["role"] == "user":
        return HumanMessage(content=m["content"])
    elif m["role"] == "assistant":
        return AIMessage(content=m["content"])
    elif m["role"] == "system":
        return SystemMessage(content=m["content"])
    else:
        raise ValueError(f"Unknown role {m['role']}")

```

- **Purpose**: This function takes a message `m` (a dictionary) as input, checks the message's role, and converts it into the appropriate Langchain message type (`HumanMessage`, `AIMessage`, or `SystemMessage`). This ensures that messages can be processed correctly in subsequent steps.

### Standalone Question Generation

The `get_standalone_question_from_chat_history_chain` function in `llm_helper.py` creates a Langchain runnable chain that condenses chat history into a standalone question, facilitating clearer and more focused AI responses.

```python
def get_standalone_question_from_chat_history_chain():
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    )
    return _inputs

```

- **Process Flow**:
    - `RunnablePassthrough.assign` takes chat history and applies `_format_chat_history` to format it into a readable string.
    - `CONDENSE_QUESTION_PROMPT` uses this formatted chat history to create a prompt for the GPT model, asking it to generate a standalone question.
    - `ChatOpenAI(temperature=0)` invokes the ChatGPT model with this prompt, generating a question that encapsulates the essence of the chat history.
    - `StrOutputParser()` ensures the output is a clean, usable string.

### Retrieval-Augmented Generation (RAG) Chain

The RAG chain combines document retrieval with question answering, enhancing responses with information fetched from specific documents. This is outlined in the `get_rag_chain` function.

```python
def get_rag_chain(file_name="Mahmoudi_Nima_202202_PhD.pdf", index_folder="index", retrieval_cb=None):
    # Function contents...

```

- **Key Components**:
    - **Document Retrieval**: Retrieves relevant document sections based on the query.
    - **Question Processing**: Condenses chat history into a clear, standalone question.
    - **Answer Generation**: Generates an answer using the GPT model, informed by the retrieved documents.

### RAG Fusion Chain

The `get_rag_fusion_chain` function builds on the RAG chain by incorporating multiple queries and documents, using a fusion technique to improve context gathering and response accuracy.

```python
def get_rag_fusion_chain(file_name="Mahmoudi_Nima_202202_PhD.pdf", index_folder="index", retrieval_cb=None):
    # Function contents...

```

- **Advanced Retrieval**: It employs a sophisticated method to combine results from multiple document queries, ensuring the AI model has a comprehensive context for generating responses.

### Document Formatting and Retrieval Callback

The `format_docs` function and the concept of a retrieval callback (`retrieval_cb`) are essential for preparing document content for processing and dynamically adjusting the retrieval logic.

```python
def format_docs(docs):
    # Formats documents into a structured string for processing.
    ...

```

- **Purpose**: Prepares the content of retrieved documents, making it ready for inclusion in prompts or further processing by the AI model.

The retrieval callback mechanism allows for custom adjustments to the document retrieval process based on dynamic factors, like the current state of the conversation or user preferences.

### Handling Multiple Files with `get_search_index`

The function `get_search_index` is crucial for initializing the search indexes for multiple files, allowing the system to retrieve information from a broader document base.

```python
def get_search_index(file_names: List[str], index_folder: str = "index") -> List[FAISS]:
    search_indexes = []
    for file_name in file_names:
        search_index = FAISS.load_local(
            folder_path=index_folder,
            index_name=file_name + ".index",
            embeddings=OpenAIEmbeddings(),
        )
        search_indexes.append(search_index)
    return search_indexes

```

- **Function Breakdown**:
    - `file_names`: A list of filenames that you want to create search indexes for.
    - `index_folder`: The directory where your index files are stored.
    - `search_indexes`: This list will hold the initialized FAISS indexes for each file.
    - Inside the loop, for each filename in `file_names`, it loads the corresponding FAISS index using `FAISS.load_local`. This process involves specifying the folder path, index name, and type of embeddings used (in this case, `OpenAIEmbeddings`).
    - Each loaded index is added to the `search_indexes` list, which is returned at the end of the function.

### Enhanced RAG Chain for Multiple Files: `get_rag_chain_files`

This function builds upon the basic RAG chain to support multiple documents, enhancing the system's ability to provide contextually rich responses.

```python
def get_rag_chain_files(file_names: List[str], index_folder: str = "index", retrieval_cb=None):
    vectorstores = get_search_index(file_names, index_folder)
    # Rest of the function...

```

- **Key Adjustments**:
    - The function begins by calling `get_search_index` with the list of filenames (`file_names`) and the index folder path, initializing the FAISS indexes for all provided documents.
    - These indexes (`vectorstores`) are then used in constructing a more complex RAG chain that can query across multiple documents.

### Multi-Document Fusion with `get_rag_fusion_chain_files`

This function applies a query fusion technique to aggregate and prioritize information from multiple documents, improving the relevance and accuracy of generated responses.

```python
def get_rag_fusion_chain_files(file_names: List[str], index_folder: str = "index", retrieval_cb=None):
    vectorstores = get_search_index(file_names, index_folder)
    query_generation_chain = get_search_query_generation_chain()
    # Continues to set up the chain...

```

- **Function Explanation**:
    - Similar to `get_rag_chain_files`, it starts by initializing search indexes for the provided file names.
    - `query_generation_chain` is prepared to generate multiple search queries from a single input query, enhancing the breadth of document retrieval.
    - The function then sets up a complex chain involving query generation, document retrieval across multiple files, and response generation based on fused query results.

### Streamlit Callback Handler Integration

Although the detailed implementation of the Streamlit callback handler (`StreamlitCallbackHandler`) is not explicitly shown in the snippet you've shared, it plays a critical role in updating the Streamlit UI based on events happening within the Langchain processing pipeline.

```python
def get_agent_chain(file_names: List[str], index_folder="index", callbacks=None, st_cb: Optional[StreamlitCallbackHandler] = None):
    # Setup involving StreamlitCallbackHandler...

```

- **Usage Highlight**:
    - In this context, `st_cb` (an instance of `StreamlitCallbackHandler`) is used to handle UI updates dynamically as the agent processes user inputs, retrieves documents, and generates responses.
    - It ensures a smooth and interactive user experience by reflecting system states, such as loading indicators or error messages, directly in the UI.

### Reciprocal Rank Fusion for Document Retrieval

The `reciprocal_rank_fusion` function is a sophisticated method for combining search results from multiple queries or documents, improving the relevance and accuracy of the information retrieved. This method is particularly effective in scenarios where multiple pieces of evidence across documents can enhance the quality of the response.

```python
def reciprocal_rank_fusion(results: List[List], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

```

- **Function Explanation**:
    - `results`: A list of lists, where each inner list contains documents retrieved based on different queries or from different indexes.
    - `fused_scores`: A dictionary to accumulate scores for each document, based on its rank in the retrieval results. Lower ranks (higher relevance) result in higher scores.
    - The function iterates through each set of documents (`docs`), ranking them and updating the `fused_scores` based on the reciprocal of their rank plus a constant `k`. This ensures that highly ranked documents across different result sets accumulate higher scores.
    - Finally, it sorts the documents based on their accumulated scores and returns this reranked list of documents.

### Generating Search Queries

The `get_search_query_generation_chain` function demonstrates how Athenas-Oracle generates multiple related search queries from a single input query. This expands the search horizon, enabling the retrieval system to gather a wider range of relevant documents.

```python
def get_search_query_generation_chain():
    prompt = ChatPromptTemplate(
        input_variables=['original_query'],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    template='You are a helpful assistant that generates multiple search queries based on a single input query.'
                )
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template='Generate multiple search queries related to: {original_query} \\n OUTPUT (4 queries):'
                )
            )
        ]
    )
    generate_queries = (
        prompt |
        ChatOpenAI(temperature=0) |
        StrOutputParser() |
        (lambda x: x.split("\\n"))
    )
    return generate_queries

```

- **Process Flow**:
    - The function sets up a `ChatPromptTemplate` that instructs the AI model to generate multiple search queries based on the `original_query`.
    - It uses a combination of system and human message prompt templates to guide the AI in this task.
    - The generated queries are then passed through the ChatOpenAI model to generate responses, parsed into a string, and split into individual queries.
    - This expanded set of queries is used to enhance document retrieval, ensuring a broad and relevant set of documents can be considered in generating the final response.

### Integration with Athenas-Oracle

Integrating these advanced retrieval and query generation mechanisms, Athenas-Oracle can provide more precise and information-rich responses to user queries. By effectively leveraging multiple documents and expanding search queries, the system ensures that the generated responses are not only relevant but also deeply informed by a wide array of sources.

This sophisticated approach to information retrieval and processing significantly enhances Athenas-Oracle's capabilities as a knowledge exploration tool, enabling it to serve as an invaluable resource for users seeking detailed and accurate information across various domains.

In summary, the addition of functions to handle multiple files and implement advanced retrieval strategies like reciprocal rank fusion and expanded query generation marks a significant enhancement in Athenas-Oracle's ability to process and generate contextually rich and accurate responses, making it a powerful tool for information retrieval and knowledge discovery.

### Putting It All Together

These components work together within Athenas-Oracle to create a powerful, AI-driven conversation agent. Messages from users are standardized and processed, chat history is condensed into clear questions, and responses are enriched with information from a vast corpus of documents. The system leverages advanced AI techniques, including retrieval-augmented generation and fusion strategies, to provide accurate, informative, and contextually relevant answers.

## Step 6: Streamlit for UI - Building the User Interface

In this step, we'll dive into the code that sets up the Streamlit user interface for Athena's Oracle. We'll break down each line of code, explaining its purpose and functionality in detail.

### Importing Necessary Libraries

```python
import base64
import streamlit as st
import os
import embed_pdf
import arxiv_downloader.utils
import requests
import re
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

```

- **Explanation**:
    - `base64`: Required for decoding base64 encoded content, particularly useful for reading GitHub README files.
    - `streamlit as st`: The Streamlit library for building interactive web applications.
    - `os`: Python's operating system module for interacting with the operating system.
    - `embed_pdf`: A custom module for embedding PDF documents.
    - `arxiv_downloader.utils`: A utility module for downloading papers from arXiv.
    - `requests`: Library for making HTTP requests.
    - `re`: Regular expression module for pattern matching.
    - `pandas as pd`: Popular library for data manipulation and analysis.
    - `st_aggrid`: Streamlit component for displaying interactive data grids.

### Function Definitions

```python
def extract_arxiv_links(readme_contents):
    """提取README内容中的所有arXiv链接"""
    arxiv_links = re.findall(r'<https://arxiv.org/abs/[^\\s)]+>', readme_contents)
    return arxiv_links

```

- **Function Explanation**:
    - This function extracts all arXiv links from the provided README contents using a regular expression pattern.
    - It returns a list of arXiv links found in the README.

```python
def get_readme_contents(repo_url):
    """通过GitHub API获取仓库README.md的内容"""
    user_repo = repo_url.replace("<https://github.com/>", "")
    api_url = f"<https://api.github.com/repos/{user_repo}/contents/README.md>"
    response = requests.get(api_url)
    if response.status_code == 200:
        content = response.json()['content']
        readme_contents = base64.b64decode(content).decode('utf-8')
        return readme_contents
    else:
        st.sidebar.error("Error: Unable to fetch README.md")
        return None

```

- **Function Explanation**:
    - This function retrieves the contents of the [README.md](http://readme.md/) file from a GitHub repository using the GitHub API.
    - It takes the repository URL as input, constructs the API URL, and makes a GET request to fetch the README contents.
    - If successful, it decodes the base64 encoded content and returns the README contents as a string. Otherwise, it displays an error message in the Streamlit sidebar.

```python
def download_arxiv_paper(link):
    """下载指定的arXiv论文"""
    arxiv_id = arxiv_downloader.utils.url_to_id(link)
    try:
        arxiv_downloader.utils.download(arxiv_id, "./pdf", False)
        st.sidebar.success(f"Downloaded: {link}")
    except Exception as e:
        st.sidebar.error(f"Failed to download {link}: {e}")

```

- **Function Explanation**:
    - This function downloads the specified arXiv paper by its URL.
    - It first extracts the arXiv ID from the URL using the `url_to_id` function from the `arxiv_downloader` module.
    - Then, it attempts to download the paper PDF to the specified directory (`./pdf` in this case).
    - If successful, it displays a success message in the Streamlit sidebar. Otherwise, it shows an error message with details of the failure.

### Sidebar Setup

```python
# create sidebar and ask for openai api key if not set in secrets
secrets_file_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_file_path):
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        else:
            print("OpenAI API Key not found in environment variables")
    except FileNotFoundError:
        print('Secrets file not found')
else:
    print('Secrets file not found')

```

- **Explanation**:
    - This section checks if a secrets file exists in the Streamlit app directory.
    - If found, it attempts to read the OpenAI API key from the secrets file and set it in the environment variables.
    - If the secrets file is not found, it prints a message indicating the absence of the file.

```python
if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    os.environ["OPENAI_API_KEY"] = st.sidebar.text_input(
        "OpenAI API Key", type="password"
    )
else:
    ...

```

- **Explanation**:
    - If the OpenAI API key is not set or doesn't start with "sk-" (indicating it's a secret key), it prompts the user to input the API key via a password input field in the

Streamlit sidebar.

- Otherwise, it proceeds with other functionalities related to GitHub repository links and arXiv paper downloads.

```python
github_link = st.sidebar.text_input("GitHub Repository URL", key="github_link")
if github_link:
    readme_contents = get_readme_contents(github_link)
    if readme_contents:
        arxiv_links = extract_arxiv_links(readme_contents)
        if arxiv_links:
            for link in arxiv_links:
                download_arxiv_paper(link)
        else:
            st.sidebar.warning("No arXiv links found in the README.")

```

- **Explanation**:
    - This part of the code adds a text input field in the Streamlit sidebar for users to input the URL of a GitHub repository.
    - Upon user input, it fetches the README contents using the `get_readme_contents` function.
    - If the README contents are retrieved successfully, it extracts arXiv links using the `extract_arxiv_links` function and proceeds to download the associated papers.
    - If no arXiv links are found in the README, it displays a warning message in the Streamlit sidebar.

### User Input and Interaction

```python
if st.sidebar.text_input("arxiv link", type="default"):
    arxiv_link = st.sidebar.text_input("arxiv link", type="default")
    arxiv_id = arxiv_downloader.utils.url_to_id(arxiv_link)
    try:
        arxiv_downloader.utils.download(arxiv_id, "./pdf", False)
        st.sidebar.info("Done!")
    except Exception as e:
        st.sidebar.error(e)
        st.sidebar.error("Failed to download arxiv link.")

```

- **Explanation**:
    - This code block adds a text input field in the Streamlit sidebar for users to directly input an arXiv link.
    - If a link is provided, it attempts to download the associated paper using the `arxiv_downloader` module.
    - It displays success or error messages in the Streamlit sidebar accordingly.

```python
if st.sidebar.button("Embed Documents"):
    st.sidebar.info("Embedding documents...")
    try:
        embed_pdf.embed_all_pdf_docs()
        st.sidebar.info("Done!")
    except Exception as e:
        st.sidebar.error(e)
        st.sidebar.error("Failed to embed documents.")

```

- **Explanation**:
    - This part creates a button in the Streamlit sidebar labeled "Embed Documents".
    - Upon clicking the button, it triggers the embedding process for all PDF documents using the `embed_pdf` module.
    - It displays success or error messages in the Streamlit sidebar based on the outcome of the embedding process.

### Setting Up the Main App

```python
st.title("🔎 Welcome to Athena's Oracle")

```

- **Explanation**:
    - This line sets the title of the Streamlit app to "🔎 Welcome to Athena's Oracle", providing a welcoming message to the users.

```python
items_per_page = st.number_input("Set the number of items per page:", min_value=1, max_value=100, value=10)

```

- **Explanation**:
    - This code snippet adds a number input widget for users to set the number of items displayed per page.
    - Users can choose a value between 1 and 100.

```python
search_query = st.text_input("Search files by name:")

```

- **Explanation**:
    - This line adds a text input field where users can enter search keywords to filter files by name.

### Data Processing and Display

```python
file_list = embed_pdf.get_all_index_files()
df_files = pd.DataFrame(file_list, columns=["File Name"])

```

- **Explanation**:
    - This part retrieves the list of all index files using the `get_all_index_files` function from the `embed_pdf` module.
    - It then converts the list into a pandas DataFrame with a single column named "File Name".

```python
if search_query:
    df_files = df_files[df_files["File Name"].str.contains(search_query, case=False)]

```

- **Explanation**:
    - If a search query is provided by the user, this code filters the DataFrame `df_files` to include only rows where the "File Name" column contains the search query (case-insensitive).

```python
gb = GridOptionsBuilder.from_dataframe(df_files)

```

- **Explanation**:
    - This line initializes a `GridOptionsBuilder` object from the DataFrame `df_files`, which is used to configure settings for the interactive data grid.

```python
gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
gb.configure_selection('multiple', use_checkbox=True, rowMultiSelectWithClick=True, suppressRowDeselection=False)
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=items_per_page)
gb.configure_side_bar()

```

- **Explanation**:
    - These lines configure various settings for the data grid, including column grouping, selection mode, pagination, and sidebar options.

```python
grid_options = gb.build()

```

- **Explanation**:
    - This line builds the grid options based on the configurations set using the `GridOptionsBuilder`.

```python
selected_files = AgGrid(
    df_files,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    allow_unsafe_jscode=True,
    fit_columns_on_grid_load=True,
)

```

- **Explanation**:
    - This code block creates an instance of the `AgGrid` component, displaying the DataFrame `df_files` with the specified grid options.
    - It enables the grid to update when the model changes and allows unsafe JavaScript code execution.

```python
selected_rows = selected_files["selected_rows"]
chosen_files = [row["File Name"] for row in selected_rows]

```

- **Explanation**:
    - These lines retrieve the selected rows from the data grid and extract the corresponding file names.
    - The selected file names are stored in the list `chosen_files`.

### Final Output and Interaction with AI Model

```python
if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
    st.stop()

```

- **Explanation**:
    - This code block checks if the OpenAI API key is set and starts with "sk-" (indicating it's a secret key).
    - If the API key is not set or doesn't match the expected format, it displays a warning message and stops the execution of further code.

```python
chosen_rag_method = st.radio(
    "Choose a RAG method", rag_method_map.keys(), index=0
)

```

- **Explanation**:
    - This line adds a radio button in the Streamlit app for users to choose a RAG (Retrieval-Augmented Generation) method.
    - It displays options based on the keys of the `rag_method_map` dictionary.

```python
get_rag_chain_func = rag_method_map[chosen_rag_method]

```

- **Explanation**:
    - This line selects the RAG function based on the user's choice of RAG method from the radio button.

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.mark

down(message["content"])

```

- **Explanation**:
    - This code block initializes the message history state and renders older messages in the chat interface.

```python
prompt = st.chat_input("Enter your message...")

```

- **Explanation**:
    - This line creates an input field in the Streamlit app where users can enter their messages.

```python
if prompt:
	with st.chat_message("user"):
		st.markdown(prompt)
```

- **Explanation**:
    - This code block renders the user's message in the chat interface.

```python
    custom_chain = get_rag_chain_func(chosen_files, retrieval_cb=retrieval_cb)

```

- **Explanation**:
    - This line generates a custom RAG (Retrieval-Augmented Generation) chain based on the chosen RAG method and the selected files.
    - It utilizes the `get_rag_chain_func` function, which represents either `get_rag_chain_files` or `get_rag_fusion_chain_files` depending on the user's selection.
    - Additionally, it provides a retrieval callback function (`retrieval_cb`) to handle context retrieval during conversation.

```python
    full_response = ""
    for response in custom_chain.stream(
        {"input": prompt, "chat_history": chat_history}
    ):
        if "output" in response:
            full_response += response["output"]
        else:
            full_response += response.content

        message_placeholder.markdown(full_response + "▌")
        update_retrieval_status()

```

- **Explanation**:
    - This loop iterates over the responses generated by the custom RAG chain using the `stream` method.
    - For each response, it checks if there is an "output" key. If present, it appends the output to the `full_response` string; otherwise, it appends the content directly.
    - The `message_placeholder` component is used to display the ongoing conversation, with each response being added to the markdown content.
    - The `update_retrieval_status` function updates the status of context retrieval in the Streamlit UI.

```python
    st.session_state.messages.append({"role": "assistant", "content": full_response})

```

- **Explanation**:
    - After the conversation loop completes, this line adds the full response generated by the assistant to the message history.
    - The role of the message is set as "assistant", indicating that it's a response from the AI model.
