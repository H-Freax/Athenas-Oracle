# 🌌 Athena's Oracle: The AI-powered Document Insight Engine 📚

Welcome to **Athena's Oracle**, the groundbreaking AI companion designed to sail through the vast ocean of documents with the wisdom and insight of the ancient goddess of wisdom herself, Athena. 🌟 Drawing upon the mighty advancements in artificial intelligence and natural language processing, Athena's Oracle empowers users to dive deep into academic papers, reports, and any form of textual content, uncovering hidden insights with ease and precision like never before.

## Features 🚀✨

- **Intuitive Query Handling**: Whisper your questions into the ear of Athena's Oracle, and be guided to the wisdom you seek, leveraging the depths of your entire document repository. 📖
- **Multi-Document Intelligence**: Capable of parsing and understanding layers of information across multiple documents, Athena's Oracle weaves answers that are not just accurate but breathtakingly comprehensive. 🌐
- **Reciprocal Rank Fusion**: Through a sophisticated dance of algorithms, the most relevant documents emerge, combining forces from various sources to deliver the ultimate insights. 🔍
- **Streamlit Integration**: Step into the temple of Athena's Oracle via a user-friendly interface, making the art of complex document analysis accessible to all mortals. 💻

## How It Works 🔮

Athena's Oracle conjures its magic by intertwining several state-of-the-art technologies and methodologies:

1. **Document Retrieval**: Harnesses the power of advanced vector search technologies to unearth the most relevant documents based on your heart's queries. 🗂️
2. **Question Answering**: Employs the wisdom of the latest language models to generate precise answers to your inquiries, drawing knowledge from the identified documents. 💡
3. **Reciprocal Rank Fusion**: Gathers the search results from multiple documents, ensuring the knowledge bestowed upon you is the most pertinent and rich. 📊

## Getting Started on Your Quest 🌟

To embark on your journey with Athena's Oracle, follow these mystic steps:

1. **Clone the Repository**: Summon the latest version of Athena's Oracle from our GitHub sanctuary.
   ```
   git clone https://github.com/H-Freax/Athenas-Oracle.git
   ```
2. **Install Dependencies**: Enter the sacred project directory and conjure the required Python elixirs.
   ```
   cd Athenas-Oracle
   pip install -r requirements.txt
   ```


## Running the App 🚴‍♂️

After you've got all the requirements set up, there's one more crucial step before you can unleash the full power of Athena's Oracle: adding your OpenAI API key. This key is like the secret password that grants you access to the vast intelligence Athena's Oracle taps into.

### Securing the OpenAI API Key 🔑

You have two options for managing the OpenAI API key:

1. **Add to Streamlit Secrets**: You can securely store your API key by adding it to the `.streamlit/secrets.toml` file. This way, it's automatically picked up by the app, and you don't have to worry about it again. Here's how you format it in the file:

   ```toml
   OPENAI_API_KEY = "sk-yourapikeyhere"
   ```

   Make sure to replace `"sk-yourapikeyhere"` with your actual OpenAI API key. 

2. **User Input in Sidebar**: If you prefer, you can also set up the app to ask users to input their OpenAI API key in the sidebar each time they visit the page. This method is a bit more flexible and great for when you're sharing your creation with others but don't want to share your API key directly.

### Awakening Athena's Oracle 🌞

With your API key securely in place, you're ready to bring Athena's Oracle to life. Open your terminal or command prompt, navigate to the folder where you've stored Athena's Oracle, and run the following spell:

```bash
streamlit run app.py
```


## Dependencies 📜🔗

Embark on this adventure equipped with these magical artifacts:

- **LangChain**: For the seamless weaving of language models into the fabric of applications.
- **FAISS**: For the swift and efficient search and clustering of dense vectors, akin to finding needles in cosmic haystacks.
- **OpenAI**: For invoking the powerful oracles housed within AI models.
- **arxiv-downloader**: A revered tool for summoning documents from the halls of arXiv, showcasing the boundless realms of document retrieval.
- **LLMStreamlitDemoBasic**: A beacon of inspiration on how to craft interactive demos with Streamlit and the oracles of language models, enhancing the mortal's experience within Athena's Oracle.

## Contributions and Feedback 📬🤝

Athena's Oracle flourishes with the contributions and insights from the community. Whether it's refining the algorithm, enhancing the divine interface, or expanding the document corpus, your wisdom is invaluable to us. Feel free to fork the repository, infuse it with your enhancements, and submit a pull request.

## Acknowledgements 🌺

- **arXiv-downloader**: Heartfelt thanks to [arXiv-downloader](https://github.com/braun-steven/arxiv-downloader) for providing easy access to the scrolls of academic papers, aiding in the continuous growth of Athena's Oracle's knowledge base.
- **LLMStreamlitDemoBasic**: Our gratitude extends to the sages at [LLMStreamlitDemoBasic](https://github.com/nimamahmoudi/LLMStreamlitDemoBasic) for laying down the mystical foundations for integrating great language models with Streamlit, thus enhancing the interactive pilgrimage through Athena's Oracle.

## License 📖

Athena's Oracle is bestowed upon the world under the MIT License. For more details, consult the LICENSE scroll.
