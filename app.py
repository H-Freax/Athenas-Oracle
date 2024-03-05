import base64

import streamlit as st
import os
import embed_pdf
import arxiv_downloader.utils

import requests
import re

def extract_arxiv_links(readme_contents):
    """提取README内容中的所有arXiv链接"""
    arxiv_links = re.findall(r'https://arxiv.org/abs/[^\s)]+', readme_contents)
    return arxiv_links

def get_readme_contents(repo_url):
    """通过GitHub API获取仓库README.md的内容"""
    user_repo = repo_url.replace("https://github.com/", "")
    api_url = f"https://api.github.com/repos/{user_repo}/contents/README.md"
    response = requests.get(api_url)
    if response.status_code == 200:
        content = response.json()['content']
        readme_contents = base64.b64decode(content).decode('utf-8')
        return readme_contents
    else:
        st.sidebar.error("Error: Unable to fetch README.md")
        return None

def download_arxiv_paper(link):
    """下载指定的arXiv论文"""
    arxiv_id = arxiv_downloader.utils.url_to_id(link)
    try:
        arxiv_downloader.utils.download(arxiv_id, "./pdf", False)
        st.sidebar.success(f"Downloaded: {link}")
    except Exception as e:
        st.sidebar.error(f"Failed to download {link}: {e}")


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

if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    os.environ["OPENAI_API_KEY"] = st.sidebar.text_input(
        "OpenAI API Key", type="password"
    )
else:
    # 输入GitHub链接
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




    if st.sidebar.text_input("arxiv link", type="default"):
        arxiv_link = st.sidebar.text_input("arxiv link", type="default")
        arxiv_id = arxiv_downloader.utils.url_to_id(arxiv_link)
        try:
            arxiv_downloader.utils.download(arxiv_id, "./pdf", False)
            st.sidebar.info("Done!")
        except Exception as e:
            st.sidebar.error(e)
            st.sidebar.error("Failed to download arxiv link.")

    if st.sidebar.button("Embed Documents"):
        st.sidebar.info("Embedding documents...")
        try:
            embed_pdf.embed_all_pdf_docs()
            st.sidebar.info("Done!")
        except Exception as e:
            st.sidebar.error(e)
            st.sidebar.error("Failed to embed documents.")



# create the app
st.title("🔎 Welcome to Athena's Oracle")

chosen_files = st.multiselect(
    "Choose files to search", embed_pdf.get_all_index_files(), default=None
)
#
# print(chosen_files)
# if chosen_files:  # Check if any files are selected
#     for chosen_file in chosen_files:

# check if openai api key is set
if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
    st.stop()

# load the agent
from llm_helper import convert_message, get_rag_chain, get_rag_fusion_chain,get_rag_chain_files,get_rag_fusion_chain_files

rag_method_map = {
    # 'Basic RAG': get_rag_chain,
    "Basic RAG": get_rag_chain_files,
    'RAG Fusion': get_rag_fusion_chain_files
}
chosen_rag_method = st.radio(
    "Choose a RAG method", rag_method_map.keys(), index=0
)
get_rag_chain_func = rag_method_map[chosen_rag_method]
## get the chain WITHOUT the retrieval callback (not used)
# custom_chain = get_rag_chain_func(chosen_file)

# create the message history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# render older messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# render the chat input
prompt = st.chat_input("Enter your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # render the user's new message
    with st.chat_message("user"):
        st.markdown(prompt)

    # render the assistant's response
    with st.chat_message("assistant"):
        retrival_container = st.container()
        message_placeholder = st.empty()

        retrieval_status = retrival_container.status("**Context Retrieval**")
        queried_questions = []
        rendered_questions = set()
        def update_retrieval_status():
            for q in queried_questions:
                if q in rendered_questions:
                    continue
                rendered_questions.add(q)
                retrieval_status.markdown(f"\n\n`- {q}`")
        def retrieval_cb(qs):
            for q in qs:
                if q not in queried_questions:
                    queried_questions.append(q)
            return qs
        
        # get the chain with the retrieval callback
        custom_chain = get_rag_chain_func(chosen_files, retrieval_cb=retrieval_cb)
        
        if "messages" in st.session_state:
            chat_history = [convert_message(m) for m in st.session_state.messages[:-1]]
        else:
            chat_history = []

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

        retrieval_status.update(state="complete")
        message_placeholder.markdown(full_response)

    # add the full response to the message history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
