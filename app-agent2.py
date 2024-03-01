import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from llm_helper import get_agent_chain, get_lc_oai_tools, convert_message
from langchain.agents import AgentExecutor

with st.sidebar:
    openai_api_key = st.secrets["OPENAI_API_KEY"]

st.title("🔎 Freax's GPT2 - Chat with search")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if "messages" in st.session_state:
        chat_history = [convert_message(m) for m in st.session_state.messages[:-1]]
    else:
        chat_history = []

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        agent = get_agent_chain(file_names=[],st_cb=st_cb)

        response = agent.invoke({
            "input": prompt,
            "chat_history": chat_history,
        })
        response = response["output"]
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
