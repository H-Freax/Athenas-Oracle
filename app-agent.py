import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm_helper import get_agent_chain, get_lc_oai_tools

with st.sidebar:
    openai_api_key = st.secrets["OPENAI_API_KEY"]

st.title("🔎 Athena's Oracle - Chat with search")


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

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key=openai_api_key, streaming=True)

    filename=["2211.00191v1.Edge_Grasp_Network_A_Graph_Based_SE_3_invariant_Approach_to_Grasp_Detection.pdf","mamba1.pdf","mamba2.pdf"]

    lc_tools, _ = get_lc_oai_tools(file_names=filename)
    search_agent = initialize_agent(lc_tools, llm, agent=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True, verbose=True)

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant, use the search tool to answer the user's question and cite only the page number when you use information coming (like [p1]) from the source document. Always use the content from the source document to answer the user's question. If you need to compare multiple subjects, search them one by one."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    search_agent.agent.prompt = agent_prompt
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(prompt, callbacks=[st_cb])
        # search_agent = get_agent_chain(callbacks=[st_cb])
        # response = search_agent.invoke({"input": prompt})
        # response = response["output"]
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
