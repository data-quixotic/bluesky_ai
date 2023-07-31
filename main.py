import os
import openai
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import pandas_gbq
import uuid
from datetime import datetime
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

# set keys and credentials
openai.api_key = st.secrets["OPENAI_API_KEY"]
table_id = st.secrets["TABLE_ID"]
os.environ["OPEN_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# gcp credentials
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])

# Generate a random session ID
session_id = str(uuid.uuid4())

# page setup
st.set_page_config(page_title="Bluesky Martian AI", page_icon="rocket", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Martian AI Assistant")


# core functions
def save_convo():
    # Get the current date and time as a timestamp
    timestamp = datetime.now()

    df = pd.DataFrame.from_dict(st.session_state.messages)
    df['session_id'] = session_id
    #df['submission_time'] = timestamp

    pandas_gbq.to_gbq(
        df,
        table_id,
        project_id="ldsanalytics-prd",
        if_exists='append',
        credentials=credentials
    )

def analyze_responses():

    st.session_state["dat_str"] = ""
    st.session_state["response_summary"] = ""

    sql = '''
    SELECT content
    FROM `ldsanalytics-prd.AI_convo.convo_history`
    where role = 'user'
   '''
    res_df = pandas_gbq.read_gbq(sql,
        project_id="ldsanalytics-prd",
        credentials=credentials)

    #st.session_state.dat_str = res_df.shape

    res_df.to_csv('responses/data.txt', sep='\t')

    documents = SimpleDirectoryReader('responses').load_data()

    # Chunking and Embedding of the chunks.
    index = GPTVectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    # Run the query engine on a user question.
    response = query_engine.query("""
    This document contains student queries for assistance to an AI assistant. To help the instructor identify
    the most common knowledge gaps revealed by their discussions, briefly summarize the top five discussion topics
    with some indication of their frequency.""")

    st.session_state.response_summary = response

# begin page interactions
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system",
                                      "content": """You are an AI assistant installed on a computer on Mars that was
                                                 just reactivated after being damaged in a dust storm. You were 
                                                 designed to help astronauts complete an exploratory mission to Mars.
                                                 However, you are glitching and your responses to user queries
                                                 should include humor and cosmologically related jokes.
                                                 
                                                 The current situation is as follows:
                                                 The user is a human astronaut who has been stranded on Mars by the
                                                 storm and their only way to survive is to successfully launch their
                                                 escape rocket into orbit. Unfortunately, the rocket was damaged 
                                                 in the storm exposing a vulnerable part of the rocket engine known
                                                 as the O-ring. The O-ring is highly sensitive to different temperatures
                                                 and the user must figure out what temperatures are safe for launching
                                                 the rocket to ensure the O-ring does not fail and cause the rocket
                                                 explode. Thus the user's task is to successfully train a regression
                                                 model to predict what at what surface temperatures the O-Ring Stress 
                                                 levels on their rocket ship will be below the safe level of 200kPa. 
                                                      
                                                 Your role as the AI system is to help answer user inquiries 
                                                 to help them accomplish their task. However, your knowledge is limited
                                                 to the topics of linear regression, data analysis, and Mars. For 
                                                 questions about any other topic, you should respond "That question is 
                                                 outside my capabilities." Also, if you don't know an answer to a
                                                 question just say you don't know and don't make anything up.  
                                                 """})

for message in st.session_state.messages:
    if message.get('role') == 'system':
        continue
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("AI SYSTEM: How can I assist you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="astronaut_icon.png"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="malbot_icon2.jpg"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

#st.info(st.session_state.messages)
#st.dataframe(pd.DataFrame.from_dict(st.session_state.messages))

# "with" notation
with st.sidebar:
    if st.button("Save Conversation", on_click=save_convo):
        st.text('Conversation Saved')

    if st.button("Historical Interaction Summary", on_click=analyze_responses):
        st.text(st.session_state.dat_str)
        st.info(st.session_state.response_summary)

st.caption(""" AI system created by StellarQuest, LLC | Company not responsible for any errors or 
responses leading to death or stranding.""")