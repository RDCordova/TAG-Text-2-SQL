import streamlit as st 
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
import boto3
from langchain_community.chat_models import BedrockChat
from boto3 import client
from botocore.config import Config
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import sentence_transformers
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.conversation.memory import ConversationBufferMemory

st.set_page_config(page_title="Chat with Database", page_icon=":speech_balloon:")

st.title("Text to SQL Engine")

on = st.toggle("Display SQL") 

# Database connection
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = 'sqlite:///./{}'.format(database)
    return SQLDatabase.from_uri(db_uri)


sqlite_uri = 'sqlite:///./snyth.db' 
db = SQLDatabase.from_uri(sqlite_uri)

with st.sidebar:
    st.subheader("Settings")
    st.write("Connect to the database and start chatting.")
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="admin", key="Password")
    st.text_input("Database", value="Synth_CDH", key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
            

        
def load_model():
    config = Config(read_timeout=1000)

    bedrock_runtime = boto3.client(service_name='bedrock-runtime', 
                          region_name='us-east-1',
                          config=config)

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    
    model_kwargs = { 
        "max_tokens": 100000,
        "temperature": 0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
        
    }
    
 
    
    model = BedrockChat(
        client=bedrock_runtime,
        model_id=model_id,
        model_kwargs=model_kwargs,
    )

    return model

sql_llm = load_model()

#functions 
def get_vector_store(filename):
    filename = filename
    pdf_reader = PdfReader(filename)
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    pdf_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 200)

    for idx, page in enumerate(pdf_reader.pages):
        if len(text) > 0:
            pdf_docs.extend(
                text_splitter.create_documents(
                    texts = [text],
                    metadatas = [{'filename': filename, 'page': idx+1}]
                )
            )
            
    embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(pdf_docs, embedding)

#add memory 
memory = ConversationBufferMemory(return_messages=True)

def get_schema(_):
    schema = db.get_table_info()
    return schema


def run_query(query):
    return db.run(query)

def sql_answer(user_question):
    # add Try to run SQL 5 times 
    error_counter = 0 
    while error_counter < 5:
        try:
            question = user_question
            ss_result = v_db.similarity_search(question)
            top_ss_docs = ss_result[0:1]
            context = " ----- ".join([ss_result.page_content for ss_result in top_ss_docs])
            sql_result = sql_chain.invoke({"context":context,"history":memory.load_memory_variables({}),
                                        "question":user_question})
            result = full_chain.invoke(({"context": context,"history":memory.load_memory_variables({}),
                                         "question": user_question})).content
            return {"query" : user_question, 'result': result, 'sql': sql_result}
        except:
            error_counter += 1 
            result = "Unable to generate answer based on the question being asked. please try again with a different question."
    memory.save_context({"input":   user_question}, {"output":  result})
    return {"query" : user_question, 'result': result}


# RAG
v_db = get_vector_store('data_dict.pdf')

#sql chain 
template = """Based on the table schema and context below, write a SQL query that would answer the user's question. Only return the sql code not any explanation.:
{schema}
{context}
{history} 
Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | sql_llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

#full chain 
template = """Based on the table schema below, question, sql query, and sql response, and context write a natural language response. The answer should be concise one or two sentance and should include any nubers from the original query. If the answer involvs a list of output, reutrn the full list. You should not refernce the database or its tables:
{schema}
{context}
{history}
Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(template)


full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda vars: run_query(vars["query"]),
    )
    | prompt_response
    | sql_llm
)



#chatbot

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm your database knowledge assistant. Ask me anything about your database."),
    ]
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
            
# user_query = st.chat_input("Type a message...")
# if user_query is not None and user_query.strip() != "":
#     st.session_state.chat_history.append(HumanMessage(content=user_query))
    
#     with st.chat_message("Human"):
#         st.markdown(user_query)
#     with st.chat_message("AI"):
#         response = sql_answer(user_query)['result']
#         st.markdown(response)
#     st.session_state.chat_history.append(AIMessage(content=response))

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    if on:
        with st.chat_message("AI"):
            sql_response = sql_answer(user_query)['sql']
            st.markdown(sql_response)
        st.session_state.chat_history.append(AIMessage(content=sql_response))
    with st.chat_message("AI"):
        response = sql_answer(user_query)['result']
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))