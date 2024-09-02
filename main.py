import cohere
import streamlit as st
import cohere
import pinecone
from helper import *
from langchain.memory import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Cohere
from langchain import PromptTemplate, LLMChain
from yolo import *
from gtts import gTTS
import os
language = 'en'

st.title('Sight 👁️')


COHERE_API_KEY = "xnm4LRAs5n5SV3IFyL11oLdvTAlkGriL5Gjz1tBJ"
co = cohere.Client('xnm4LRAs5n5SV3IFyL11oLdvTAlkGriL5Gjz1tBJ')
pinecone.init(api_key="88a8b75c-3069-4111-8d30-e13f5ef3c8fd", environment="gcp-starter")

@st.cache_resource
def Vision_GPT():
    index_name = 'vision'
    # No need to create it Pinecone index
    # if index_name not in pinecone.list_indexes():
    #     pinecone.create_index(
    #         index_name,
    #         dimension=4096,
    #         metric='cosine'
    #     )

    # Connect to index
    index = pinecone.Index(index_name)
    return index

index = Vision_GPT()

vision_info = yolo()
formatted_string = ', '.join(f"{key}: {item[key]}" for item in vision_info for key in item)
template = f"""I am visually impaired, I am trying to walk with the help of my computer vision model. The computer vision model gives me all persons and objects with their position in my view. 
Please guide me through to my destination. All objects are infront of me so I will be walking forward.  First you will list all the objects and their position, make sure you advise me on being careful if two objects are in the same are (left, right, center). Then I will ask a question, please help me in two to three sentences so I can find my way. Also if I ask, let me know if you see the object before. If there is not question, then respond accordingly.

+ {formatted_string}
""" + """
{prompt}
Answer:
"""
llm = Cohere(model="command-nightly", cohere_api_key=COHERE_API_KEY)

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)


prompt = PromptTemplate(template=template, input_variables=['prompt'])
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)


view_messages = st.expander("View the message contents in session state")


if 'key' not in st.session_state:
        st.session_state.key = ""

for msg in msgs.messages:
    st.session_state.key = st.session_state.key + " " + str(msg.type) +": " + str(msg.content) + ","
    if msg.type != 'system':
        st.chat_message(msg.type).write(msg.content)

# Store in Pinecone
add_embeddings(st.session_state.key, index)


# if prompt := st.chat_input():
if prompt:= st.chat_input():
    st.chat_message("human").write(prompt)

    response = llm_chain.run(prompt)
    
    myobj = gTTS(text=response, lang=language, slow=False)
    myobj.save("audio.mp3")
    os.system("afplay audio.mp3")
    st.chat_message("ai").write(response)


with view_messages:
    """
    Memory initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
