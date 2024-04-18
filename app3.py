from dotenv import load_dotenv
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from functions import *
import datetime

def main():
    load_dotenv()
    st.set_page_config(page_title="Pitaj svoj PDF",page_icon="http://ie.mas.bg.ac.rs/data_store/images/mf_logo.png")
    

    #initialzation of items in sasion state
    if "conversation" not in st.session_state:
          st.session_state.conversation = None
    if "chat_history" not in st.session_state:
          st.session_state.chat_history = None
    

    #Headder columns 
    col1, col2 = st.columns([0.3,0.7],gap="small")
    with col1:
            st.image("http://ie.mas.bg.ac.rs/data_store/images/ie_logo_top_green.png",width=50)

    with col2:
            st.header("MasinacAI")
    


    # sidebar customization and functions 
    with st.sidebar:
        st.subheader('Tvoj dokument')
        pdfs =  st.file_uploader("Unesi svoj PDF i klikni na 'Procesuiraj'",accept_multiple_files=True , type='pdf')
        if len(pdfs) > 0:
            if st.button('Procesuiraj'):
                with st.spinner("Procesuiranje"): 
                    # get pdf text
                    raw_text = get_pdf_text(pdfs)
                        
                    #split into chunks 
                    chunks = get_text_chunks(raw_text)
                
                    #create embeddings 
                    vector_store = get_vectorstore_from_pdf(chunks)
            else:
                vector_store = get_from_vector_store_qdrant()
        
    
    #chatbot messaging                      
    prompt = st.chat_input("Postavi pitanje")
    if prompt:
        st.session_state.conversation = get_conversation_chain(vector_store)                   
        handle_user_input(prompt)
    




if __name__ == '__main__':
    main()



