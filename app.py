import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy import create_engine
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

DATABASE_URL = 'postgresql+psycopg2://postgres.oifpqngnyxthcptbfgqr:F8qpcxfBeXHiypGM@aws-0-ap-south-1.pooler.supabase.com:6543/postgres'
engine = create_engine(DATABASE_URL)
# Ensure to set your environment variables or replace these values accordingly
SUPABASE_URL = "https://oifpqngnyxthcptbfgqr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9pZnBxbmdueXh0aGNwdGJmZ3FyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTc2NjE5MTAsImV4cCI6MjAzMzIzNzkxMH0.-JYoel_gkWOswP6UKn3AcJ1Lbu4mld8UbDrYadXbc0o"

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    max_chunk_size = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=10,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_embeddings(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    return embeddings

def find_best_chunk(user_query, content, content_embeddings):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([user_query], convert_to_tensor=True)
    similarities = np.dot(content_embeddings, query_embedding.T).squeeze()

    # Find the index of the best matching chunk
    best_chunk_index = np.argmax(similarities)
    best_chunk = content[best_chunk_index]  # Return the actual text chunk
    return best_chunk


def get_conversation_chain(vectorstore):
    # This function is kept for completeness but is not used in this scenario
    pass




api_token="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"




def handle_userinput(user_question):
    # Get the stored embeddings and content
    id_value = st.session_state['id']
    response = supabase_client.table('pdfs').select('embeddings', 'content').eq('id', id_value).execute()

    st.write("Retrieved Data:", response.data)  # Debug output

    data = response.data[0]
    content = data['content']
    content_embeddings = np.array(data['embeddings'])

    st.write("Raw content from database:", content)
    st.write("Type of content:", type(content))

    if isinstance(content, str):
        try:
            content = eval(content)
            st.write("Content was evaluated using eval")
        except:
            st.error("Content is not in the expected format")
            return

    st.write("Content:", content)  # Debug output
    st.write("Content Embeddings Shape:", content_embeddings.shape)  # Debug output

    # Find the best matching chunk
    best_chunk = find_best_chunk(user_question, content, content_embeddings)

    st.write("Best Chunk:", best_chunk)  # Debug output

    input_text = f"You are an AI language model and will answer the query based on the best chunk provided. Query: {user_question} Best chunk: {best_chunk}"
    st.write(f"Input text: {input_text}")

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    # Define the payload for the request
    payload = {"inputs": input_text, "parameters": {"max_length": 512, "temperature": 0.7, "repetition_penalty": 1.2}}

    st.write("Payload generated successfully")

    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers=headers,
        json=payload
    )

    # Debugging: Print the response status code and content
    st.write("LLM Response Status Code:", response.status_code)
    st.write("LLM Response Content:", response.content.decode('utf-8'))

    # Extract the response text
    response_data = response.json()
    response_text = response_data[0]['generated_text'] if isinstance(response_data, list) else response_data['generated_text']

    # Post-process the response to remove repetition
    response_text = ' '.join(dict.fromkeys(response_text.split()))

    st.write("LLM Response:", response_text)  # Debug output

    # Update chat history in session state
    st.session_state.chat_history = [
        {"role": "user", "content": user_question},
        {"role": "bot", "content": response_text}
    ]

    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "id" not in st.session_state:
        st.session_state.id = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    st.header("Chat with multiple PDFs :books:")

    id_value = st.number_input("Enter your ID", value=1)
    st.session_state.id = id_value

    user_question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                st.write("Raw Text:", raw_text)  # Debug output

                text_chunks = get_text_chunks(raw_text)
                st.write("Text Chunks:", text_chunks)  # Debug output

                embeddings = get_embeddings(text_chunks)
                st.write("Embeddings Shape:", embeddings.shape)  # Debug output

                # Convert embeddings to list for storage
                embedding_list = embeddings.tolist()
                data = {'id': id_value, 'content': text_chunks, 'embeddings': embedding_list}
                response = supabase_client.table('pdfs').insert(data).execute()

                # st.write("Insert Response:", response)  # Debug output

                
    # if st.session_state.pdf_processed and user_question:
        handle_userinput(user_question)

if __name__ == "__main__":
    main()
