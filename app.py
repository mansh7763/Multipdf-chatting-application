import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy import create_engine
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import re

DATABASE_URL = 'postgresql+psycopg2://postgres.oifpqngnyxthcptbfgqr:F8qpcxfBeXHiypGM@aws-0-ap-south-1.pooler.supabase.com:6543/postgres'
engine = create_engine(DATABASE_URL)
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
    max_chunk_size = 250
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

def find_top_chunks(user_query, content, content_embeddings, top_n=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([user_query], convert_to_tensor=True)
    similarities = np.dot(content_embeddings, query_embedding.T).squeeze()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_chunks = [content[idx] for idx in top_indices]
    return top_chunks

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,@-]', '', text)
    return text.strip()

def extract_chunks(texts):
    cleaned_chunks = [clean_text(text) for text in texts]
    return cleaned_chunks

api_token = "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"

def handle_userinput(user_question):
    id_value = st.session_state['id']
    response = supabase_client.table('pdfs').select('embeddings', 'content').eq('id', id_value).execute()
    data = response.data[0]
    content = data['content']
    content_embeddings = np.array(data['embeddings'])

    if isinstance(content, str):
        try:
            content = eval(content)
        except:
            st.error("Content is not in the expected format")
            return

    best_chunk = find_top_chunks(user_question, content, content_embeddings)
    best_chunk = extract_chunks(best_chunk)

    input_text = f"You are an AI language model and will answer the query based on the best chunk provided. Query: {user_question} Best chunk: {best_chunk}"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": input_text, "parameters": {"max_length": 512, "temperature": 0.7, "repetition_penalty": 1.2}}

    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        response_data = response.json()
        if isinstance(response_data, list) and 'generated_text' in response_data[0]:
            response_text = response_data[0]['generated_text']
        else:
            response_text = "Sorry, I couldn't generate a response. Please try again."
    else:
        response_text = f"Error: {response.status_code}. {response.content.decode('utf-8')}"

    response_text = ' '.join(dict.fromkeys(response_text.split()))

    st.session_state.chat_history = [
        {"role": "user", "content": user_question},
        {"role": "bot", "content": response_text}
    ]

    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

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
            if id_value and pdf_docs and user_question:
                with st.spinner("Processing"):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        embeddings = get_embeddings(text_chunks)
                        embedding_list = embeddings.tolist()
                        data = {'id': id_value, 'content': text_chunks, 'embeddings': embedding_list}
                        supabase_client.table('pdfs').insert(data).execute()
                        st.session_state.pdf_processed = True
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")
            else:
                st.error("Please enter your ID, upload PDFs, and ask a question before processing.")

    if st.session_state.pdf_processed and user_question:
        handle_userinput(user_question)

if __name__ == "__main__":
    main()
