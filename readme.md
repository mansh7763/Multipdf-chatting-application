# MultiPDF Chat App
<img width="956" alt="Screenshot 2024-06-15 130614" src="https://github.com/mansh7763/Multipdf-chatting-application/assets/130061782/fc03572f-c6d3-46ba-8dd4-48804b160283">

## Introduction
------------
The MultiPDF Chat App is a streamlit Python application that allows you to chat with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.

This application also helps you to use your id to get the information related to the documents you have previously uploaded.
It also helps you to add documents with your previous uploaded documents.

## How It Works

The application follows these steps to provide responses to your questions:
1. Take your user id. It helps to understand whether you are a old user or new user. If you are a new user, just insert a id and then upload the pdf.
   If you have previously created a id then you can use it get information with your previous document and it also gives you facility to upload and add new documents with 
   your previous documents.

2. PDF Loading: The app reads multiple PDF documents and extracts their text content.

3. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

4. Text Store: Chunks of the text is stored in supabase database with your given user id.

5. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

6. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

7. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

## Usage
--------------------------------------------------------------
To use this MultiPDF Chat App, follow these steps:

1. Clone this repo in a new folder and open the folder using vs code

2. To install dependencies, creating a virtual environment is highly appreciated. Create a virtual environement using terminal:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```
   venv/Scripts/Activate
   ```

4. Install the the dependencies with:
   ```
   pip install -r requirements.txt
   ```

5. Create a .env file and insert appropriate keys, urls and token

6. After this you have to comment my streamlit keys access method and uncomment the "os.getenv()"

7. Now run with:
   ```
   streamlit run app.py
   ```
8. Insert your id and and upload pdfs and click process my data.
   
9. Ask questions in natural language about the loaded PDFs using the chat interface.

## Contributing
------------
This repo is waiting for your fruitful contributions.
