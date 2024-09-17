import asyncio
import os
import streamlit as st
import fitz  # PyMuPDF
from nltk import word_tokenize
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import (
    UnstructuredCSVLoader, UnstructuredExcelLoader,
    Docx2txtLoader, UnstructuredPowerPointLoader
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter



# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA"

# Title of the app
st.title("Document Analysis with Retrieval-Augmented Generation (RAG)")

# Upload the file
uploaded_file = st.file_uploader("Upload Document:")
question = st.text_input("Insert Question", "Put your question here about the document")

def extract_text_with_pymupdf(file_path):
    text = ""
    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text() + "\n"
    return text

async def process_file():
    if uploaded_file and question:
        # Save the uploaded file with appropriate extension
        file_path = f"file.{uploaded_file.name.split('.')[-1]}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize the LLM and embeddings with the Google API key
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=os.environ["GOOGLE_API_KEY"])
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Extract text using PyMuPDF
        if file_path.endswith(".pdf"):
            all_text = extract_text_with_pymupdf(file_path)
        else:
            # Load the document using existing loaders for other formats
            def get_loader(file_path):
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension == ".csv":
                    return UnstructuredCSVLoader(file_path, mode="elements", encoding="utf8", errors="ignore")
                elif file_extension == ".xlsx":
                    return UnstructuredExcelLoader(file_path, mode="elements")
                elif file_extension == ".docx":
                    return Docx2txtLoader(file_path)
                elif file_extension == ".pptx":
                    return UnstructuredPowerPointLoader(file_path)
                else:
                    st.error("Unsupported file type")
                    return None
            
            loader = get_loader(file_path)
            
            if not loader:
                return

            try:
                docs = loader.load()
                all_text = " ".join([doc.page_content for doc in docs])
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return
        
        # Debug: Show the total length of the extracted text
        st.markdown(f"**<span style='font-size:16px;'>Total length of the extracted text: {len(all_text)} characters</span>**", unsafe_allow_html=True)
        
        # Enforce splitting into smaller chunks using CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(all_text)

        st.markdown(f"**<span style='font-size:16px;'>Total chunks created: {len(chunks)}</span>**", unsafe_allow_html=True)  # Verify the number of chunks

        # Preview the first few chunks
        for idx, chunk in enumerate(chunks[:3]):
            st.write(f"Chunk {idx + 1}: {chunk[:500]}...")  # Display a preview of each chunk

        # Create the FAISS vector store with the chunks
        document_search = FAISS.from_texts(chunks, embeddings)

        # Check if the vector store is properly initialized
        if document_search:
            query_embedding = embeddings.embed_query(question)
            results = document_search.similarity_search_by_vector(query_embedding, k=3)  # Retrieve top 3 chunks

            if results:
                retrieved_texts = " ".join([result.page_content for result in results])
            else:
                retrieved_texts = "No matching document found in the database."
        else:
            st.error("Vector database not initialized.")
            return

        # Display the similarity search result
        st.markdown("### Relevant Document Sections Based on the Question")
        st.write(retrieved_texts)

        # Augment the LLM response with retrieved documents using RAG
        template = """
        Based on the following retrieved context:
        "{retrieved_texts}"
        
        Answer the question: {question}
        
        Answer:"""
        prompt = PromptTemplate(input_variables=["retrieved_texts", "question"], template=template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        response = llm_chain.run(retrieved_texts=retrieved_texts, question=question)

        # Display the LLM response
        st.markdown("### Augmented Response from the LLM")
        st.write(response)

        # Clean up the temporary file
        os.remove(file_path)

if st.button("Process"):
    asyncio.run(process_file())
