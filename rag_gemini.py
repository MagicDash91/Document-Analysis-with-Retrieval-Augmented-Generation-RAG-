import asyncio
import os
import streamlit as st
import fitz  # PyMuPDF
from nltk import word_tokenize
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA"

# Title of the app
st.title("PDF Document Analysis with Retrieval-Augmented Generation (RAG) and Zero-Shot")

# Upload the file (restrict to PDF)
uploaded_file = st.file_uploader("Upload PDF Document:", type=["pdf"])
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
        # Save the uploaded file as a PDF
        file_path = f"file.pdf"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize the LLM and embeddings with the Google API key
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=os.environ["GOOGLE_API_KEY"])
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Extract text using PyMuPDF
        all_text = extract_text_with_pymupdf(file_path)

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
        rag_template = """
        Based on the following retrieved context:
        "{retrieved_texts}"
        
        Answer the question: {question}
        
        Answer:"""
        rag_prompt = PromptTemplate(input_variables=["retrieved_texts", "question"], template=rag_template)
        rag_llm_chain = LLMChain(llm=llm, prompt=rag_prompt)

        rag_response = rag_llm_chain.run(retrieved_texts=retrieved_texts, question=question)

        # Display the LLM response from RAG
        st.markdown("### Augmented Response from the LLM (RAG)")
        st.write(rag_response)



        # Now, handle zero-shot capability by querying the LLM directly without any retrieved documents
        point_template = """
        Summarize the key points from the document in bullet form.

        Document: "{all_text}"

        Summary (use bullet points):
        """
        point_prompt = PromptTemplate(input_variables=["all_text"], template=point_template)
        point_llm_chain = LLMChain(llm=llm, prompt=point_prompt)

        # Make sure to pass `all_text` to the run function
        point_response = point_llm_chain.run(all_text=all_text)

        # Display the zero-shot response
        st.markdown("### Key Points based on the Documents")
        st.write(point_response)


        # Now, handle zero-shot capability by querying the LLM directly without any retrieved documents
        sentiment_template = """
        List the positive, neagtive and neutral sentiment from the document in bullet form.

        Document: "{all_text}"

        Summary (use bullet points):
        """
        sentiment_prompt = PromptTemplate(input_variables=["all_text"], template=sentiment_template)
        sentiment_llm_chain = LLMChain(llm=llm, prompt=sentiment_prompt)

        # Make sure to pass `all_text` to the run function
        sentiment_response = sentiment_llm_chain.run(all_text=all_text)

        # Display the zero-shot response
        st.markdown("### Sentiment Analysis based from the Documents")
        st.write(sentiment_response)



        # Now, handle zero-shot capability by querying the LLM directly without any retrieved documents
        zeroshot_template = """
        Classify this document into one of these categories: "Financial Report", "Scientific Paper", "News Article", "Legal Document", "Tutorial", "Documentation" or  "Other"

        Document: "{all_text}"

        Category:
        Short Reason why it fit into that category:
        """
        zeroshot_prompt = PromptTemplate(input_variables=["all_text"], template=zeroshot_template)
        zeroshot_llm_chain = LLMChain(llm=llm, prompt=zeroshot_prompt)

        # Make sure to pass `all_text` to the run function
        zeroshot_response = zeroshot_llm_chain.run(all_text=all_text)

        # Display the zero-shot response
        st.markdown("### Zero-Shot Response from the LLM")
        st.write(zeroshot_response)

        # Clean up the temporary file
        os.remove(file_path)

if st.button("Process"):
    asyncio.run(process_file())
