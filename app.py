import streamlit as st
import PyPDF2
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import tempfile
import time
import re
import tiktoken
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="PDF Knowledge Base RAG",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'token_count' not in st.session_state:
    st.session_state.token_count = 0
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# Configure OpenAI API
def configure_openai():
    # Try environment variable first
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If not in environment, try secrets
    if not api_key:
        try:
            api_key = st.secrets["openai_api_key"]
        except:
            # If not in secrets, get from user input
            api_key = st.session_state.get("openai_api_key", None)
    
    if api_key:
        openai.api_key = api_key
        return True
    return False

# Function to count tokens
def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to extract text from PDF with page numbers
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text_with_pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                # Clean text: remove multiple spaces, newlines, etc.
                text = re.sub(r'\s+', ' ', text).strip()
                text_with_pages.append({
                    'text': text,
                    'page': page_num
                })
        return text_with_pages, None
    except Exception as e:
        return None, f"Error extracting text from PDF: {str(e)}"

# Function to chunk text into smaller pieces with overlap
def chunk_text(text, filename, page_num, max_tokens=500, overlap=50):
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    overlap_words = []
    
    for word in words:
        current_chunk.append(word)
        current_length += 1
        
        if current_length >= max_tokens:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'filename': filename,
                'page': page_num,
                'token_count': count_tokens(chunk_text)
            })
            
            # Keep the overlap words for the next chunk
            overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_words.copy()
            current_length = len(current_chunk)
    
    # Don't forget the last chunk
    if current_chunk and (len(chunks) == 0 or current_chunk != overlap_words):
        chunk_text = " ".join(current_chunk)
        chunks.append({
            'content': chunk_text,
            'filename': filename,
            'page': page_num,
            'token_count': count_tokens(chunk_text)
        })
    
    return chunks

# Function to get embeddings with rate limit handling
import logging

def get_embeddings(text, retry_count=3):
    for attempt in range(retry_count):
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response['data'][0]['embedding']
        
        except openai.error.RateLimitError as e:
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt
                st.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)  # Exponential backoff
            else:
                st.error("Rate limit exceeded. Please try again later.")
                raise
        except openai.error.AuthenticationError as e:
            st.error(f"Authentication error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error generating embeddings: {str(e)}")
            return None

# Function to search content
def search_content(df, input_text, top_k=3):
    try:
        if df is None or df.empty:
            return None, "No document data available"
        
        embedded_value = get_embeddings(input_text)
        if embedded_value is None:
            return None, "Failed to generate embeddings for your query"
        
        # Calculate similarity
        df["similarity"] = df['embeddings'].apply(
            lambda x: cosine_similarity(
                np.array(x).reshape(1, -1), 
                np.array(embedded_value).reshape(1, -1)
            )[0][0] if x else 0
        )
        
        # Sort and get top results
        results = df.sort_values('similarity', ascending=False).head(top_k)
        return results, None
    except Exception as e:
        return None, f"Error searching content: {str(e)}"

# Function to generate response
def generate_output(input_prompt, similar_content, model="gpt-4o-mini"):
    try:
        # Combine relevant contexts
        context = "\n\n".join([
            f"From {row['filename']}, Page {row['page']}:\n{row['content']}"
            for _, row in similar_content.iterrows()
        ])
        
        # Create a prompt with context
        prompt = f"""
        You are a knowledgeable assistant that provides accurate information based on documents.
        
        QUERY: {input_prompt}
        
        RELEVANT DOCUMENT SECTIONS:
        {context}
        
        Based solely on the information provided in these document sections, answer the query. 
        If the information is not in the documents, say "I don't have enough information in the documents to answer this question."
        Do not make up information. Cite the document name and page number in your answer when appropriate.
        """
        
        completion = openai.ChatCompletion.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based strictly on the provided document context."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content, None
    except Exception as e:
        return None, f"Error generating response: {str(e)}"

# Function to process uploaded files
def process_files(uploaded_files):
    if not uploaded_files:
        return None, "No files uploaded"
    
    documents = []
    total_token_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        if file_size > 5:
            return None, f"File {uploaded_file.name} exceeds 5MB size limit"
        
        status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
        
        # Save to temporary file to ensure PyPDF2 can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_filename = temp_file.name
        
        try:
            # Extract text from PDF
            with open(temp_filename, 'rb') as file:
                text_with_pages, error = extract_text_from_pdf(BytesIO(file.read()))
                
            # Clean up temporary file
            os.unlink(temp_filename)
            
            if error:
                return None, error
                
            # Process each page
            for item in text_with_pages:
                text = item['text']
                page_num = item['page']
                
                # Chunk the text
                chunks = chunk_text(text, uploaded_file.name, page_num)
                
                for chunk in chunks:
                    total_token_count += chunk['token_count']
                    
                    # Get embeddings
                    embedding = get_embeddings(chunk['content'])
                    if embedding:
                        chunk['embeddings'] = embedding
                        documents.append(chunk)
                    
            # Update progress bar
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            # Clean up temporary file in case of error
            try:
                os.unlink(temp_filename)
            except:
                pass
            return None, f"Error processing {uploaded_file.name}: {str(e)}"
    
    status_text.text("Processing complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    if documents:
        df = pd.DataFrame(documents)
        st.session_state.token_count = total_token_count
        return df, None
    else:
        return None, "No text content could be extracted from the uploaded files"

# Main Streamlit UI
def main():
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 0px 16px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("Configuration")
        
        # OpenAI API configuration
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            st.session_state.openai_api_key = api_key
            openai.api_key = api_key
        
        # Model selection
        model = st.selectbox("Select OpenAI Model:", 
                            ["gpt-4o-mini"], 
                            index=0)
        
        # Chunk size configuration
        chunk_size = st.slider("Chunk Size (words):", 
                              min_value=100, 
                              max_value=1000, 
                              value=500, 
                              step=50)
        
        # Number of results to return
        top_k = st.slider("Number of document chunks to retrieve:", 
                         min_value=1, 
                         max_value=10, 
                         value=3)
        
        # Display token usage if documents are processed
        if st.session_state.token_count > 0:
            st.info(f"Total tokens used for embeddings: {st.session_state.token_count}")
        
        # Clear data button
        if st.button("Clear All Data"):
            st.session_state.dataframe = None
            st.session_state.documents_processed = False
            st.session_state.uploaded_files = []
            st.session_state.processing_complete = False
            st.session_state.token_count = 0
            st.session_state.error_message = None
            st.experimental_rerun()
    
    # Main content
    st.title("📚 PDF Knowledge Base RAG")
    st.write("Upload PDFs and ask questions about their content")
    
    if not st.session_state.get("openai_api_key"):
        st.warning("Please configure your OpenAI API key in the sidebar first")
        return
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "❓ Ask Questions", "ℹ️ Document Info"])
    
    # Tab 1: Upload Documents
    with tab1:
        st.header("Upload Your Documents")
        
        # File uploader
        uploaded_files = st.file_uploader("Upload PDF files (max 5MB each)", 
                                         type="pdf", 
                                         accept_multiple_files=True)
        
        # Process button
        col1, col2 = st.columns([1, 5])
        process_button = col1.button("Process Files")
        status_area = col2.empty()
        
        if process_button and uploaded_files:
            status_area.info("Processing files, please wait...")
            
            # Store uploaded files in session state
            st.session_state.uploaded_files = uploaded_files
            
            # Process the files
            df, error = process_files(uploaded_files)
            
            if error:
                status_area.error(error)
                st.session_state.error_message = error
            else:
                st.session_state.dataframe = df
                st.session_state.documents_processed = True
                st.session_state.processing_complete = True
                status_area.success(f"✅ {len(uploaded_files)} documents processed successfully!")
        
        elif process_button and not uploaded_files:
            status_area.warning("Please upload at least one PDF file")
        
        # Show success message if documents are already processed
        if st.session_state.processing_complete and not process_button:
            status_area.success(f"✅ {len(st.session_state.uploaded_files)} documents processed and ready for queries!")
    
    # Tab 2: Ask Questions
    with tab2:
        st.header("Ask Questions About Your Documents")
        
        if not st.session_state.documents_processed:
            st.info("Please upload and process documents first in the 'Upload Documents' tab")
        else:
            # Query input
            query = st.text_area("Enter your question about the documents:", height=100)
            query_button = st.button("Submit Question")
            
            if query_button and query:
                with st.spinner("Searching documents and generating answer..."):
                    # Search for relevant content
                    matching_content, search_error = search_content(
                        st.session_state.dataframe, 
                        query, 
                        top_k
                    )
                    
                    if search_error:
                        st.error(search_error)
                    elif matching_content is not None and not matching_content.empty:
                        # Generate response
                        response, response_error = generate_output(
                            query, 
                            matching_content,
                            model
                        )
                        
                        if response_error:
                            st.error(response_error)
                        else:
                            # Display response
                            st.subheader("Answer:")
                            st.markdown(response)
                            
                            # Display source information
                            st.subheader("Sources:")
                            for i, match in matching_content.iterrows():
                                with st.expander(f"Source {i+1}: {match['filename']} (Page {match['page']}) - Relevance: {match['similarity']:.2f}"):
                                    st.markdown(match['content'])
                    else:
                        st.warning("No relevant content found in your documents for this query")
            
            elif query_button:
                st.warning("Please enter a question")
    
    # Tab 3: Document Info
    with tab3:
        st.header("Document Information")
        
        if not st.session_state.documents_processed:
            st.info("Please upload and process documents first")
        else:
            # Display information about processed documents
            df = st.session_state.dataframe
            
            # Document statistics
            st.subheader("Document Statistics")
            col1, col2, col3 = st.columns(3)
            
            unique_docs = df['filename'].nunique()
            total_pages = df['page'].max()
            total_chunks = len(df)
            
            col1.metric("Documents", unique_docs)
            col2.metric("Total Pages", total_pages)
            col3.metric("Content Chunks", total_chunks)
            
            # Files overview
            st.subheader("Files Overview")
            file_stats = df.groupby('filename').agg({
                'page': 'max',
                'content': lambda x: sum(len(text.split()) for text in x)
            }).reset_index()
            file_stats.columns = ['Filename', 'Pages', 'Word Count']
            st.dataframe(file_stats, use_container_width=True)
            
            # Sample content
            st.subheader("Sample Content")
            selected_file = st.selectbox("Select a document:", df['filename'].unique())
            selected_page = st.slider(
                "Select a page:", 
                min_value=1, 
                max_value=int(df[df['filename'] == selected_file]['page'].max())
            )
            
            # Show sample content from the selected document and page
            sample_content = df[(df['filename'] == selected_file) & (df['page'] == selected_page)]
            if not sample_content.empty:
                st.markdown(f"**Content from {selected_file}, Page {selected_page}:**")
                with st.expander("View Content", expanded=True):
                    st.write(sample_content.iloc[0]['content'])
            else:
                st.info("No content found for the selected page")

if __name__ == "__main__":
    main()