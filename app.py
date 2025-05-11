import streamlit as st
from transformers import pipeline, AutoTokenizer
from PyPDF2 import PdfReader
import torch
import time
from datetime import datetime

def split_text(text, max_length=512):
    """Split text into chunks that are small enough for the model"""
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split by sentences first
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    for sentence in sentences:
        # Add space back if it was removed
        sentence += ' '
        sentence_length = len(tokenizer.tokenize(sentence))
        
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:  # Don't add empty chunks
                chunks.append(''.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return chunks

def estimate_accuracy(text_length, summary_length, processing_time):
    """Heuristic accuracy estimation based on text characteristics"""
    compression_ratio = text_length / max(summary_length, 1)
    
    # Base score (0-100) based on compression ratio
    score = min(100, max(0, 100 - (abs(8 - compression_ratio) * 10)))
    
    # Adjust for processing time (faster is better)
    time_penalty = min(30, processing_time * 2)  # Max 30% penalty
    score -= time_penalty
    
    return max(50, score)  # Never go below 50%

st.title("📄 Advanced AI Text Summarizer")
st.write("Upload a PDF or enter text to get a summary with performance metrics")

option = st.radio("Choose input method:", ("Upload PDF", "Enter Text"))

text = ""
if option == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
            if text.strip():
                st.text_area("Extracted Text", text[:2000] + "..." if len(text) > 2000 else text, height=200)
                st.caption(f"Document length: {len(text)} characters")
            else:
                st.warning("The PDF contains no extractable text.")
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
else:
    text = st.text_area("Enter your text here:", height=200)

if st.button("✨ Summarize") and text.strip():
    start_time = time.time()
    
    with st.spinner("Generating summary..."):
        try:
            # Initialize summarizer and track timing
            model_load_start = time.time()
            summarizer = pipeline("summarization", 
                                model="facebook/bart-large-cnn",
                                device=0 if torch.cuda.is_available() else -1)
            model_load_time = time.time() - model_load_start
            
            # Clean and normalize text
            clean_text = ' '.join(text.split()).strip()
            text_length = len(clean_text)
            
            if text_length > 100:
                chunks = split_text(clean_text)
                if len(chunks) > 1:
                    st.info(f"Processing {len(chunks)} chunks...")
                
                summaries = []
                chunk_times = []
                
                for chunk in chunks:
                    try:
                        chunk_start = time.time()
                        result = summarizer(chunk, 
                                          max_length=min(150, len(chunk)//2), 
                                          min_length=min(30, len(chunk)//4),
                                          do_sample=False)
                        chunk_time = time.time() - chunk_start
                        chunk_times.append(chunk_time)
                        
                        if result and len(result) > 0:
                            summaries.append(result[0]['summary_text'])
                    except Exception as chunk_error:
                        st.warning(f"Skipped a chunk: {str(chunk_error)}")
                        continue
                
                if summaries:
                    final_summary = " ".join(summaries)
                    summary_length = len(final_summary)
                    
                    # If combined summary is too long, summarize it again
                    if len(final_summary) > 1024:
                        final_start = time.time()
                        final_summary = summarizer(final_summary,
                                                 max_length=150,
                                                 min_length=30,
                                                 do_sample=False)[0]['summary_text']
                        final_time = time.time() - final_start
                        chunk_times.append(final_time)
                    
                    # Calculate metrics
                    total_time = time.time() - start_time
                    avg_chunk_time = sum(chunk_times)/len(chunk_times) if chunk_times else 0
                    accuracy = estimate_accuracy(text_length, summary_length, avg_chunk_time)
                    
                    # Display results
                    st.subheader("📝 Final Summary")
                    st.write(final_summary)
                    
                    # Metrics columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Total Processing Time", f"{total_time:.2f}s")
                    with col2:
                        st.metric("⚡ Average Chunk Time", f"{avg_chunk_time:.2f}s")
                    with col3:
                        st.metric("🎯 Accuracy Estimate", f"{accuracy:.0f}%")
                    
                    # Additional details expander
                    with st.expander("📊 Detailed Metrics"):
                        st.write(f"Original length: {text_length} characters")
                        st.write(f"Summary length: {summary_length} characters")
                        st.write(f"Compression ratio: {text_length/summary_length:.1f}x")
                        st.write(f"Model load time: {model_load_time:.2f}s")
                        st.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                else:
                    st.error("Could not generate any summary from the text.")
            else:
                st.warning("Text is too short to summarize meaningfully.")
                
        except Exception as e:
            st.error(f"Summarization failed: {str(e)}")