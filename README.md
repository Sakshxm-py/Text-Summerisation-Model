# 🧠 AI Text Summarizer

This is a powerful and interactive AI-powered text summarization tool built with [Streamlit](https://streamlit.io/) and the `facebook/bart-large-cnn` model from Hugging Face Transformers. It allows users to upload PDF files or paste in text to generate concise summaries, along with performance metrics like processing time and estimated accuracy.

## 🚀 Features

- 📄 Upload and summarize PDFs
- 📝 Paste and summarize plain text
- ⚙️ Chunk-based processing for long texts
- 📊 Real-time performance metrics:
  - Total processing time
  - Average time per chunk
  - Heuristic-based accuracy estimation
- 📉 Auto re-summarization for long summaries
- 📅 Timestamped metrics display

## 🧰 Requirements
Install the required dependencies with :

```bash
pip install -r requirements.txt
```


Dependencies include:

- streamlit

- transformers

- torch

- PyPDF2

## ⚙️ Model Used

- facebook/bart-large-cnn via Hugging Face transformers.pipeline("summarization")
- Automatically detects and uses GPU (CUDA) if available.

## 📌 Notes

- Designed for medium-to-large documents.

- Automatically splits long text into manageable chunks for accurate summarization.

- The model performs better on well-formatted English text.

- Includes fallback logic for unextractable PDFs.

## 🖥️ Demo

Launch the app locally:

```bash
streamlit run app.py
