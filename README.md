# Research Paper Analysis System

A streamlit-based application that helps researchers analyze academic papers, generate literature reviews, and interact with document content using advanced language models.

## Features

### 1. Literature Review Generation
- Upload multiple PDF papers on a specific topic
- Organize papers by research topics
- Generate comprehensive literature reviews using GPT-4 or Gemini 1.5 Pro
- Interactive citations with source context
- Downloadable reviews in Markdown format

### 2. Document Analysis
- Single document deep-dive analysis
- Generate detailed document summaries
- Interactive Q&A with the document content
- Citation-supported responses

### 3. Document Management
- View PDFs directly in the application
- Download original documents
- Delete documents with confirmation
- Organize papers by topics
- Track paper metadata (authors, year, key topics)

### 4. Interactive Interface
- Clean, two-column layout
- Tab-based navigation
- Progress tracking for file processing
- Interactive citations with source context
- Error handling and user feedback

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd research-paper-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install required spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Configuration

1. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
```

2. Ensure you have appropriate API access:
- OpenAI API access for GPT-4
- Google AI API access for Gemini 1.5 Pro

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Access the web interface at `http://localhost:8501`

### Literature Review Generation
1. Create or select a topic
2. Upload PDF research papers
3. Select preferred language model
4. Generate comprehensive literature review
5. Download review in Markdown format

### Document Analysis
1. Select a document from uploaded papers
2. Generate detailed summary
3. Ask questions about the document
4. View responses with source citations

## Project Structure
```
project_root/
├── main.py                 # Main Streamlit application
├── rag_system.py          # Core RAG implementation
├── utils.py               # Utility functions
├── components/           
│   └── review_display.py  # Review visualization
├── research_papers/       # Data storage directory
│   └── topic_name/        # Topic-specific directories
│       ├── paper_1.pdf    # Uploaded papers
│       ├── papers_metadata.json  # Paper metadata
│       └── vector_store/  # FAISS vector storage
└── requirements.txt       # Project dependencies
```

## Dependencies
- streamlit
- langchain
- openai
- google-generativeai
- faiss-cpu
- spacy
- transformers
- torch
- python-dotenv

## Technical Details

### RAG System
- Uses FAISS for vector storage and retrieval
- Implements semantic chunking for better context
- Maintains citation tracking
- Hybrid search combining semantic and keyword matching

### Language Models
- GPT-4 Turbo via OpenAI API
- Gemini 1.5 Pro via Google AI API

### Document Processing
- PDF text extraction
- Metadata extraction using LLMs
- Semantic section identification
- Vector embeddings using SPECTER model

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- OpenAI for GPT-4 API
- Google for Gemini API
- Streamlit for the web framework
- FAISS for vector storage
- Langchain for RAG implementation

## Support

For support or questions, please open an issue in the repository.