import os
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import tempfile
import json

from transformers import AutoTokenizer, AutoModel
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from utils import (
    clean_topic_name, clean_text, get_review_prompt, get_metadata_prompt,
    save_paper_metadata, load_existing_papers, load_existing_topics
)

class EnhancedResearchRAG:
    def __init__(self):
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OpenAI API key not found!")
        if not os.getenv('GOOGLE_API_KEY'):
            raise ValueError("Google API key not found!")
            
        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.model = AutoModel.from_pretrained('allenai/specter').to(self.device)
        
        # Initialize LLMs
        self.openai_model = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.2
        )
        
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.gemini_model = GoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.2
        )
        
        # Initialize embeddings
        self.openai_embeddings = OpenAIEmbeddings()
        
        # Initialize NLP
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize storage
        self.base_path = "research_papers"
        os.makedirs(self.base_path, exist_ok=True)
        
        # Load existing data
        self.topics = load_existing_topics(self.base_path)
        self.papers_by_topic = load_existing_papers(self.base_path)
        self.vector_stores = {}
        self.paper_count = self._get_max_paper_id()
        
        # Define sections for retrieval
        self.section_queries = {
            "Technical Background": [
                "technical foundation background theory methodology",
                "core concepts technical approach",
                "fundamental technical principles"
            ],
            "Methodological Analysis": [
                "specific methodology implementation details",
                "technical process methodology steps",
                "experimental setup methodology"
            ],
            "Results": [
                "quantitative results metrics performance",
                "experimental findings measurements",
                "performance evaluation metrics"
            ],
            "Technical Challenges": [
                "technical limitations challenges problems",
                "implementation difficulties issues",
                "technical constraints barriers"
            ],
            "Future Directions": [
                "future work research directions",
                "proposed improvements extensions",
                "potential developments recommendations"
            ]
        }

    def _get_max_paper_id(self) -> int:
        """Get maximum paper ID from existing papers"""
        max_id = 0
        for topic_papers in self.papers_by_topic.values():
            if topic_papers:
                max_id = max(max_id, max(int(k) for k in topic_papers.keys()))
        return max_id

    def _extract_title_from_pdf(self, text: str, paper_file) -> str:
        """Extract title from first page content"""
        first_page = text[:1000]
        title_prompt = """Extract the exact paper title from this text. 
        Return ONLY the title, no quotes or extra text. 
        If no clear title is found, return 'Unknown Title'.
        
        Text: {text}"""
        
        try:
            title = self.openai_model.predict(title_prompt.format(text=first_page)).strip()
            return title if title != 'Unknown Title' else os.path.splitext(paper_file.name)[0]
        except:
            return os.path.splitext(paper_file.name)[0]

    def _get_specter_embedding(self, text: str) -> np.ndarray:
        """Get embeddings using SPECTER model"""
        inputs = self.tokenizer(text, padding=True, truncation=True, 
                              max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings[0]

    def _extract_sections(self, text: str) -> List[Dict[str, str]]:
        """Extract sections from research paper text"""
        section_patterns = [
            "abstract", "introduction", "background", "related work",
            "methodology", "methods", "experimental", "results",
            "discussion", "conclusion", "references"
        ]
        
        sections = []
        current_text = []
        current_section = "introduction"
        
        for line in text.split('\n'):
            line_lower = line.lower().strip()
            is_header = any(pattern in line_lower for pattern in section_patterns)
            
            if is_header:
                if current_text:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_text)
                    })
                current_section = line.strip()
                current_text = []
            else:
                current_text.append(line)
        
        if current_text:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_text)
            })
        
        return sections

    def _semantic_chunking(self, text: str) -> List[str]:
        """Chunk text based on semantic boundaries"""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        MAX_CHUNK_SIZE = 100
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(sent_text.split())
            
            if current_length + sent_length > MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sent_text]
                current_length = sent_length
            else:
                current_chunk.append(sent_text)
                current_length += sent_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _get_vector_store(self, topic: str) -> Optional[FAISS]:
        """Get vector store for topic"""
        if topic not in self.vector_stores:
            topic_path = os.path.join(self.base_path, clean_topic_name(topic))
            vector_store_path = os.path.join(topic_path, "vector_store")
            
            if os.path.exists(vector_store_path):
                try:
                    self.vector_stores[topic] = FAISS.load_local(
                        vector_store_path,
                        self.openai_embeddings,
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    print(f"Error loading vector store: {e}")
                    self.vector_stores[topic] = None
            else:
                self.vector_stores[topic] = None
        
        return self.vector_stores[topic]

    def _save_vector_store(self, vector_store: FAISS, topic: str) -> None:
        """Save vector store to disk"""
        topic_path = os.path.join(self.base_path, clean_topic_name(topic))
        vector_store_path = os.path.join(topic_path, "vector_store")
        vector_store.save_local(vector_store_path)
        self.vector_stores[topic] = vector_store

    def _get_citation_context(self, chunk: str, paper_id: str, paper_info: Dict) -> Dict:
        """Create detailed citation context with unique citation ID"""
        citation_id = f"{paper_id}_{hash(chunk) % 10000}"
        
        return {
            "citation_id": citation_id,
            "paper_id": paper_id,
            "title": paper_info['metadata'].get('title', os.path.splitext(paper_info['original_name'])[0]),
            "filename": paper_info['original_name'],
            "quote": chunk,
            "section": paper_info['metadata'].get('section', 'Unknown'),
            "year": paper_info['metadata'].get('year', 'Unknown'),
            "authors": paper_info['metadata'].get('authors', 'Unknown')
        }

    def _get_relevant_chunks(self, query: str, vector_store: FAISS, topic: str, k: int = 3) -> List[Tuple[str, str]]:
        """Get more relevant chunks using hybrid search"""
        results = vector_store.similarity_search(
            query,
            k=k,
            include_metadata=True
        )
        
        doc = self.nlp(query)
        key_phrases = [token.text.lower() for token in doc if not token.is_stop]
        
        scored_chunks = []
        for doc in results:
            content_lower = doc.page_content.lower()
            matches = sum(1 for phrase in key_phrases if phrase in content_lower)
            
            similarity = vector_store.similarity_search_with_score(
                doc.page_content,
                k=1
            )[0][1]
            
            final_score = matches * (1 / (1 + similarity))
            scored_chunks.append((doc, final_score))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [(chunk.page_content, chunk.metadata['paper_id']) 
                for chunk, _ in scored_chunks[:k]]

    def delete_paper(self, topic: str, paper_id: str) -> None:
        """Delete a paper and its associated data"""
        if topic not in self.papers_by_topic or paper_id not in self.papers_by_topic[topic]:
            return
        
        paper_info = self.papers_by_topic[topic][paper_id]
        pdf_path = paper_info['path']
        
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        del self.papers_by_topic[topic][paper_id]
        
        topic_path = os.path.join(self.base_path, clean_topic_name(topic))
        save_paper_metadata(topic_path, self.papers_by_topic[topic])
        
        vector_store = self._get_vector_store(topic)
        if vector_store is not None:
            documents = []
            for doc in vector_store.docstore._dict.values():
                if doc.metadata.get('paper_id') != paper_id:
                    documents.append(doc)
            
            if documents:
                new_vector_store = FAISS.from_documents(documents, self.openai_embeddings)
                self._save_vector_store(new_vector_store, topic)
            else:
                vector_store_path = os.path.join(topic_path, "vector_store")
                if os.path.exists(vector_store_path):
                    import shutil
                    shutil.rmtree(vector_store_path)
                self.vector_stores[topic] = None

    def process_pdf(self, pdf_file, topic: str) -> None:
        """Process PDF and store with topic organization"""
        topic = clean_topic_name(topic)
        topic_path = os.path.join(self.base_path, topic)
        os.makedirs(topic_path, exist_ok=True)
        
        file_id = str(self.paper_count + 1)
        new_filename = f"paper_{file_id}_{pdf_file.name}"
        pdf_path = os.path.join(topic_path, new_filename)
        
        with open(pdf_path, 'wb') as f:
            pdf_content = pdf_file.getvalue()
            f.write(pdf_content)
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])
        
        title = self._extract_title_from_pdf(full_text, pdf_file)
        
        try:
            metadata_response = self.openai_model.predict(
                get_metadata_prompt().format(text=full_text[:4000])
            )
            
            metadata_json = metadata_response.strip()
            if not metadata_json.startswith('{'): 
                metadata_json = metadata_json[metadata_json.find('{'):]
            if not metadata_json.endswith('}'): 
                metadata_json = metadata_json[:metadata_json.rfind('}')+1]
                
            paper_metadata = json.loads(metadata_json)
            
            required_fields = ['authors', 'year', 'key_topics', 'abstract']
            for field in required_fields:
                if field not in paper_metadata:
                    paper_metadata[field] = 'Unknown' if field != 'key_topics' else []
                    
            paper_metadata['title'] = title
            
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            paper_metadata = {
                "title": title,
                "authors": "Unknown",
                "year": "Unknown",
                "key_topics": [],
                "abstract": ""
            }
        
        sections = self._extract_sections(full_text)
        documents = []
        
        for section in sections:
            cleaned_content = clean_text(section['content'])
            chunks = self._semantic_chunking(cleaned_content)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'section': section['title'],
                        'chunk_index': i,
                        'topic': topic,
                        'paper_id': file_id,
                        'filename': new_filename,
                        'title': paper_metadata['title']
                    }
                )
                documents.append(doc)
        
        vector_store = self._get_vector_store(topic)
        if vector_store is None:
            vector_store = FAISS.from_documents(documents, self.openai_embeddings)
        else:
            vector_store.add_documents(documents)
        
        self._save_vector_store(vector_store, topic)
        
        if topic not in self.papers_by_topic:
            self.papers_by_topic[topic] = {}
        
        self.papers_by_topic[topic][file_id] = {
            'filename': new_filename,
            'original_name': pdf_file.name,
            'path': pdf_path,
            'metadata': paper_metadata
        }
        
        save_paper_metadata(topic_path, self.papers_by_topic[topic])
        self.topics.add(topic)
        self.paper_count = int(file_id)

    def generate_review(self, topic: str, model_choice: str = "gpt-4") -> Tuple[str, Dict]:
        """Generate review with unique citations for each reference"""
        vector_store = self._get_vector_store(topic)
        if not vector_store:
            return "No papers have been uploaded for this topic.", {}

        section_chunks = {}
        citations_data = {}
        citation_mapping = {}
        
        for section, queries in self.section_queries.items():
            chunks = []
            for query in queries:
                results = self._get_relevant_chunks(query, vector_store, topic)
                for content, paper_id in results:
                    citation_context = self._get_citation_context(
                        content,
                        paper_id,
                        self.papers_by_topic[topic][paper_id]
                    )
                    citation_id = citation_context['citation_id']
                    
                    citations_data[citation_id] = citation_context
                    
                    if paper_id not in citation_mapping:
                        citation_mapping[paper_id] = []
                    citation_mapping[paper_id].append(citation_id)
                    
                    chunks.append((content, citation_id))
            section_chunks[section] = chunks

        context = self._format_chunks_with_citations(section_chunks)
        
        model = self.openai_model if model_choice == "gpt-4" else self.gemini_model
        
        review = model.predict(get_review_prompt().format(
            topic=topic,
            context=context
        ))
        
        return review, citations_data

    def _format_chunks_with_citations(self, section_chunks: Dict) -> str:
        """Format chunks with unique citations"""
        formatted_sections = []
        for section, chunks in section_chunks.items():
            section_texts = [
                f"{content} [{citation_id}]"
                for content, citation_id in chunks
            ]
            formatted_sections.append(
                f"\n{section}:\n" + "\n".join(section_texts)
            )
        return "\n\n".join(formatted_sections)
    
    def generate_document_summary(self, topic: str, paper_id: str) -> Tuple[str, Dict]:
        """Generate a comprehensive summary of a single document with citations"""
        if topic not in self.papers_by_topic or paper_id not in self.papers_by_topic[topic]:
            raise ValueError("Document not found")
        
        paper_info = self.papers_by_topic[topic][paper_id]
        vector_store = self._get_vector_store(topic)
        
        # Get relevant chunks for each section
        sections = {
            "Main Objectives": "research goals objectives purpose main aim",
            "Methodology": "methodology approach method technique implementation",
            "Results": "results findings outcomes performance metrics",
            "Conclusions": "conclusions implications findings contribution",
            "Technical Details": "technical details implementation specifics architecture"
        }
        
        chunks_by_section = {}
        citations_data = {}
        
        for section, query in sections.items():
            chunks = []
            results = self._get_relevant_chunks(query, vector_store, topic, k=3)
            for content, pid in results:
                if pid == paper_id:  # Only use chunks from this paper
                    citation_context = self._get_citation_context(
                        content, pid, paper_info
                    )
                    citations_data[citation_context['citation_id']] = citation_context
                    chunks.append((content, citation_context['citation_id']))
            chunks_by_section[section] = chunks
        
        # Create summary prompt with sections
        summary_prompt = """Generate a comprehensive summary of this research paper.
        For each section, use the provided excerpts and include citations in [citation_id] format.
        
        Paper Title: {title}
        
        Relevant excerpts by section:
        {context}
        
        Generate a detailed summary that covers each section, using appropriate citations for specific claims and findings.
        Maintain technical precision and ensure every major claim has a citation."""
        
        # Format context with sections and citations
        context = []
        for section, chunks in chunks_by_section.items():
            section_chunks = [f"{content} [{cid}]" for content, cid in chunks]
            context.append(f"{section}:\n" + "\n".join(section_chunks))
        
        formatted_context = "\n\n".join(context)
        
        try:
            summary = self.openai_model.predict(
                summary_prompt.format(
                    title=paper_info['metadata']['title'],
                    context=formatted_context
                )
            )
            return summary, citations_data
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")

    def chat_about_document(self, topic: str, paper_id: str, 
                        user_query: str, chat_history: List[Dict]) -> Tuple[str, Dict]:
        """Handle chat interactions about a specific document with citations"""
        if topic not in self.papers_by_topic or paper_id not in self.papers_by_topic[topic]:
            raise ValueError("Document not found")
        
        paper_info = self.papers_by_topic[topic][paper_id]
        vector_store = self._get_vector_store(topic)
        
        # Get relevant chunks
        results = self._get_relevant_chunks(user_query, vector_store, topic, k=5)
        citations_data = {}
        relevant_chunks = []
        
        # Process chunks and create citations
        for content, pid in results:
            if pid == paper_id:  # Only use chunks from this paper
                citation_context = self._get_citation_context(
                    content, pid, paper_info
                )
                citations_data[citation_context['citation_id']] = citation_context
                relevant_chunks.append((content, citation_context['citation_id']))
        
        # Format chat prompt with citations
        chat_context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in chat_history[-5:]  # Include last 5 messages for context
        ])
        
        chat_prompt = """Answer the user's question about this research paper.
        Use the provided relevant excerpts and include citations in [citation_id] format.
        
        Paper Title: {title}
        
        Relevant excerpts:
        {context}
        
        Chat history:
        {chat_history}
        
        User question: {question}
        
        Provide a detailed response based on the paper content. Include citations for specific claims.
        If the answer cannot be found in the provided excerpts, say so clearly."""
        
        # Format context with citations
        formatted_chunks = [f"{content} [{cid}]" for content, cid in relevant_chunks]
        formatted_context = "\n".join(formatted_chunks)
        
        try:
            response = self.openai_model.predict(
                chat_prompt.format(
                    title=paper_info['metadata']['title'],
                    context=formatted_context,
                    chat_history=chat_context,
                    question=user_query
                )
            )
            return response, citations_data
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")