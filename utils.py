import re
import json
import os
from typing import Dict, List, Set
from pathlib import Path

def clean_topic_name(topic: str) -> str:
    """Convert topic to valid folder name"""
    return re.sub(r'[^\w\-_]', '_', topic.lower().strip())

def clean_text(text: str) -> str:
    """Remove citations and clean text"""
    # Remove citations like [1], [2,3], [4-6]
    text = re.sub(r'\[\d+(?:[,-]\d+)*\]', '', text)
    # Remove other citation patterns
    text = re.sub(r'\(\w+\s*et\s*al\.\s*,\s*\d{4}\)', '', text)
    text = re.sub(r'\w+\s*et\s*al\.\s*\(\d{4}\)', '', text)
    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_review_prompt() -> str:
    """Get the prompt template for literature review generation"""
    return """You are a technical research expert writing a detailed literature review.
Generate a comprehensive literature review that integrates information from the provided research papers.

Topic: {topic}

Here are relevant excerpts from papers with citations:
{context}

Write a detailed technical literature review that:
1. Focuses on technical details, methodologies, and implementations
2. Compares and contrasts specific technical approaches
3. Analyzes quantitative results and performance metrics
4. Discusses technical challenges and solutions
5. Identifies specific technical gaps and future research directions

Structure the review with these sections:
- Technical Background
- Methodological Analysis
- Results and Performance Analysis
- Technical Challenges
- Future Directions

Requirements:
- Use numeric citations [X] exactly as provided in the context
- Every technical claim must have a citation
- Maintain technical precision
- Compare approaches between papers
- Focus on specific technical details and metrics

Note: Keep all citations in [X] format and ensure they match the provided paper IDs."""

def get_metadata_prompt() -> str:
    """Get the prompt template for metadata extraction"""
    return """Extract the following information from this research paper text and format as JSON:
    {
        "authors": "author names",
        "year": "publication year",
        "key_topics": ["list", "of", "main", "topics"],
        "abstract": "2-3 sentence summary"
    }

    Text: {text}"""

def save_paper_metadata(topic_path: str, papers_metadata: Dict) -> None:
    """Save paper metadata to disk"""
    metadata_path = os.path.join(topic_path, "papers_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(papers_metadata, f, indent=2)

def load_existing_papers(base_path: str) -> Dict[str, Dict]:
    """Load existing papers metadata by topic"""
    papers_by_topic = {}
    if not os.path.exists(base_path):
        return papers_by_topic
        
    for topic_dir in Path(base_path).iterdir():
        if topic_dir.is_dir():
            metadata_path = topic_dir / "papers_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    papers_by_topic[topic_dir.name] = json.load(f)
    
    return papers_by_topic

def load_existing_topics(base_path: str) -> Set[str]:
    """Load existing topics from research_papers directory"""
    if not os.path.exists(base_path):
        return set()
    return {p.name for p in Path(base_path).iterdir() if p.is_dir()}