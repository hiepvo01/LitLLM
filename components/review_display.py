import re
import os

def get_review_component(review: str, citations: dict, model_choice: str) -> str:
    """Generate HTML for interactive review display with improved formatting"""
    
    # CSS styles
    styles = """
    <style>
        .review-container {
            font-family: system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
        }
        h1, h2, h3, h4, h5, h6 {
            margin-top: 1.5em;
            margin-bottom: 0.75em;
            font-weight: 600;
            line-height: 1.25;
        }
        h1 { font-size: 2em; }
        h2 { font-size: 1.5em; }
        h3 { font-size: 1.25em; }
        p { margin: 1em 0; }
        .citation {
            color: #2563eb;
            cursor: pointer;
            position: relative;
            display: inline-block;
        }
        .citation:hover {
            text-decoration: underline;
        }
        .citation-popup {
            display: none;
            position: absolute;
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 16px;
            width: 300px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.9em;
            line-height: 1.4;
        }
        .citation:hover .citation-popup {
            display: block;
        }
        .model-info {
            background-color: #f3f4f6;
            padding: 8px 16px;
            border-radius: 6px;
            margin-bottom: 16px;
            font-size: 0.875rem;
        }
        .quote {
            border-left: 3px solid #e5e7eb;
            padding-left: 16px;
            margin: 8px 0;
            font-style: italic;
            max-height: 150px;
            overflow-y: auto;
        }
        .quote::-webkit-scrollbar {
            width: 6px;
        }
        .quote::-webkit-scrollbar-thumb {
            background-color: #cbd5e1;
            border-radius: 3px;
        }
    </style>
    """
    
    # JavaScript for citation handling
    script = """
    <script>
        function toggleCitation(id) {
            const popup = document.getElementById(`popup-${id}`);
            popup.style.display = popup.style.display === 'none' ? 'block' : 'none';
        }
    </script>
    """
    
    def process_markdown_headers(text):
        """Process markdown headers and wrap paragraphs"""
        lines = text.split('\n')
        processed_lines = []
        for line in lines:
            # Convert markdown headers to HTML
            line = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', line)
            line = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', line)
            line = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', line)
            # Wrap non-header text in paragraphs if not empty
            if not line.startswith('<h') and line.strip():
                line = f'<p>{line}</p>'
            processed_lines.append(line)
        return '\n'.join(processed_lines)
    
    def get_simplified_references(citations: dict) -> dict:
        """Create simplified reference mapping and list"""
        paper_mapping = {}
        unique_papers = {}
        
        for cit_id, citation in citations.items():
            paper_id = citation['paper_id']
            if paper_id not in unique_papers:
                unique_papers[paper_id] = {
                    'title': citation['title'],
                    'filename': citation['filename']
                }
            paper_mapping[cit_id] = paper_id
            
        return paper_mapping, unique_papers

    def process_citations(text, paper_mapping):
        """Process citations while maintaining detailed popups"""
        citation_pattern = r'\[([0-9]+_[0-9]+)\]'
        
        def replace_citation(match):
            citation_id = match.group(1)
            if citation_id in citations:
                citation = citations[citation_id]
                simple_id = paper_mapping[citation_id]
                return f"""
                <span class="citation" onclick="toggleCitation('{citation_id}')">
                    [{simple_id}]
                    <div class="citation-popup" id="popup-{citation_id}">
                        <strong>{citation['title']}</strong><br>
                        <small>Source: {os.path.splitext(citation['filename'])[0]}</small>
                        <div class="quote">{citation['quote']}</div>
                    </div>
                </span>
                """
            return match.group(0)
        
        return re.sub(citation_pattern, replace_citation, text)

    def format_references(unique_papers):
        """Format simplified references section"""
        refs = []
        for paper_id, paper_info in sorted(unique_papers.items(), key=lambda x: int(x[0])):
            filename = os.path.splitext(paper_info['filename'])[0]
            refs.append(f"<p>[{paper_id}] {paper_info['title']} ({filename})</p>")
        return "\n".join(refs)

    # Process the review text
    paper_mapping, unique_papers = get_simplified_references(citations)
    processed_text = process_markdown_headers(review)
    processed_text = process_citations(processed_text, paper_mapping)
    
    # Add references section
    references_section = f"""
    <h2>References</h2>
    {format_references(unique_papers)}
    """
    
    # Combine all components
    html = f"""
    {styles}
    {script}
    <div class="review-container">
        <div class="model-info">
            Generated using: {model_choice}
        </div>
        {processed_text}
        {references_section}
    </div>
    """
    
    return html