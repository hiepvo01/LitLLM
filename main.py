import streamlit as st
import os
import base64
from dotenv import load_dotenv
from rag_system import EnhancedResearchRAG
from utils import clean_topic_name
import streamlit.components.v1 as components
from components.review_display import get_review_component

# Load environment variables
load_dotenv()

def initialize_system():
    """Initialize the RAG system"""
    if 'rag_system' not in st.session_state:
        try:
            st.session_state.rag_system = EnhancedResearchRAG()
        except ValueError as e:
            st.error(f"Error initializing system: {e}")
            st.stop()
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    
    if 'citations' not in st.session_state:
        st.session_state.citations = {}

def display_existing_papers(topic: str):
    """Display existing papers with view, download, and delete options"""
    if topic in st.session_state.rag_system.papers_by_topic:
        st.markdown("### Uploaded Papers")
        papers = st.session_state.rag_system.papers_by_topic[topic]
        
        if not papers:
            st.info("No papers uploaded yet.")
            return
            
        # Initialize delete confirmation state if not exists
        if 'delete_confirmation' not in st.session_state:
            st.session_state.delete_confirmation = {}
        
        # Initialize view PDF state if not exists
        if 'view_pdf' not in st.session_state:
            st.session_state.view_pdf = {}
            
        for paper_id, paper_info in papers.items():
            title = paper_info['metadata'].get('title', paper_info['original_name'])
            
            # Create expandable section for each paper
            with st.expander(f"**[{paper_id}]** {title}"):
                # Paper metadata
                st.markdown(f"**Authors:** {paper_info['metadata'].get('authors', 'Unknown')}")
                st.markdown(f"**Year:** {paper_info['metadata'].get('year', 'Unknown')}")
                st.markdown(f"**File:** {paper_info['original_name']}")
                
                # Action buttons row
                col1, col2, col3 = st.columns([1, 1, 1])
                
                # View PDF button and display
                with col1:
                    if st.button("View PDF", key=f"view_{paper_id}"):
                        # Toggle PDF viewer
                        current_state = st.session_state.view_pdf.get(paper_id, False)
                        st.session_state.view_pdf[paper_id] = not current_state
                
                # Download button
                with col2:
                    # Read PDF file for download
                    with open(paper_info['path'], 'rb') as pdf_file:
                        pdf_bytes = pdf_file.read()
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=paper_info['original_name'],
                        mime="application/pdf",
                        key=f"download_{paper_id}"
                    )
                
                # Delete button with confirmation
                with col3:
                    if f"confirm_{paper_id}" not in st.session_state.delete_confirmation:
                        if st.button("Delete", key=f"delete_{paper_id}", type="secondary"):
                            st.session_state.delete_confirmation[f"confirm_{paper_id}"] = True
                            st.rerun()
                    else:
                        st.button(
                            "Confirm Delete",
                            key=f"confirm_{paper_id}",
                            type="primary",
                            on_click=lambda pid=paper_id, pinfo=paper_info, t=title: delete_paper(pid, pinfo, t, topic)
                        )
                        st.button(
                            "Cancel",
                            key=f"cancel_{paper_id}",
                            on_click=lambda pid=paper_id: cancel_delete(pid)
                        )
                
                # Display PDF viewer if enabled
                if st.session_state.view_pdf.get(paper_id, False):
                    # Read and display PDF file
                    with open(paper_info['path'], 'rb') as pdf_file:
                        pdf_bytes = pdf_file.read()
                        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)

def delete_paper(paper_id: str, paper_info: dict, title: str, topic: str):
    """Handle paper deletion"""
    st.session_state.rag_system.delete_paper(topic, paper_id)
    if paper_info['original_name'] in st.session_state.processed_files:
        st.session_state.processed_files.remove(paper_info['original_name'])
    del st.session_state.delete_confirmation[f"confirm_{paper_id}"]
    st.success(f"Paper '{title}' deleted successfully!")
    st.rerun()

def cancel_delete(paper_id: str):
    """Handle delete cancellation"""
    del st.session_state.delete_confirmation[f"confirm_{paper_id}"]
    st.rerun()

def process_uploaded_files(files, topic: str):
    """Process uploaded PDF files"""
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, file in enumerate(files):
        if file.name not in st.session_state.processed_files:
            progress_text.text(f"Processing: {file.name}")
            progress_value = (i + 1) / len(files)
            progress_bar.progress(progress_value)
            
            st.session_state.rag_system.process_pdf(file, topic)
            st.session_state.processed_files.add(file.name)
            
            progress_text.text(f"Processed: {file.name}")
        else:
            st.info(f"Already processed: {file.name}")
    
    progress_text.text("All files processed!")
    progress_bar.progress(1.0)
    st.success("Processing complete!")

def document_summary_tab():
    """Handle document summary and chat functionality"""
    st.markdown("### Document Summary and Chat")
    
    # Initialize states
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_summary' not in st.session_state:
        st.session_state.current_summary = None
    if 'chat_citations' not in st.session_state:
        st.session_state.chat_citations = {}
    if 'last_user_input' not in st.session_state:
        st.session_state.last_user_input = None
    
    # Get all available papers across all topics
    all_papers = {}
    for topic, papers in st.session_state.rag_system.papers_by_topic.items():
        for paper_id, paper_info in papers.items():
            key = f"{topic} - {paper_info['metadata']['title']}"
            all_papers[key] = (topic, paper_id, paper_info)
    
    if not all_papers:
        st.warning("Please upload some papers in the Literature Review tab first.")
        return
    
    # Document selection
    selected_doc = st.selectbox(
        "Select a document to summarize:",
        options=list(all_papers.keys()),
        key="doc_selector"
    )
    
    if selected_doc:
        topic, paper_id, paper_info = all_papers[selected_doc]
        
        # Generate summary button
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Generating document summary..."):
                try:
                    summary, summary_citations = st.session_state.rag_system.generate_document_summary(
                        topic, paper_id
                    )
                    st.session_state.current_summary = summary
                    st.session_state.chat_citations = summary_citations
                    # Reset chat history when new summary is generated
                    st.session_state.chat_history = []
                    st.session_state.last_user_input = None
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    return
        
        # Display summary if available
        if st.session_state.current_summary:
            st.markdown("### Document Summary")
            
            # Display summary with citations
            summary_html = get_review_component(
                st.session_state.current_summary,
                st.session_state.chat_citations,
                "Document Summary"
            )
            components.html(summary_html, height=400, scrolling=True)
            
            # Chat interface
            st.markdown("### Chat about the Document")
            
            # Display chat history with citations
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history:
                    role = "🤔 You" if msg["role"] == "user" else "🤖 Assistant"
                    if msg["role"] == "user":
                        st.markdown(f"**{role}:** {msg['content']}")
                    else:
                        # Display assistant response with citations
                        response_html = get_review_component(
                            msg['content'],
                            msg.get('citations', {}),
                            "AI Response"
                        )
                        st.markdown(f"**{role}:**")
                        components.html(response_html, height=300, scrolling=True)
            
            # Chat input
            user_input = st.text_input(
                "Ask a question about the document:",
                key="chat_input"
            )
            
            # Only process new input if it's different from the last one
            if user_input and user_input != st.session_state.last_user_input:
                st.session_state.last_user_input = user_input
                
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Get AI response with citations
                with st.spinner("Thinking..."):
                    try:
                        response, citations = st.session_state.rag_system.chat_about_document(
                            topic, paper_id, user_input,
                            st.session_state.chat_history
                        )
                        
                        # Add AI response to chat history with citations
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "citations": citations
                        })
                        
                        # Clear the input field by rerunning
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.session_state.last_user_input = None  # Reset on error

def main():
    st.set_page_config(layout="wide", page_title="Research Paper Analysis System")
    
    st.title("Research Paper Analysis System")
    
    # Initialize system
    initialize_system()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Literature Review", "Document Summary"])
    
    with tab1:
        st.markdown("""
        Generate technical literature reviews from your research papers with AI assistance.
        Upload PDFs of research papers and get an organized review with citations.
        """)
        st.markdown("---")
        
        # Create two columns for the main layout
        left_col, right_col = st.columns([1, 2])

        with left_col:
            st.markdown("### Topic Selection")
            
            # Topic selection
            if st.session_state.rag_system.topics:
                topic_options = ["Create New Topic"] + sorted(list(st.session_state.rag_system.topics))
                selected_option = st.selectbox(
                    "Select existing topic or create new:",
                    options=topic_options,
                    key="topic_selector"
                )
                
                if selected_option == "Create New Topic":
                    topic = st.text_input(
                        "Enter topic name:",
                        key="new_topic"
                    )
                else:
                    topic = selected_option
            else:
                topic = st.text_input(
                    "Enter topic name:",
                    key="first_topic"
                )

            if topic and topic != "Create New Topic":
                st.markdown("### Upload Papers")
                uploaded_files = st.file_uploader(
                    "Upload PDF files",
                    type="pdf",
                    accept_multiple_files=True,
                    key="pdf_uploader"
                )
                
                if uploaded_files:
                    with st.spinner("Processing uploaded files..."):
                        process_uploaded_files(uploaded_files, topic)
                
                display_existing_papers(topic)

        with right_col:
            if topic and topic != "Create New Topic":
                st.markdown("### Literature Review Generation")
                
                # Model selection
                model_choice = st.selectbox(
                    "Select Language Model",
                    ["GPT-4 Turbo", "Gemini 1.5 Pro"],
                    key="model_selector"
                )
                
                # Add info about each model
                if model_choice == "GPT-4 Turbo":
                    st.info("OpenAI's most capable model, best for complex analysis")
                else:  # Gemini 1.5 Pro
                    st.info("Google's advanced model, good for technical content")
                
                st.markdown("""
                The review will:
                - Analyze and compare methodologies
                - Evaluate results and performance
                - Identify technical challenges
                - Suggest future directions
                """)
                
                if st.button("Generate Literature Review", type="primary"):
                    if not st.session_state.rag_system.papers_by_topic.get(topic):
                        st.error("Please upload some papers first!")
                        st.stop()
                    
                    with st.spinner("Generating comprehensive review..."):
                        # Map selection to model identifier
                        model_id = "gpt-4" if model_choice == "GPT-4 Turbo" else "gemini-1.5"
                        
                        try:
                            review, citations = st.session_state.rag_system.generate_review(
                                topic,
                                model_id
                            )
                            
                            # Store citations in session state
                            st.session_state.citations = citations
                            
                            # Display interactive review
                            st.markdown("### Generated Literature Review")
                            review_html = get_review_component(review, citations, model_choice)
                            components.html(review_html, height=800, scrolling=True)
                            
                            # Download button
                            st.download_button(
                                "Download Review (Markdown)",
                                review,
                                file_name=f"literature_review_{topic}.md",
                                mime="text/markdown"
                            )
                        except Exception as e:
                            st.error(f"Error generating review: {str(e)}")
    
    with tab2:
        document_summary_tab()

if __name__ == "__main__":
    main()