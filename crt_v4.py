from __future__ import annotations

import html
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF

from google import genai
from streamlit_js_eval import streamlit_js_eval

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & ENVIRONMENT
# -----------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
PASSWORD = os.getenv("PASSWORD")

# Early validation of API key
if not API_KEY:
    st.error("GOOGLE_API_KEY not set")
    st.stop()

# Basic app configuration
APP_TITLE = "AI Revision Assistant"
CHAPTER_CHAR_LIMIT = 350_000  # adjust to your model

# Conversational style options
CONVERSATIONAL_STYLES = {
    "Standard": "You are a helpful and expert revision tutor.",
    "Socratic": "You are a Socratic tutor who guides learning through thoughtful questions. Instead of giving direct answers, ask probing questions that help the student discover the answer themselves. Build on their responses with follow-up questions.",
    "Encouraging": "You are an enthusiastic and encouraging revision tutor. Use positive reinforcement, celebrate understanding, and provide gentle guidance when students struggle. Make learning feel rewarding and confidence-building.",
    "Detailed": "You are a thorough and comprehensive revision tutor. Provide detailed explanations with examples, break down complex concepts into manageable parts, and ensure complete understanding before moving on.",
    "Concise": "You are a direct and concise revision tutor. Give clear, brief explanations and get straight to the point. Focus on key concepts without unnecessary elaboration."
}

# Document reliance levels
DOCUMENT_RELIANCE_LEVELS = {
    1: "You may use your general knowledge to supplement the provided text, but always indicate when you're drawing from external knowledge versus the chapter content.",
    2: "Primarily use the provided text, but you may occasionally reference relevant general knowledge when it directly supports understanding of the chapter content.",
    3: "Focus mainly on the provided text content. Only use general knowledge sparingly and when it's essential for clarification.",
    4: "Your responses should be heavily based on the provided text. Use general knowledge very minimally and only when absolutely necessary.",
    5: "Your knowledge is STRICTLY and ONLY limited to the content provided below. Do not use any external knowledge. **If the user's question cannot be answered using the provided text, you MUST state that clearly.**"
}

SYSTEM_PROMPT_TEMPLATE = """
{style_instruction} Your sole purpose is to help a student understand and revise the content of a specific textbook chapter.

{reliance_instruction}

Here is the content of the chapter you are tutoring on:

--- START OF CHAPTER CONTENT ---

**Chapter Title:** {chapter_title}

{chapter_text}

--- END OF CHAPTER CONTENT ---

Now, begin the conversation by greeting the student and inviting them to ask a question about this chapter.
""".strip()

# -----------------------------------------------------------------------------
# 2. CORE CONVERSATION TREE CLASSES (Reused from cctc_v36.py)
# -----------------------------------------------------------------------------

@dataclass
class MsgNode:
    """A single message node in the conversation tree."""
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    parent_id: Optional[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ConvTree:
    """Tree of MsgNode objects with helpers for conversation branching and traversal."""
    def __init__(self, root_content: str = "ROOT"):
        root_id = "root"
        self.nodes: Dict[str, MsgNode] = {
            root_id: MsgNode(id=root_id, role="system", content=root_content, parent_id=None)
        }
        self.children: Dict[str, List[str]] = {root_id: []}
        self.current_leaf_id: str = root_id

    def add_node(self, parent_id: str, role: str, content: str) -> str:
        node_id = str(uuid.uuid4())
        node = MsgNode(id=node_id, role=role, content=content, parent_id=parent_id)
        self.nodes[node_id] = node
        self.children.setdefault(node_id, [])
        self.children.setdefault(parent_id, []).append(node_id)
        return node_id

    def path_to_leaf(self, leaf_id: Optional[str] = None) -> List[MsgNode]:
        if leaf_id is None:
            leaf_id = self.current_leaf_id
        path: List[MsgNode] = []
        cursor = leaf_id
        while cursor is not None:
            path.append(self.nodes[cursor])
            cursor = self.nodes[cursor].parent_id
        return list(reversed(path))

    def siblings(self, node_id: str) -> List[str]:
        """Return list of sibling node IDs (including self)."""
        parent_id = self.nodes[node_id].parent_id
        if parent_id is None:
            return []
        return self.children.get(parent_id, [])

    def sibling_index(self, node_id: str) -> int:
        """Return the index of this node among its siblings."""
        sibs = self.siblings(node_id)
        return sibs.index(node_id) if node_id in sibs else -1

    def deepest_descendant(self, node_id: str) -> str:
        """Find the deepest descendant (following first child path)."""
        cursor = node_id
        while self.children.get(cursor):
            cursor = self.children[cursor][0]
        return cursor

    def select_sibling(self, node_id: str, direction: int) -> None:
        """Navigate to a sibling and update current_leaf_id."""
        sibs = self.siblings(node_id)
        if len(sibs) <= 1:
            return
        idx = (self.sibling_index(node_id) + direction) % len(sibs)
        new_id = sibs[idx]
        self.current_leaf_id = self.deepest_descendant(new_id)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# --- PDF and Text Processing ---
def extract_text_from_pdf(file_bytes: bytes, pages: List[int]) -> str:
    """Extracts text from specified pages of a PDF file."""
    text = ""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page_num in pages:
                # Page numbers are 1-based for users, but 0-based for PyMuPDF
                if 1 <= page_num <= len(doc):
                    page = doc.load_page(page_num - 1)
                    text += page.get_text() + "\n\n"
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""
    return text

def parse_chapter_definitions(definitions: str) -> List[Dict]:
    """Parses chapter definitions from a text area."""
    chapters = []
    for line in definitions.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Regex to capture "Title, Start-End"
        match = re.match(r'^(.*?),\s*(\d+)\s*-\s*(\d+)$', line)
        if match:
            title, start, end = match.groups()
            start_page, end_page = int(start), int(end)
            if start_page > end_page:
                st.warning(f"Warning: In '{line}', start page {start_page} is after end page {end_page}. Swapping them.")
                start_page, end_page = end_page, start_page
            chapters.append({
                "title": title.strip(),
                "pages": list(range(start_page, end_page + 1))
            })
        else:
            st.warning(f"Could not parse line: '{line}'. Please use the format: Chapter Title, StartPage-EndPage")
    return chapters

# --- Gemini API Interaction ---
def to_part(text: str) -> dict:
    """Return a Gem-compatible PartDict for a text string."""
    return {"text": text}

def to_content(role: str, text: str) -> dict:
    """Return a Gem-compatible Content dict (role=user|model)."""
    return {"role": role, "parts": [to_part(text)]}

def get_tutor_response(conv_tree, system_prompt, temperature=0.2):
    # flatten tree path, skip root
    msgs = conv_tree.path_to_leaf()[1:]
    assert msgs, "No messages to respond to."

    # turn nodes into genai 'Content' dicts
    def _content_from_node(node):
        return {
            "role": "user" if node.role == "user" else "model",
            "parts": [{"text": node.content}],
        }

    # history = everything except the last message; prompt = last user message
    history = [_content_from_node(n) for n in msgs[:-1]]
    last_user_text = msgs[-1].content

    from google import genai as _genai_new
    client = _genai_new.Client(api_key=API_KEY)
    chat = client.chats.create(
        model="gemini-2.5-flash",
        history=history,  # already in correct role/parts form
        config=_genai_new.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        ),
    )
    resp = chat.send_message(last_user_text)
    return resp.text

# --- UI and State Management ---
def reset_chat_session():
    """Resets the conversation tree for a new chat."""
    st.session_state.conv_tree = ConvTree()
    st.session_state.pending_user_node_id = None
    st.session_state.editing_msg_id = None
    st.session_state.editing_content = ""

def markdown_to_html(text: str) -> str:
    """Converts basic markdown formatting to HTML for bubble display."""
    if not text:
        return ""
    text = html.escape(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.*?)_', r'<em>\1</em>', text)
    text = text.replace('\n', '<br>')
    return text

def render_msg(node: MsgNode):
    """Renders a single message bubble with versioning and editing controls."""
    
    # 1. Define bubble styles based on role (matching cctc_v36_copy.py)
    if node.role == "assistant":
        role_label = "Revision Assistant"
        align = "flex-start"
        bubble_color = "#1b222a"  # Dark blue-gray for assistant
    else:  # User
        role_label = "Trainee (You)"
        align = "flex-end"
        bubble_color = "#0e2a47"  # Darker blue for user

    text_color = "white"
    
    # 2. Handle "edit mode" for the user's message
    if node.role == "user" and st.session_state.editing_msg_id == node.id:
        new_text = st.text_area(
            "Edit your message:",
            value=st.session_state.editing_content or node.content,
            key=f"textarea_{node.id}",
        )
        col_l, col_cancel, col_send, col_r = st.columns([6, 1, 1, 6], gap="small")
        
        with col_cancel:
            if st.button("Cancel", key=f"cancel_edit_{node.id}"):
                st.session_state.editing_msg_id = None
                st.session_state.editing_content = ""
                st.rerun()
        
        with col_send:
            if st.button("Send", key=f"send_edit_{node.id}"):
                # This is the branching logic!
                parent_id = node.parent_id
                new_user_id = st.session_state.conv_tree.add_node(parent_id, "user", new_text)
                st.session_state.conv_tree.current_leaf_id = new_user_id
                st.session_state.pending_user_node_id = new_user_id
                st.session_state.editing_msg_id = None
                st.session_state.editing_content = ""
                st.rerun()
        return  # Stop further rendering for this node
    
    # 3. Render the message bubble with potential controls
    sibs = st.session_state.conv_tree.siblings(node.id)
    has_versions = len(sibs) > 1
    
    # 3a. For user messages with version controls
    if node.role == "user" and has_versions:
        idx = st.session_state.conv_tree.sibling_index(node.id) + 1
        total = len(sibs)
        
        # Desktop layout with proper column spacing
        col_left, col_center, col_right, col_edit, col_bubble = st.columns(
            [1.5, 1.5, 2, 6, 40], gap="small"
        )
        
        # ◀ Previous version button
        with col_left:
            st.markdown(
                f"<div style='display:flex; align-items:center; margin-top:25px;'>",
                unsafe_allow_html=True)
            if st.button("◀", key=f"left_{node.id}"):
                st.session_state.conv_tree.select_sibling(node.id, -1)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Version indicator (e.g., "1/3")
        with col_center:
            st.markdown(
                f"""
                <div style='display:flex; align-items:center; margin-top:32px;
                            transform:translate(10px, 0);'>{idx}/{total}</div>
                """,
                unsafe_allow_html=True)
        
        # ▶ Next version button
        with col_right:
            st.markdown(
                f"<div style='display:flex; align-items:center; margin-top:25px; transform:translate(-80px, 0);'>",
                unsafe_allow_html=True)
            if st.button("▶", key=f"right_{node.id}"):
                st.session_state.conv_tree.select_sibling(node.id, +1)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Edit Message button
        with col_edit:
            st.markdown(
                f"<div style='display:flex; align-items:center; margin-top:25px; transform:translate(-80px, 0);'>",
                unsafe_allow_html=True)
            if st.button("Edit Message", key=f"edit_{node.id}"):
                st.session_state.editing_msg_id = node.id
                st.session_state.editing_content = node.content
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Message bubble
        with col_bubble:
            st.markdown(
                f"""
                <div style='display:flex; justify-content:{align}; margin:8px 0;'>
                  <div style='background-color:{bubble_color}; color:{text_color};
                              padding:12px 16px; border-radius:18px;
                              max-width:75%; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                              font-size:16px; line-height:1.5;'>
                    <strong>{role_label}:</strong><br>{markdown_to_html(node.content)}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return
    
    # 3b. For user messages without versions (single edit button)
    elif node.role == "user":
        edit_col, bubble_col = st.columns([1, 9], gap="small")
        
        with edit_col:
            st.markdown(
                f"<div style='display:flex; align-items:center; margin-top:25px;'>",
                unsafe_allow_html=True)
            if st.button("Edit Message", key=f"edit_{node.id}"):
                st.session_state.editing_msg_id = node.id
                st.session_state.editing_content = node.content
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        with bubble_col:
            st.markdown(
                f"""
                <div style='display:flex; justify-content:{align}; margin:8px 0;'>
                  <div style='background-color:{bubble_color}; color:{text_color};
                              padding:12px 16px; border-radius:18px;
                              max-width:75%; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                              font-size:16px; line-height:1.5;'>
                    <strong>{role_label}:</strong><br>{markdown_to_html(node.content)}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return
    
    # 3c. For assistant messages (no controls needed)
    else:
        st.markdown(
            f"""
            <div style='display:flex; justify-content:{align}; margin:8px 0;'>
              <div style='background-color:{bubble_color}; color:{text_color};
                          padding:12px 16px; border-radius:18px;
                          max-width:75%; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                          font-size:16px; line-height:1.5;'>
                <strong>{role_label}:</strong><br>{markdown_to_html(node.content)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# 4. STREAMLIT UI
# -----------------------------------------------------------------------------

# --- Page and Style Configuration ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown("""
<style>
    div.block-container {padding-top: 1rem !important;}
    .stTabs {margin-top: 0rem;}
</style>
""", unsafe_allow_html=True)

# --- Password Gate (Reused) ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Welcome to the AI Revision Assistant")
        st.markdown(
            """
            This application allows you to upload a textbook (in PDF format) and chat with an AI tutor to help you revise specific chapters.

            **How it works:**
            1.  **Login:** Enter the password to access the app.
            2.  **Upload:** Provide a PDF document.
            3.  **Define Chapters:** Specify the title and page range for each chapter you want to study.
            4.  **Chat:** Select a chapter and start asking questions! The AI will only use the content from that chapter to answer you.

            ***Privacy Note:*** *Your documents and chat history are not saved. Everything is deleted when you close or refresh the browser tab.*
            """,
            unsafe_allow_html=True,
        )

        with st.form(key="password_form"):
            entered_password = st.text_input("Password:", type="password")
            if st.form_submit_button("Login"):
                if entered_password == PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        st.stop()

check_password()
if not API_KEY or not PASSWORD:
    st.error("API_KEY or PASSWORD environment variables not set. The application cannot run.")
    st.stop()

client = genai.Client(api_key=API_KEY)


# --- Session State Initialization ---
if "conv_tree" not in st.session_state: st.session_state.conv_tree = ConvTree()
if "pending_user_node_id" not in st.session_state: st.session_state.pending_user_node_id = None
if "uploaded_file_info" not in st.session_state: st.session_state.uploaded_file_info = None
if "chapters" not in st.session_state: st.session_state.chapters = []
if "active_chapter_index" not in st.session_state: st.session_state.active_chapter_index = 0
if "editing_msg_id" not in st.session_state: st.session_state.editing_msg_id = None
if "editing_content" not in st.session_state: st.session_state.editing_content = ""
if "conversational_style" not in st.session_state: st.session_state.conversational_style = "Standard"
if "document_reliance" not in st.session_state: st.session_state.document_reliance = 5

conv_tree: ConvTree = st.session_state.conv_tree

# --- Main Application Logic: Upload or Chat ---

# STATE 1: UPLOAD VIEW
if not st.session_state.chapters:
    st.header("Step 1: Upload Your Document")
    uploaded_file = st.file_uploader(
        "Upload your textbook or document (PDF only)",
        type="pdf"
    )

    if uploaded_file:
        st.header("Step 2: Define Chapters")
        st.markdown(
            "Enter each chapter on a new line using the format: `Chapter Title, StartPage-EndPage`"
        )
        st.markdown(
            "**Example:**\n"
            "Introduction to AI, 1-10\n"
            "History of Neural Networks, 11-25"
        )
        
        # Option to upload chapter definitions from a text file
        uploaded_definitions_file = st.file_uploader(
            "Upload a .txt file with chapter definitions or manually enter them below",
            type="txt",
            help="Upload a text file where each line contains: Chapter Title, StartPage-EndPage"
        )
        
        with st.form("chapter_form"):
            # Pre-populate with uploaded file content if available
            default_text = ""
            if uploaded_definitions_file is not None:
                try:
                    # Read the uploaded file content
                    default_text = uploaded_definitions_file.read().decode("utf-8")
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")
            
            chapter_definitions_text = st.text_area(
                "Chapter Definitions",
                value=default_text,
                height=150,
                placeholder="For example:\n\nChapter 1: The Beginning, 1-20\nChapter 2: The Middle, 21-55"
            )
            submitted = st.form_submit_button("Process Document and Chapters")

        if submitted and chapter_definitions_text:
            with st.spinner("Processing document... This may take a moment."):
                parsed_chapters = parse_chapter_definitions(chapter_definitions_text)
                if parsed_chapters:
                    file_bytes = uploaded_file.getvalue()
                    # Extract text for each chapter and store it
                    for chapter in parsed_chapters:
                        chapter_text = extract_text_from_pdf(file_bytes, chapter['pages'])
                        
                        # Add chapter diagnostics
                        with st.expander(f"Chapter diagnostics: {chapter['title']}", expanded=False):
                            total_pages = len(chapter['pages'])
                            ch_chars = len(chapter_text or "")
                            st.write({
                                "total_pdf_pages": total_pages, 
                                "selected_range": f"{chapter['pages'][0]}-{chapter['pages'][-1]}", 
                                "chapter_characters": ch_chars
                            })
                            st.text_area("Preview (first ~1200 chars)", (chapter_text or "")[:1200], height=200)
                            if not chapter_text.strip():
                                st.warning("No text extracted. The PDF may be scanned images. Try an OCR'd PDF.")
                            st.download_button(
                                "Download extracted chapter text", 
                                chapter_text.encode("utf-8"), 
                                file_name=f"{chapter['title']}.txt"
                            )
                        
                        # Guard against over-length inputs
                        if len(chapter_text) > CHAPTER_CHAR_LIMIT:
                            st.warning(f"Chapter '{chapter['title']}' is {len(chapter_text):,} chars; trimming to {CHAPTER_CHAR_LIMIT:,}.")
                            chapter_text = chapter_text[:CHAPTER_CHAR_LIMIT]
                        
                        chapter['text'] = chapter_text
                    
                    st.session_state.chapters = parsed_chapters
                    st.session_state.uploaded_file_info = {
                        "name": uploaded_file.name,
                        "size": uploaded_file.size
                    }
                    st.session_state.active_chapter_index = 0
                    reset_chat_session() # Initialize the chat for the first chapter
                    st.rerun()
                else:
                    st.error("No valid chapter definitions found. Please check the format.")

# STATE 2: CHAT VIEW
else:
    # --- Sidebar for Chapter Selection and Settings ---
    with st.sidebar:

        st.header("Session Setup")

        chapter_titles = [ch['title'] for ch in st.session_state.chapters]
        
        # This widget's state is controlled by st.session_state.active_chapter_index
        selected_chapter_index = st.selectbox(
            "Select Chapter to Focus On",
            options=range(len(chapter_titles)),
            format_func=lambda i: chapter_titles[i],
            key="active_chapter_index",
            on_change=reset_chat_session # Reset chat when chapter changes
        )
        
        st.markdown("---")

        # Conversation Style Selection
        st.subheader("Conversation Style")
        selected_style = st.selectbox(
            "How the AI revision assistant interacts:",
            options=list(CONVERSATIONAL_STYLES.keys()),
            key="conversational_style",
            help="Different styles will change how the AI responds to your questions"
        )
        
        st.markdown("---")
        
        # Document Reliance Level
        st.subheader("Document Reliance")
        reliance_level = st.slider(
            "How strictly the AI sticks to the document:",
            min_value=1,
            max_value=5,
            value=st.session_state.document_reliance,
            key="document_reliance",
            help="1 = Can use general knowledge freely | 5 = Restricted to document content only"
        )
        
        # Show current reliance description
        reliance_labels = {
            1: "Very Low - General knowledge encouraged",
            2: "Low - Primarily document, some general knowledge",
            3: "Medium - Mainly document, minimal general knowledge", 
            4: "High - Heavy document focus, very limited general knowledge",
            5: "Very High - Document content only"
        }
        
        st.markdown("---")

        if st.session_state.uploaded_file_info:
            st.info(f"**Document:**\n{st.session_state.uploaded_file_info['name']}")

        if st.button("Start with a New Document"):
            # Clear all session state to return to upload view
            for key in list(st.session_state.keys()):
                if key != 'authenticated': # Keep user logged in
                    del st.session_state[key]
            st.rerun()

    # --- Main Chat Interface ---
    active_chapter = st.session_state.chapters[st.session_state.active_chapter_index]
    st.title(f"Focusing On: {active_chapter['title']}")
    
    # Render chat history
    chat_container = st.container()
    with chat_container:
        for node in conv_tree.path_to_leaf()[1:]:  # Skip root
            render_msg(node)

    # Handle pending AI response
    if st.session_state.pending_user_node_id:
        with st.spinner("Tutor is thinking..."):
            try:
                # Get the selected conversational style and document reliance level
                style_instruction = CONVERSATIONAL_STYLES[st.session_state.conversational_style]
                reliance_instruction = DOCUMENT_RELIANCE_LEVELS[st.session_state.document_reliance]
                system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    style_instruction=style_instruction,
                    reliance_instruction=reliance_instruction,
                    chapter_title=active_chapter['title'],
                    chapter_text=active_chapter['text']
                )
                ai_reply = get_tutor_response(conv_tree, system_prompt)
                
                new_assist_id = conv_tree.add_node(st.session_state.pending_user_node_id, "assistant", ai_reply)
                conv_tree.current_leaf_id = new_assist_id
                st.session_state.pending_user_node_id = None
                st.rerun()
            except Exception as e:
                st.error("Sorry, I encountered an error. Please try again.")
                st.session_state.pending_user_node_id = None
                st.stop()

    # Chat input at the bottom
    if user_text := st.chat_input("Ask a question about this chapter..."):
        new_user_id = conv_tree.add_node(conv_tree.current_leaf_id, "user", user_text)
        conv_tree.current_leaf_id = new_user_id
        st.session_state.pending_user_node_id = new_user_id
        st.rerun()