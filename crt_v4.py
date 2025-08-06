from __future__ import annotations

import html
import os
import re
import uuid
import json
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from io import BytesIO, StringIO

import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from google import genai
from google.genai import types as gx
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
CHAPTER_CHAR_LIMIT = 700_000  # adjust to your model

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
# 2. FLASHCARD SCHEMA
# -----------------------------------------------------------------------------

class Flashcard(BaseModel):
    q: str = Field(description="Front/question. Avoid prefixes like 'Q:'.")
    a: str = Field(description="Back/answer. Concise and unambiguous.")
    type: Literal["basic","cloze","definition","list","true_false","multiple_choice","compare_contrast"] = "basic"
    difficulty: Literal["easy","medium","hard"] = "medium"
    tags: List[str] = []
    source: str = Field(description="Specific section heading, subsection title, or page reference where this information appears in the chapter (e.g., 'Section 2.3: Neural Networks', 'Page 45', 'Introduction paragraph')")

# -----------------------------------------------------------------------------
# 3. MCQ SCHEMA & UTILITIES
# -----------------------------------------------------------------------------

class MCQ(BaseModel):
    stem: str = Field(description="Question stem. No numbering or 'Q:'.")
    options: List[str] = Field(description="Answer options, at least 4 where possible.")
    correct: List[int] = Field(description="Zero-based indices of correct options.")
    explanation: str = ""
    difficulty: Literal["easy","medium","hard"] = "medium"
    tags: List[str] = []
    source: str = ""    # section heading, page, or paragraph reference
    id: str = ""        # filled by app
    hash: str = ""      # normalized stem hash for dedupe

def _norm_stem(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _stem_hash(s: str) -> str:
    import hashlib
    return hashlib.sha1(_norm_stem(s).encode()).hexdigest()[:12]

def _mcq_dedupe_hashes(existing: List[dict]) -> set[str]:
    return {_stem_hash(x.get("stem","")) for x in existing if x.get("stem")}

def _assign_ids_and_hashes(items: List[dict]) -> List[dict]:
    import hashlib
    out = []
    for itm in items:
        itm["hash"] = _stem_hash(itm["stem"])
        if not itm.get("id"):
            itm["id"] = hashlib.md5((itm["hash"] + "|".join(itm["options"])).encode()).hexdigest()[:12]
        out.append(itm)
    return out

# -----------------------------------------------------------------------------
# 4. CORE CONVERSATION TREE CLASSES (Reused from cctc_v36.py)
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
# 5. HELPER FUNCTIONS
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

# --- Flashcard Generation Functions ---
def _active_cards_and_hashes() -> tuple[list[dict], set[str]]:
    ch_idx = st.session_state.active_chapter_index
    cards = st.session_state.flashcards_by_chapter.get(ch_idx, [])
    # plus any imported cards (not yet merged)
    imported = st.session_state.flash_import_buffer or []
    all_cards = cards + imported
    def _h(q: str) -> str:
        return re.sub(r"\s+", " ", q.strip().lower())
    hashes = {_h(c.get("q","")) for c in all_cards if c.get("q")}
    return all_cards, hashes

def _chapter_excerpt(text: str, limit: int=15000) -> str:
    # Use your CHAPTER_CHAR_LIMIT logic; show start + key headings if you have them
    return text[:limit]

def _build_prompt(chapter_title: str, chapter_text: str, settings: dict, exclude_hashes: set[str]) -> str:
    n = settings["n_cards"]
    types_s = settings["types"]  # Now a single string instead of list
    diff = settings["difficulty"]
    
    # Build the base prompt with detailed instructions for each card type
    type_instructions = {
        "basic": "Basic Q&A format with straightforward questions and answers.",
        "cloze": "Fill-in-the-blank format where key words or phrases are replaced with [...]. Example: 'The [...] is responsible for executive functions' → 'The prefrontal cortex is responsible for executive functions'",
        "definition": "Ask for the definition of key terms or concepts. Format as 'Define: [term]' and provide the definition.",
        "list": "Questions that require listing multiple items, steps, or components. Example: 'List the three types of memory' → '1. Sensory memory, 2. Short-term memory, 3. Long-term memory'",
        "true_false": "True/False statements about concepts. Format as 'True or False: [statement]' and provide the correct answer with brief explanation.",
        "multiple_choice": "Multiple choice questions with 4 options (A, B, C, D). Format question with options, answer should indicate correct letter and brief explanation.",
        "compare_contrast": "Questions asking to compare or contrast two or more concepts. Example: 'Compare classical and operant conditioning' with detailed comparison in answer."
    }
    
    base_prompt = f"""You are generating {n} flashcards for chapter "{chapter_title}". card type: {types_s}. Difficulty: {diff}.

CARD TYPE INSTRUCTIONS for {types_s}:
{type_instructions.get(types_s, "Standard format appropriate for the selected type.")}

Guidelines:
- Avoid duplicates and trivial rephrasings
- Only create cards that can be answered from this chapter
- Prefer high-yield facts and key concepts
- For psychology content, focus on theories, researchers, definitions, and applications
- Ensure answers are concise but complete

IMPORTANT: For each flashcard, you MUST include a 'source' field indicating the specific section, heading, or page reference where the information comes from."""
    
    # Add custom instructions if provided
    custom_instructions = st.session_state.get("custom_instructions", "").strip()
    if custom_instructions:
        base_prompt += f"\n\nAdditional Instructions: {custom_instructions}"
    
    # Add the deduplication info and chapter content
    base_prompt += f"""

Known question hashes to avoid (normalized): {sorted(list(exclude_hashes))[:200]}

CHAPTER CONTENT:
---
{chapter_text}
---
"""
    
    return base_prompt

def _generate_flashcards_for_active_chapter():
    ch = st.session_state.chapters[st.session_state.active_chapter_index]
    s = st.session_state.flash_settings
    _, exclude_hashes = _active_cards_and_hashes()

    prompt = _build_prompt(ch["title"], _chapter_excerpt(ch["text"]), s, exclude_hashes)

    with st.spinner("Generating flashcards…"):
        try:
            # Use the global client
            from google import genai as _genai_new
            flash_client = _genai_new.Client(api_key=API_KEY)
            
            cfg = gx.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[Flashcard],
                temperature=float(s["temperature"]),
            )
            
            resp = flash_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=cfg,
            )
            
            # Prefer parsed; fall back to JSON text
            cards = []
            try:
                if resp.parsed:
                    cards = [c.model_dump() for c in resp.parsed]
                elif resp.text:
                    # Check if response might be truncated
                    text = resp.text.strip()
                    if not text.endswith(']') and not text.endswith('}'):
                        st.warning("Response appears to be truncated. Trying to fix...")
                        # Try to fix truncated JSON by adding closing brackets
                        if text.count('[') > text.count(']'):
                            text += ']'
                        if text.count('{') > text.count('}'):
                            text += '}'
                    
                    cards = json.loads(text)
                else:
                    st.error("No response received from API")
                    return
            except json.JSONDecodeError as e:
                st.warning("Received incomplete response. Attempting to recover cards...")
                # Try to extract valid cards from partial JSON
                try:
                    # Look for complete card objects in the response
                    text = resp.text or ""
                    card_pattern = r'\{[^{}]*"q":[^{}]*"a":[^{}]*\}'
                    matches = re.findall(card_pattern, text)
                    partial_cards = []
                    for match in matches:
                        try:
                            partial_cards.append(json.loads(match))
                        except:
                            continue
                    if partial_cards:
                        cards = partial_cards
                        st.info(f"Recovered {len(cards)} cards from partial response")
                    else:
                        st.error("Could not recover any cards from response. Try reducing the number of cards.")
                        return
                except Exception as e2:
                    st.error(f"Could not parse response: {e2}")
                    return
            except Exception as e:
                st.error(f"Unexpected error parsing response: {e}")
                return

            # Normalize, dedup, enforce length caps
            def _norm(c: dict) -> Optional[dict]:
                q = (c.get("q") or "").strip()
                a = (c.get("a") or "").strip()
                source = (c.get("source") or "").strip()
                if not q or not a or not source: 
                    return None
                c["q"] = q
                c["a"] = a
                c["source"] = source[:120]  # Enforce length limit
                c["type"] = c.get("type","basic")
                c["difficulty"] = c.get("difficulty","medium")
                c["tags"] = c.get("tags") or []
                return c

            normed = list(filter(None, (_norm(c) for c in cards)))

            # Deduplicate vs existing and inside batch
            _, exclude = _active_cards_and_hashes()
            seen = set()
            def _h(q): return re.sub(r"\s+"," ", q.strip().lower())
            unique = []
            for c in normed:
                h = _h(c["q"])
                if h in exclude or h in seen:
                    continue
                seen.add(h)
                unique.append(c)

            ch_idx = st.session_state.active_chapter_index
            st.session_state.flashcards_by_chapter.setdefault(ch_idx, [])
            st.session_state.flashcards_by_chapter[ch_idx].extend(unique)
            st.success(f"Added {len(unique)} new cards (discarded {len(normed)-len(unique)} duplicates).")
            st.rerun()  # Refresh the UI to immediately show the new cards
            
        except Exception as e:
            st.error(f"Error during flashcard generation: {e}")
            st.info("Try reducing the number of cards or adjusting the settings.")

def _anki_csv_bytes(cards: list[dict]) -> bytes:
    # Anki: simplest import = CSV with 2–3 fields (Front, Back, Tags). No header is safest.
    sio = StringIO()
    w = csv.writer(sio)
    for c in cards:
        tags = " ".join(c.get("tags") or [])
        w.writerow([c.get("q",""), c.get("a",""), tags])
    return sio.getvalue().encode("utf-8")

# --- MCQ Generation Functions ---
def _chapter_excerpt_for_mcq(text: str, limit: int=15000) -> str:
    return text[:limit]

def _build_mcq_prompt(chapter_title: str, chapter_text: str, settings: dict, exclude_hashes: set[str]) -> str:
    n = settings["n_qs"]
    allow_multi = settings["allow_multi"]
    diff = settings["difficulty_mix"]

    style = """Write exam-quality MCQs.
- Use clear, unambiguous stems.
- Randomize plausible distractors.
- Prefer 4 options; 5 if needed. Avoid 'All/None of the above' unless pedagogically strong.
- Include a short explanation referencing the chapter content.
- Tag with section/topic; include difficulty from {diff}.
- {multi}"""

    style = style.format(
        diff=", ".join(diff),
        multi=("Allow multiple correct answers when appropriate."
               if allow_multi else "Make exactly one correct option.")
    )

    schema_hint = """Return ONLY JSON in this exact schema:
list[MCQ] where MCQ = {
  "stem": str,
  "options": List[str],
  "correct": List[int],  # zero-based indices
  "explanation": str,
  "difficulty": "easy"|"medium"|"hard",
  "tags": List[str],
  "source": str
}"""

    prompt = f"""
You are creating MCQs strictly from the chapter.

Chapter Title: {chapter_title}

Known question hashes to avoid: {sorted(list(exclude_hashes))[:200]}

{style}

{schema_hint}

Generate {n} MCQs.

--- CHAPTER CONTENT (excerpt) ---
{chapter_text}
--- END ---
""".strip()
    return prompt

def generate_mcqs_for_active_chapter():
    ch = st.session_state.chapters[st.session_state.active_chapter_index]
    s = st.session_state.mcq_settings
    existing = st.session_state.mcq_by_chapter.get(st.session_state.active_chapter_index, [])
    exclude_hashes = _mcq_dedupe_hashes(existing)
    prompt = _build_mcq_prompt(ch["title"], _chapter_excerpt_for_mcq(ch["text"]), s, exclude_hashes)

    with st.spinner("Generating MCQs…"):
        try:
            from google import genai as _genai_new
            client = _genai_new.Client(api_key=API_KEY)

            cfg = gx.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[MCQ],
                temperature=float(s["temperature"]),
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=cfg,
            )

            # Prefer parsed; fallback to JSON text
            mcqs = []
            try:
                if hasattr(resp, "parsed") and resp.parsed:
                    mcqs = [x.model_dump() for x in resp.parsed]
                elif resp.text:
                    mcqs = json.loads(resp.text)
                else:
                    st.error("No response received from API")
                    return
            except json.JSONDecodeError:
                st.warning("Received incomplete response. Trying to extract valid MCQs…")
                text = resp.text or ""
                # VERY lenient fallback; you can tighten this later
                try:
                    mcqs = json.loads(text[text.find("["):text.rfind("]")+1])
                except Exception:
                    st.error("Could not parse MCQs from the response.")
                    return

            # Cleanup, dedupe, and save
            mcqs = [m for m in mcqs if m.get("stem") and m.get("options") and isinstance(m.get("correct"), list)]
            mcqs = _assign_ids_and_hashes(mcqs)
            new_only = [m for m in mcqs if m["hash"] not in exclude_hashes]

            st.session_state.mcq_by_chapter.setdefault(st.session_state.active_chapter_index, []).extend(new_only)
            st.success(f"Added {len(new_only)} new MCQs (filtered {len(mcqs)-len(new_only)} duplicates).")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during MCQ generation: {e}")
            st.info("Try reducing the number of questions or adjusting the settings.")

def mcqs_from_flashcards(cards: list[dict], k: int=10) -> list[dict]:
    """Generate MCQs from existing flashcards"""
    import random
    pool = [c for c in cards if c.get("type","basic") in ("basic","definition")]
    random.shuffle(pool)
    out = []
    for c in pool[:k]:
        stem = c["q"]
        correct = c["a"]
        # naive distractors = other answers; can refine with LLM later
        distractors = [x["a"] for x in pool if x is not c][:3]
        options = [correct] + distractors
        random.shuffle(options)
        corr_idx = [options.index(correct)]
        out.append({
            "stem": stem, "options": options, "correct": corr_idx,
            "explanation": correct, "difficulty": c.get("difficulty","medium"),
            "tags": c.get("tags",[]), "source": c.get("source","flashcard"),
        })
    return _assign_ids_and_hashes(out)

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
# 6. STREAMLIT UI
# -----------------------------------------------------------------------------

# --- Page and Style Configuration ---
# Decide sidebar state from session
sidebar_state = "expanded" if st.session_state.get("chapters") else "collapsed"

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state=sidebar_state,
)
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

# --- Flashcard Session State ---
if "flashcards_by_chapter" not in st.session_state:
    st.session_state.flashcards_by_chapter = {}  # {chapter_index: [card, ...]}
if "flash_import_buffer" not in st.session_state:
    st.session_state.flash_import_buffer = []  # holds imported cards before merge
if "flash_settings" not in st.session_state:
    st.session_state.flash_settings = {
        "n_cards": 5,  # Default number of cards to generate
        "types": "basic",  # single selection: basic or cloze
        "difficulty": "mixed",        # easy|medium|hard|mixed
        "temperature": 0.8
    }

# --- MCQ Session State ---
if "mcq_by_chapter" not in st.session_state:
    st.session_state.mcq_by_chapter = {}  # {chapter_index: [mcq, ...]}

if "mcq_settings" not in st.session_state:
    st.session_state.mcq_settings = {
        "n_qs": 10,
        "allow_multi": True,
        "difficulty_mix": ["easy","medium","hard"],  # chosen set
        "temperature": 0.7,
        "show_expl_immediately": True,
        "timer_seconds": 0,  # 0 = untimed
    }

if "mcq_sessions" not in st.session_state:
    st.session_state.mcq_sessions = {}  # keyed by chapter index

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
                    reset_chat_session # Initialize the chat for the first chapter
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

    # --- Main Interface with Tabs ---
    active_chapter = st.session_state.chapters[st.session_state.active_chapter_index]
    st.title(f"Focusing On: {active_chapter['title']}")
    
    tab_conv, tab_flash, tab_mcq = st.tabs(["Conversation", "Flashcards", "MCQs"])
    
    # CONVERSATION TAB
    with tab_conv:
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
    
    # FLASHCARDS TAB
    with tab_flash:

        # Generation settings in two columns at the top
        st.markdown("**Generation Settings**")
        col1, col2 = st.columns([1,1])
        s = st.session_state.flash_settings
        
        with col1:
            s["n_cards"] = st.slider("How many cards?", 1, 20, s["n_cards"])
            
            # Single card type selection with expanded options
            card_type_options = [
                "basic", 
                "cloze", 
                "definition", 
                "list", 
                "true_false", 
                "multiple_choice", 
                "compare_contrast"
            ]
            
            # Add descriptions for each card type
            card_type_descriptions = {
                "basic": "Basic Q&A format",
                "cloze": "Fill-in-the-blank [...] format",
                "definition": "Define key terms/concepts",
                "list": "List items, steps, or components",
                "true_false": "True/False statements",
                "multiple_choice": "Multiple choice questions",
                "compare_contrast": "Compare/contrast concepts"
            }
            
            current_index = 0
            if s["types"] in card_type_options:
                current_index = card_type_options.index(s["types"])
            
            s["types"] = st.selectbox(
                "Flashcard type:", 
                card_type_options,
                index=current_index,
                format_func=lambda x: card_type_descriptions[x]
            )

        with col2:
            s["temperature"] = st.slider("Temperature (How creative should the AI be?)", 0.0, 2.0, s["temperature"], 0.05)
            difficulty_options = ["mixed","easy","medium","hard"]
            s["difficulty"] = st.selectbox(
                "Difficulty:", 
                difficulty_options,
                index=difficulty_options.index(s["difficulty"]),
                format_func=lambda x: x.capitalize()
            )

        # Previously generated cards upload section
        # st.markdown("---")
        # st.markdown("**Import Previously Generated Cards**")
        imported = st.file_uploader("**(Optional)** Upload JSON file to import existing flashcards", type="json", key="flash_uploader")
        if imported:
            try:
                data = json.load(imported)
                assert isinstance(data, list)
                st.session_state.flash_import_buffer = data
                st.success(f"Loaded {len(data)} cards. They will be used to avoid duplicates.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

        # Merge imported cards button
        if st.session_state.flash_import_buffer:
            if st.button("Merge imported cards into this chapter"):
                ch_idx = st.session_state.active_chapter_index
                st.session_state.flashcards_by_chapter.setdefault(ch_idx, [])
                st.session_state.flashcards_by_chapter[ch_idx].extend(st.session_state.flash_import_buffer)
                st.session_state.flash_import_buffer = []
                st.success("Imported cards merged.")
                st.rerun()

        # Custom instructions for flashcard generation
        # st.markdown("**Custom Instructions** *(Optional)*")
        if "custom_instructions" not in st.session_state:
            st.session_state.custom_instructions = ""
        
        st.session_state.custom_instructions = st.text_area(
            "**(Optional)** Add specific instructions to guide flashcard generation:",
            value=st.session_state.custom_instructions,
            placeholder="e.g., Focus on definitions and key concepts, avoid overly detailed questions, include more visual/diagram-based content, etc.",
            height=80,
            help="These instructions will be included in the prompt to help generate more targeted flashcards"
        )

        # Generation button and card count at the bottom
        # st.markdown("---")
        st.markdown("")
        cards = st.session_state.flashcards_by_chapter.get(st.session_state.active_chapter_index, [])
        
        col_gen, col_randomise, col_empty = st.columns([2.4, 2.4, 8], gap="small")
        with col_gen:
            if st.button("Generate (More) Flashcards", type="primary"):
                # Validate settings before generation
                if s["n_cards"] <= 0:
                    st.error("Please set the number of cards to generate to at least 1!")
                else:
                    _generate_flashcards_for_active_chapter()
        
        with col_randomise:
            if cards and st.button("Shuffle Flashcards", help="Shuffle the order of flashcards"):
                import random
                random.shuffle(cards)
                st.session_state.flashcards_by_chapter[st.session_state.active_chapter_index] = cards
                st.success("Cards shuffled!")
                st.rerun()
        
        # Card count display
        st.markdown(f"**{len(cards)} cards in this chapter**")

        # Display generated cards with enhanced formatting
        for i, c in enumerate(cards):
            card_type = c.get('type', 'basic')
            
            # Add type indicator to the expander title
            type_emoji = {
                'basic': '📝',
                'cloze': '🔤',
                'definition': '📖',
                'list': '📋',
                'true_false': '✅',
                'multiple_choice': '🔘',
                'compare_contrast': '⚖️'
            }
            
            emoji = type_emoji.get(card_type, '📝')
            
            # Show the complete question in the expander title without truncation
            with st.expander(f"{emoji} {i+1}. {c['q']}"):
                # Format answer based on card type
                if card_type == 'multiple_choice':
                    st.write(f"**Answer:** {c['a']}")
                elif card_type == 'list':
                    st.write(f"**Answer:** {c['a']}")
                else:
                    st.write(f"**Answer:** {c['a']}")
                
                # Enhanced metadata display
                type_display = card_type.replace('_', ' ').title()
                difficulty_display = c.get('difficulty', 'medium').title()
                source_display = c.get('source', 'Unknown')
                
                st.caption(f"**Type:** {type_display} | **Difficulty:** {difficulty_display}")
                st.caption(f"**Source:** {source_display}")
                
                # Optional per-card delete
                if st.button("Delete this card", key=f"del_{i}"):
                    del cards[i]
                    st.session_state.flashcards_by_chapter[st.session_state.active_chapter_index] = cards
                    st.rerun()
        
        # Export buttons
        if cards:
            st.markdown("---")
            colA, colB, colC, colD = st.columns([1, 1, 1, 4], gap="small")
            with colA:
                st.download_button(
                    "Download as JSON",
                    data=json.dumps(cards, ensure_ascii=False, indent=2),
                    file_name=f"{active_chapter['title']}_flashcards.json",
                    mime="application/json",
                    help="Download flashcards in JSON format importing into this app or other applications"
                )
            with colB:
                csv_bytes = _anki_csv_bytes(cards)
                st.download_button(
                    "Download as CSV",
                    data=csv_bytes,
                    file_name=f"{active_chapter['title']}_anki.csv",
                    mime="text/csv",
                    help="Download flashcards in CSV format for importing into Anki or other spaced repetition software"
                )
            with colC:
                if st.button("Delete All", type="secondary", help="Delete all flashcards for this chapter"):
                    # Delete all cards immediately
                    st.session_state.flashcards_by_chapter[st.session_state.active_chapter_index] = []
                    st.success("All flashcards deleted!")
                    st.rerun()
    
    # MCQ TAB
    with tab_mcq:
        def _get_active_mcqs() -> list[dict]:
            return st.session_state.mcq_by_chapter.get(st.session_state.active_chapter_index, [])

        st.subheader(f"Multiple Choice Questions for: {active_chapter['title']}")

        # Create sub-tabs for Generate, Practice, Review
        sub_tab_gen, sub_tab_practice, sub_tab_review = st.tabs(["Generate", "Practice", "Review & Export"])
        
        with sub_tab_gen:
            st.markdown("**Generation Settings**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.session_state.mcq_settings["n_qs"] = st.number_input("Number of MCQs", 1, 50, st.session_state.mcq_settings["n_qs"])
            with col2:
                st.session_state.mcq_settings["allow_multi"] = st.checkbox("Allow multi-select", value=st.session_state.mcq_settings["allow_multi"])
            with col3:
                st.session_state.mcq_settings["temperature"] = st.slider("Creativity (temp)", 0.0, 1.0, st.session_state.mcq_settings["temperature"], 0.05)
            with col4:
                choices = ["easy","medium","hard"]
                current = st.session_state.mcq_settings["difficulty_mix"]
                st.session_state.mcq_settings["difficulty_mix"] = st.multiselect("Difficulty mix", choices, default=current)

            col_gen, col_from_flash, col_empty = st.columns([2, 2, 6])
            with col_gen:
                if st.button("Generate MCQs from Chapter", type="primary"):
                    if not st.session_state.mcq_settings["difficulty_mix"]:
                        st.error("Please select at least one difficulty level!")
                    else:
                        generate_mcqs_for_active_chapter()
            
            with col_from_flash:
                flashcards = st.session_state.flashcards_by_chapter.get(st.session_state.active_chapter_index, [])
                if flashcards and st.button("Convert from Flashcards"):
                    converted_mcqs = mcqs_from_flashcards(flashcards, min(10, len(flashcards)))
                    st.session_state.mcq_by_chapter.setdefault(st.session_state.active_chapter_index, []).extend(converted_mcqs)
                    st.success(f"Converted {len(converted_mcqs)} flashcards to MCQs!")
                    st.rerun()

            # Show current MCQ count
            mcqs = _get_active_mcqs()
            st.markdown(f"**{len(mcqs)} MCQs in this chapter**")

        with sub_tab_practice:
            mcqs = list(_get_active_mcqs())
            if not mcqs:
                st.info("No MCQs yet. Generate some in the 'Generate' tab.")
            else:
                import random
                
                # Practice settings
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.mcq_settings["show_expl_immediately"] = st.checkbox(
                        "Show explanation immediately", 
                        value=st.session_state.mcq_settings["show_expl_immediately"]
                    )
                with col2:
                    st.session_state.mcq_settings["timer_seconds"] = st.number_input(
                        "Timer per question (0 = no timer)", 
                        0, 300, 
                        st.session_state.mcq_settings["timer_seconds"]
                    )

                # Create/restore a quiz session
                ch_idx = st.session_state.active_chapter_index
                sess = st.session_state.mcq_sessions.get(ch_idx)
                start_new = st.button("Start New Quiz", type="primary") or (sess is None)

                if start_new:
                    # Shuffle questions and options for this attempt
                    q_order = list(range(len(mcqs)))
                    random.shuffle(q_order)
                    shuffled = []
                    for qi in q_order:
                        q = dict(mcqs[qi])
                        opt_perm = list(range(len(q["options"])))
                        random.shuffle(opt_perm)
                        q["options"] = [q["options"][i] for i in opt_perm]
                        # map correct indices through permutation
                        corr = set(q["correct"])
                        q["correct"] = [i for i, old in enumerate(opt_perm) if old in corr]
                        q["_opt_perm"] = opt_perm  # keep if you need review mapping
                        shuffled.append(q)

                    sess = {
                        "idx": 0,
                        "items": shuffled,
                        "responses": [None] * len(shuffled),
                        "correct_flags": [False] * len(shuffled),
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "show_expl_immediately": st.session_state.mcq_settings["show_expl_immediately"],
                        "timer_seconds": st.session_state.mcq_settings["timer_seconds"],
                    }
                    st.session_state.mcq_sessions[ch_idx] = sess
                    st.rerun()

                if sess:
                    # Render current question
                    q = sess["items"][sess["idx"]]
                    st.markdown(f"**Question {sess['idx']+1} of {len(sess['items'])}**")
                    st.markdown(f"**{q['stem']}**")
                    
                    multi = len(q["correct"]) > 1
                    user_sel = None

                    if multi:
                        # Multi-select as checkboxes
                        st.info("Multiple answers may be correct. Select all that apply:")
                        selections = []
                        for i, opt in enumerate(q["options"]):
                            if st.checkbox(f"{chr(65+i)}. {opt}", key=f"mcq_{q['id']}_{i}"):
                                selections.append(i)
                        user_sel = sorted(selections)
                    else:
                        # Single select as radio buttons
                        options_with_letters = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(q["options"])]
                        selected_option = st.radio("Choose one:", options_with_letters, index=None, key=f"mcq_{q['id']}_radio")
                        if selected_option is not None:
                            user_sel = [options_with_letters.index(selected_option)]

                    # Control buttons
                    colA, colB, colC = st.columns([1,1,2])
                    with colA:
                        check = st.button("Check Answer", disabled=(user_sel is None))
                    with colB:
                        nxt = st.button("Next ➜", disabled=(sess["idx"] >= len(sess["items"]) - 1))
                    with colC:
                        if sess["idx"] < len(sess["items"]) - 1:
                            st.write(f"Progress: {sess['idx']+1}/{len(sess['items'])}")

                    # Handle answer checking
                    if check and user_sel is not None:
                        is_correct = set(user_sel) == set(q["correct"])
                        sess["responses"][sess["idx"]] = user_sel
                        sess["correct_flags"][sess["idx"]] = is_correct
                        st.session_state.mcq_sessions[ch_idx] = sess  # Update session

                    # Show feedback
                    if sess["responses"][sess["idx"]] is not None and (sess["show_expl_immediately"] or check):
                        is_correct = sess["correct_flags"][sess["idx"]]
                        if is_correct:
                            st.success("✅ Correct! 🎉")
                        else:
                            st.error("❌ Not quite right.")
                            
                        # Show correct answer(s)
                        correct_letters = [chr(65+i) for i in q["correct"]]
                        st.info(f"**Correct answer(s):** {', '.join(correct_letters)}")
                        
                        if q.get('explanation'):
                            st.markdown(f"**Explanation:** {q['explanation']}")

                    # Advance to next question
                    if nxt:
                        sess["idx"] += 1
                        st.session_state.mcq_sessions[ch_idx] = sess
                        st.rerun()

                    # Show final results
                    if sess["idx"] >= len(sess["items"]) - 1 and sess["responses"][-1] is not None:
                        st.markdown("---")
                        st.subheader("Quiz Complete! 🎊")
                        total = len(sess["items"])
                        score = sum(sess["correct_flags"])
                        percentage = round(score * 100 / total) if total > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Score", f"{score}/{total}")
                        with col2:
                            st.metric("Percentage", f"{percentage}%")
                        with col3:
                            st.metric("Questions", total)
                        
                        st.progress(score/total if total > 0 else 0)
                        
                        # Performance feedback
                        if percentage >= 90:
                            st.success("Excellent work! You've mastered this material! 🌟")
                        elif percentage >= 70:
                            st.success("Good job! You have a solid understanding. 👍")
                        elif percentage >= 50:
                            st.warning("Not bad, but there's room for improvement. Keep studying! 📚")
                        else:
                            st.error("You might want to review this chapter more thoroughly. 🔄")

        with sub_tab_review:
            mcqs = _get_active_mcqs()
            if not mcqs:
                st.info("No MCQs to review yet.")
            else:
                st.markdown(f"**{len(mcqs)} MCQs for review:**")
                
                # Simple review interface
                for i, q in enumerate(mcqs):
                    with st.expander(f"Q{i+1}: {q['stem'][:80]}{'...' if len(q['stem']) > 80 else ''}"):
                        st.markdown(f"**Question:** {q['stem']}")
                        
                        # Show options with letters
                        st.markdown("**Options:**")
                        for j, opt in enumerate(q['options']):
                            marker = "✅" if j in q['correct'] else "○"
                            st.markdown(f"{marker} {chr(65+j)}. {opt}")
                        
                        if q.get('explanation'):
                            st.markdown(f"**Explanation:** {q['explanation']}")
                        
                        # Metadata
                        st.caption(f"**Difficulty:** {q.get('difficulty', 'medium').title()} | **Source:** {q.get('source', 'Unknown')}")
                        
                        # Delete individual MCQ
                        if st.button("Delete this MCQ", key=f"del_mcq_{i}"):
                            del mcqs[i]
                            st.session_state.mcq_by_chapter[st.session_state.active_chapter_index] = mcqs
                            st.rerun()

                # Export and delete options
                if mcqs:
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 4])
                    
                    with col1:
                        # JSON Export
                        st.download_button(
                            "Download JSON",
                            data=json.dumps(mcqs, ensure_ascii=False, indent=2),
                            file_name=f"{active_chapter['title']}_mcqs.json",
                            mime="application/json",
                            help="Download MCQs in JSON format"
                        )
                    
                    with col2:
                        # CSV Export
                        output = StringIO()
                        writer = csv.writer(output)
                        writer.writerow(["stem","options","correct","explanation","difficulty","tags","source"])
                        for q in mcqs:
                            writer.writerow([
                                q["stem"],
                                " | ".join(q["options"]),
                                ",".join(map(str, q["correct"])),
                                q.get("explanation",""),
                                q.get("difficulty",""),
                                ",".join(q.get("tags",[])),
                                q.get("source",""),
                            ])
                        
                        st.download_button(
                            "Download CSV",
                            data=output.getvalue().encode("utf-8"),
                            file_name=f"{active_chapter['title']}_mcqs.csv",
                            mime="text/csv",
                            help="Download MCQs in CSV format"
                        )
                    
                    with col3:
                        if st.button("Delete All MCQs", type="secondary"):
                            st.session_state.mcq_by_chapter[st.session_state.active_chapter_index] = []
                            st.success("All MCQs deleted!")
                            st.rerun()
