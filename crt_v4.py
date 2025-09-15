from __future__ import annotations

import html
import time
import mimetypes
import os
import re
import uuid
import json
import csv
import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Any
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
CHAPTER_CHAR_LIMIT = 1000000  # adjust to your model

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

# --- Chapter schema (structured output for auto extraction) ---
class ChapterDef(BaseModel):
    title: str = Field(description="Exact top-level chapter or unit title as it appears in the PDF table of contents or headings.")
    start_page: int = Field(ge=1, description="Start page number, using the PDF's 1-based page order.")
    end_page: int = Field(ge=1, description="Inclusive end page number, using the PDF's 1-based page order.")

# --- ToC-first light extraction schema (model returns ToC labels + chapter ranges) ---
class TocChapter(BaseModel):
    title: str = Field(description="Exact top-level chapter/part title as written in the ToC (no normalization beyond trimming).")
    toc_start_page: int = Field(ge=1, description="The page number as printed in the Table of Contents for this chapter/part (an integer).")
    inferred_end_page: int = Field(ge=1, description="The inferred end page for this chapter based on ToC analysis and document structure. Should be the page before the next chapter starts, or before document end matter if it's the last chapter.")

class TocExtraction(BaseModel):
    chapters: List[TocChapter] = Field(description="Top-level chapters/parts in ToC order with their printed page numbers and inferred end pages.")
    first_chapter_toc_page: Optional[int] = Field(
        default=None,
        description=(
            "The page number as printed in the Table of Contents for the FIRST chapter (should match chapters[0].toc_start_page)."
        ),
    )
    first_chapter_actual_page: Optional[int] = Field(
        default=None,
        description=(
            "The actual 1-based PDF page index where the first chapter's heading appears in the document body."
        ),
    )

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
    qtype: Literal[
        "cloze","definition","list","true_false","multiple_choice","compare_contrast",
        "assertion_reason","sequence_ordering","matching_matrix","k_type","best_next_step",
        "cause_effect","negative"
    ] = "multiple_choice"
    id: str = ""        # filled by app
    hash: str = ""      # normalized stem hash for dedupe
    meta: Optional[Dict[str, Any]] = None  # optional metadata e.g., entities, format hints

# NOTE: Gemini typed response schemas do not support free-form objects (additionalProperties).
# To avoid schema errors, use this minimal schema for model responses and enrich later in-app.
class MCQGen(BaseModel):
    stem: str
    options: List[str]
    correct: List[int]
    explanation: str = ""
    difficulty: Literal["easy","medium","hard"] = "medium"
    tags: List[str] = []
    source: str = ""
    qtype: Literal[
        "cloze","definition","list","true_false","multiple_choice","compare_contrast",
        "assertion_reason","sequence_ordering","matching_matrix","k_type","best_next_step",
        "cause_effect","negative"
    ] = "multiple_choice"

def _norm_stem(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _stem_hash(s: str) -> str:
    return hashlib.sha1(_norm_stem(s).encode()).hexdigest()

def _mcq_dedupe_hashes(existing: List[dict]) -> set[str]:
    return {_stem_hash(x.get("stem","")) for x in existing if x.get("stem")}

def _mcq_norm_stems(existing: List[dict]) -> set[str]:
    """Normalized stems for prompt-time semantic-ish dedupe and planning."""
    return {_norm_stem(x.get("stem", "")) for x in existing if x.get("stem")}

def _assign_ids_and_hashes(items: List[dict]) -> List[dict]:
    out = []
    for itm in items:
        itm["hash"] = _stem_hash(itm["stem"])
        if not itm.get("id"):
            itm["id"] = hashlib.md5((itm["hash"] + "|".join(itm["options"])).encode()).hexdigest()
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

def validate_and_sort_chapter_ranges(chapters: list[dict], total_pages: Optional[int] = None) -> tuple[list[dict], list[str]]:
    """
    Normalize chapters to start/end, sort by start_page, and detect overlaps.
    Accepts either {title, start_page, end_page} or {title, pages: [...]}
    Returns (normalized_sorted_list, warnings)
    """
    norm: list[dict] = []
    warns: list[str] = []

    for ch in chapters:
        title = (ch.get("title") or "").strip()
        if not title:
            continue
        if "pages" in ch and ch.get("pages"):
            s = int(min(ch["pages"]))
            e = int(max(ch["pages"]))
        else:
            s = int(ch.get("start_page", 0) or 0)
            e = int(ch.get("end_page", 0) or 0)
        if s > e:
            s, e = e, s
        if total_pages:
            # Clamp start to [1, total_pages]
            if s < 1:
                warns.append(f"‘{title}’ start_page {s} < 1. Clamping to 1.")
                s = 1
            if s > total_pages:
                warns.append(f"‘{title}’ start_page {s} > total pages {total_pages}. Clamping to {total_pages}.")
                s = total_pages
            # Clamp end to [1, total_pages]
            if e < 1:
                warns.append(f"‘{title}’ end_page {e} < 1. Clamping to 1.")
                e = 1
            if e > total_pages:
                warns.append(f"‘{title}’ end_page {e} > total pages {total_pages}. Clamping to {total_pages}.")
                e = total_pages
        if title and s >= 1 and e >= 1:
            norm.append({"title": title, "start_page": s, "end_page": e})

    norm.sort(key=lambda x: x["start_page"])  # order by start

    for i in range(1, len(norm)):
        prev, cur = norm[i-1], norm[i]
        if cur["start_page"] <= prev["end_page"]:
            warns.append(
                f"‘{cur['title']}’ ({cur['start_page']}-{cur['end_page']}) overlaps previous "
                f"‘{prev['title']}’ ({prev['start_page']}-{prev['end_page']})."
            )

    return norm, warns

# --- Gemini API Interaction ---
def to_part(text: str) -> dict:
    """Return a Gem-compatible PartDict for a text string."""
    return {"text": text}

def to_content(role: str, text: str) -> dict:
    """Return a Gem-compatible Content dict (role=user|model)."""
    return {"role": role, "parts": [to_part(text)]}

def extract_chapters_with_gemini(uploaded_file, client: genai.Client) -> List[Dict]:
    """
    Use gemini-2.5-flash-lite to extract chapter definitions from a PDF via Files API.
    Returns a list of dicts: [{title, start_page, end_page}, ...].
    """
    # Determine MIME type (force PDF as the app restricts uploads to PDF)
    mime = "application/pdf"

    # Attempt a lightweight subset approach: extract ToC pages and a few anchor pages.
    subset_text = None
    subset_meta = {}
    
    
    try:
        with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as _doc:
            total_pages = len(_doc)
            scan_limit = min(40, total_pages)  # scan first N pages to find ToC
            

            def _page_text(pi0: int) -> str:
                try:
                    return _doc.load_page(pi0).get_text("text") or ""
                except Exception:
                    return ""

            toc_candidates = []  # 0-based indices likely containing ToC
            dotline_re = re.compile(r"\.{2,}\s+\d{1,4}\s*$")
            toc_kw_re = re.compile(r"\b(contents|table of contents|目錄|目录)\b", re.IGNORECASE)
            for i in range(scan_limit):
                t = _page_text(i)
                if not t:
                    continue
                lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
                dot_count = sum(1 for ln in lines if dotline_re.search(ln))
                has_kw = bool(toc_kw_re.search(t))
                if has_kw or dot_count >= 3:
                    toc_candidates.append(i)

            # Collapse consecutive candidates into ranges, use full span from earliest to latest
            toc_start = toc_end = None
            if toc_candidates:
                
                ranges = []
                start = toc_candidates[0]
                prev = start
                for x in toc_candidates[1:]:
                    if x == prev + 1:
                        prev = x
                        continue
                    ranges.append((start, prev))
                    start = prev = x
                ranges.append((start, prev))
                
                # Select from earliest detected page to latest detected page across ALL ranges
                toc_start = toc_candidates[0]  # Earliest detected page
                toc_end = toc_candidates[-1]   # Latest detected page
                
                if toc_end - toc_start > 19:  # Only trim if range exceeds 20 pages
                    toc_end = toc_start + 19  # Limit to 20 pages (0-19 inclusive)

            # Build subset payload if we found likely ToC
            if toc_start is not None:
                
                parts = []
                parts.append(f"DOC_TOTAL_PAGES: {total_pages}")
                # ToC pages (1-based labels included)
                for i in range(toc_start, toc_end + 1):
                    parts.append(f"\n=== TOC_PAGE (pdf_page={i+1}) ===\n" + _page_text(i))

                # Anchor windows: a small window right after ToC, and a later window
                def _add_window(start_i0: int, count: int, label: str):
                    if start_i0 < 0 or start_i0 >= total_pages:
                        return
                    end_i0 = min(total_pages - 1, start_i0 + count - 1)
                    for j in range(start_i0, end_i0 + 1):
                        parts.append(f"\n=== {label} (pdf_page={j+1}) ===\n" + _page_text(j))

                _add_window(toc_end + 1, 10, "ANCHOR_NEAR_TOC")

                subset_text = "\n".join(parts)
                subset_meta = {
                    "mode": "subset",
                    "toc_pages": [toc_start + 1, toc_end + 1],
                    "total_pages": total_pages,
                }
                
    except Exception as _e:
        subset_text = None
        subset_meta = {"mode": "subset_error", "error": str(_e)}

    # If no ToC subset was built, fall back to a generic front-window excerpt so the model can still parse ToC
    if not subset_text:
        
        try:
            with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as _doc2:
                total_pages2 = len(_doc2)
                front = min(30, total_pages2)
                
                parts2 = [f"DOC_TOTAL_PAGES: {total_pages2}"]
                for i in range(front):
                    try:
                        parts2.append(f"\n=== FRONT_WINDOW (pdf_page={i+1}) ===\n" + _doc2.load_page(i).get_text("text"))
                    except Exception:
                        continue
                subset_text = "\n".join(parts2)
                subset_meta = {"mode": "front_window", "total_pages": total_pages2, "pages": [1, front]}
                
        except Exception as _e2:
            subset_text = None
            subset_meta = {"mode": "subset_error", "error": f"front_window_fallback_failed: {_e2}"}

    # Decide mode: use subset if we have at least one ToC page; else error
    use_subset = bool(subset_text and subset_text.strip())
    

    # If no subset text available, raise error instead of falling back to Files API
    if not use_subset:
        raise RuntimeError("Could not detect table of contents or extract chapter information. Please define chapters manually.")

    # System instruction: return ToC chapters with inferred end pages
    system_inst = (
        "Task: Read the Table of Contents (ToC) and return chapters with inferred end pages.\n\n"
        "CRITICAL FIRST STEP: Find the actual PDF page where Chapter 1 appears in the document body.\n"
        "This is essential for accurate page calculations. Search thoroughly through the document.\n\n"
        "Definitions:\n"
        "- toc_start_page: The page number as printed in the ToC for a chapter/part (e.g., 1, 23, 145).\n"
        "- inferred_end_page: The page where this chapter logically ends (typically next chapter start - 1).\n"
        "- first_chapter_toc_page: The page number as printed in the ToC for Chapter 1.\n"
        "- first_chapter_actual_page: The actual PDF page where Chapter 1's heading appears (CRITICAL!).\n\n"
        "Instructions:\n"
        "1. FIRST: Search the document body and find where 'Chapter 1' (or first numbered chapter) actually appears. Record this page number.\n"
        "2. Parse ToC for chapter titles and their printed page numbers.\n"
        "3. For each chapter, infer end page: next chapter start - 1, or references start - 1 for last chapter.\n"
        "4. EXCLUDE front matter: preface, foreword, acknowledgments, non-numbered introduction.\n"
        "5. INCLUDE: numbered chapters, named content chapters, parts, units.\n\n"
        "Output Schema (JSON): { chapters: [{title, toc_start_page, inferred_end_page}], first_chapter_toc_page?: int, first_chapter_actual_page?: int }."
    )

    cfg = gx.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TocExtraction,
        temperature=0.0,
        system_instruction=system_inst,
    )

    prompt = (
        "PRIORITY 1: Find where Chapter 1 actually appears in the document body (not just ToC).\n"
        "PRIORITY 2: Parse ToC and infer chapter end pages.\n\n"
        "Return JSON: { chapters: [{title, toc_start_page, inferred_end_page}], first_chapter_toc_page?: int, first_chapter_actual_page?: int }.\n"
        "\n"
        "CRITICAL: Search through the document body to find the actual page where Chapter 1 heading appears. This is different from the ToC page number.\n"
        "\n"
        "For chapters - Only include content chapters. EXCLUDE:\n"
        "- Preface, Foreword, Acknowledgments\n"
        "- Introduction (unless numbered as Chapter 1)\n" 
        "- References, Bibliography, Index, Appendices, Glossary\n"
        "\n"
        "For inferred_end_page - Simple rule:\n"
        "- End = next chapter start - 1\n"
        "- Last chapter: end = references start - 1\n"
        "\n"
        "REMEMBER: The first_chapter_actual_page is the most critical field for accurate results."
    )


    # Generate model response using subset text only
    try:
        contents = f"{prompt}\n\n--- DOCUMENT EXCERPT ---\n{subset_text}"
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=cfg,
        )
    except Exception as e:
        raise RuntimeError(f"Gemini ToC extraction call failed: {e}")

    # Prefer parsed (TocExtraction)
    toc_result: Dict = {}
    if getattr(resp, "parsed", None):
        pr = resp.parsed
        if hasattr(pr, "model_dump"):
            toc_result = pr.model_dump(exclude_none=True)
        elif isinstance(pr, dict):
            toc_result = pr
        else:
            # Fallback to text JSON
            toc_result = json.loads(getattr(resp, "text", "{}") or "{}")
    elif getattr(resp, "text", None):
        toc_result = json.loads(resp.text)
    else:
        raise RuntimeError("No response from Gemini ToC extraction")


    # Build absolute chapter ranges from chapters with model-inferred end pages + offset
    chapters_data = toc_result.get("chapters") or []
    
    # Normalize and filter chapters
    norm_chapters: List[Dict] = []
    front_matter_keywords = {
        "preface", "foreword", "acknowledgment", "acknowledgments", "introduction", 
        "references", "bibliography", "index", "appendix", "appendices", "glossary",
        "table of contents", "contents", "list of figures", "list of tables"
    }
    
    for ch in chapters_data:
        # Normalize title to single line: collapse all whitespace (incl. newlines) to a single space
        raw_title = (ch.get("title") or "")
        t = re.sub(r"\s+", " ", raw_title).strip()
        
        # Check if this looks like front matter (case-insensitive)
        t_lower = t.lower()
        is_front_matter = any(keyword in t_lower for keyword in front_matter_keywords)
        
        # Skip if it looks like front matter, unless it's clearly a numbered chapter
        if is_front_matter and not re.search(r'\b(chapter|ch\.?)\s*\d+', t_lower):
            continue
            
        try:
            toc_start = int(ch.get("toc_start_page") or 0)
            inferred_end = int(ch.get("inferred_end_page") or 0)
        except Exception:
            continue
            
        if t and toc_start and inferred_end and toc_start > 0 and inferred_end > 0:
            norm_chapters.append({"title": t, "toc_start_page": toc_start, "inferred_end_page": inferred_end})

    if not norm_chapters:
        st.warning("No top-level chapters found in ToC.")
        return []

    # Additional validation: warn if first item still looks like front matter
    first_title = norm_chapters[0]["title"].lower()
    if any(keyword in first_title for keyword in ["preface", "foreword", "acknowledgment", "introduction"]):
        st.warning(f"⚠️ The first chapter appears to be front matter: '{norm_chapters[0]['title']}'. You may need to manually adjust the chapter definitions.")

    # Determine total PDF pages for clamping
    total_pages = None
    try:
        with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as _doc:
            total_pages = _doc.page_count
    except Exception:
        pass

    # Compute offset programmatically from the two page numbers provided by the model
    offset = None
    first_chapter_toc_page = toc_result.get("first_chapter_toc_page")
    first_chapter_actual_page = toc_result.get("first_chapter_actual_page")
    
    
    # If model provided the page numbers, use them
    if first_chapter_toc_page and first_chapter_actual_page:
        try:
            offset = first_chapter_actual_page - first_chapter_toc_page
        except Exception as e:
            offset = None
    
    # Fallback: find first numbered chapter programmatically
    if offset is None and first_chapter_actual_page and norm_chapters:
        
        try:
            first_ch = next(ch for ch in norm_chapters if re.search(r'\b(chapter|ch\.?)\s*1\b', ch["title"], re.I))
            offset = first_chapter_actual_page - first_ch["toc_start_page"]
        except Exception as e:
            offset = None
    
    if offset is None:
        st.warning("Could not calculate offset from model response; defaulting to 0. You may need to adjust chapter ranges manually.")
        offset = 0
    else:
        # Handle negative offsets by taking absolute value and warning user
        if offset < 0:
            st.warning(f"Calculated negative offset ({offset}). Using absolute value {abs(offset)}. Please verify chapter ranges.")
            offset = abs(offset)

    # Apply offset to model-inferred chapter ranges
    final_chapters: List[Dict] = []
    
    for ch in norm_chapters:
        abs_start = ch["toc_start_page"] + offset
        abs_end = ch["inferred_end_page"] + offset
        
        # Clamp to document bounds
        if total_pages:
            abs_start = max(1, min(abs_start, total_pages))
            abs_end = max(1, min(abs_end, total_pages))
        
        # Ensure end >= start
        if abs_end < abs_start:
            abs_end = abs_start
        
        # Ensure title is single-line when stored
        title = re.sub(r"\s+", " ", ch["title"]).strip()
        final_chapters.append({"title": title, "start_page": abs_start, "end_page": abs_end})


    # Validate/sort and clamp the model-inferred chapter ranges
    norm, warns = validate_and_sort_chapter_ranges(final_chapters, total_pages=total_pages)
    for w in warns:
        st.warning(w)
    
    
    return norm

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
    norm_stems = {_h(c.get("q","")) for c in all_cards if c.get("q")}
    return all_cards, norm_stems

def _chapter_excerpt(text: str) -> str:
    # Chapter text is already limited by CHAPTER_CHAR_LIMIT during initial processing
    return text

def _build_prompt(chapter_title: str, chapter_text: str, settings: dict, exclude_norm_stems: set[str]) -> str:
    n = settings["n_cards"]
    types_s = settings["types"]  # Now a single string instead of list
    diff = settings["difficulty"]
    
    # Build the base prompt with detailed instructions for each card type
    type_instructions = {
        "basic": "Basic Q&A format with straightforward questions and answers.",
        "cloze": "Fill-in-the-blank format where key words or phrases are replaced with _____. Example: 'The _____ is responsible for executive functions' → 'The prefrontal cortex is responsible for executive functions'",
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

Known question stems to avoid (normalized): {sorted(list(exclude_norm_stems))[:500]}

CHAPTER CONTENT:
---
{chapter_text}
---
"""
    
    return base_prompt

def _generate_flashcards_for_active_chapter():
    ch = st.session_state.chapters[st.session_state.active_chapter_index]
    s = st.session_state.flash_settings
    _, exclude_norm_stems = _active_cards_and_hashes()

    prompt = _build_prompt(ch["title"], _chapter_excerpt(ch["text"]), s, exclude_norm_stems)


    with st.spinner("Generating flashcards…"):
        try:
            # Use the global client
            from google import genai as _genai_new
            flash_client = _genai_new.Client(api_key=API_KEY)
            
            cfg = gx.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[Flashcard],
                temperature=float(s["temperature"]),
                system_instruction=f"Avoid generating flashcards whose questions are semantically similar to any of these normalized stems: {sorted(list(exclude_norm_stems))[:500]}"
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
                c["source"] = source[:300]  # Enforce length limit
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
def _chapter_excerpt_for_mcq(text: str) -> str:
    return text

def _is_two_clause(opt: str) -> bool:
    """Two-clause option with a clear divider between entity A and entity B."""
    if not isinstance(opt, str):
        return False
    return bool(re.search(r".+\s(?:;|—|–)\s.+", opt.strip()))


def _looks_comparative(stem: str) -> bool:
    """Check for explicit comparative cue in stem."""
    if not isinstance(stem, str):
        return False
    return bool(re.search(r"\b(vs\.?|versus|between|compare|contrast|differentiat\w*)\b", stem, re.IGNORECASE))


def _salvage_compare_to_mcq(m: dict) -> dict | None:
    """If compare/contrast validation fails but item is a valid single-best MCQ, downgrade type."""
    try:
        correct = m.get("correct")
        opts = m.get("options")
        if isinstance(correct, list) and len(correct) == 1 and isinstance(opts, list) and len(opts) >= 4:
            m2 = dict(m)
            m2["qtype"] = "multiple_choice"
            tags = (m2.get("tags") or [])
            if "downgraded_from_contrast" not in tags:
                tags.append("downgraded_from_contrast")
            m2["tags"] = tags
            return m2
    except Exception:
        return None
    return None


def _looks_negative(stem: str) -> bool:
    return bool(re.search(r"\b(EXCEPT|NOT)\b", stem or "", re.I))


def _is_permutation(s: str) -> bool:
    """Detect simple token permutations like 'A→B→C→D' or 'A-B-C-D' or '1→2→3'."""
    if not isinstance(s, str):
        return False
    txt = s.strip()
    return bool(re.search(r"^[A-Za-z0-9]+(\s*[→\-–>]\s*[A-Za-z0-9]+){2,}$", txt))


def _is_mapping(s: str) -> bool:
    """Detect mapping like 'A→2, B→4, C→1, D→3'. Looser variant allows more than A-D/1-4."""
    if not isinstance(s, str):
        return False
    txt = s.strip()
    return bool(re.search(r"^[A-Za-z]\s*→\s*[0-9]+(\s*,\s*[A-Za-z]\s*→\s*[0-9]+){2,}$", txt))

def _filter_mcqs_by_qtype(mcqs: list[dict], qtype: str, *, audit: Optional[dict] = None, salvage_non_contrast: bool = True) -> list[dict]:
    """Enforce qtype-specific structural constraints post-generation.

    audit: optional dict to collect reasons for rejection/downgrade.
    """
    out: list[dict] = []
    if audit is not None:
        audit.setdefault("input", 0)
        audit.setdefault("kept", 0)
        audit.setdefault("downgraded", 0)
        audit.setdefault("reasons", [])
    for m in mcqs:
        if audit is not None:
            audit["input"] += 1
        stem = (m.get("stem") or "").strip()
        options = m.get("options") or []
        correct = m.get("correct") or []

        # Common sanity checks
        if not stem or not isinstance(options, list) or not options:
            if audit is not None:
                audit["reasons"].append("missing stem/options")
            continue
        if not isinstance(correct, list) or not correct:
            if audit is not None:
                audit["reasons"].append("missing/invalid correct indices")
            continue

        # Cardinality
        if qtype in ("list", "k_type"):
            ok = len(correct) >= 2
        else:
            ok = len(correct) == 1
        if not ok:
            if audit is not None:
                audit["reasons"].append("cardinality mismatch")
            continue

        if qtype == "true_false":
            # Normalize to strict True/False options and enforce canonical order ["True","False"]
            if len(options) != 2:
                if audit is not None:
                    audit["reasons"].append("true_false not 2 options")
                continue
            opt_norm = [str(o).strip().lower() for o in options]
            # Broaden acceptance to common variants
            mapping = {
                "t": "True", "true": "True", "yes": "True", "y": "True", "agree": "True",
                "f": "False", "false": "False", "no": "False", "n": "False", "disagree": "False",
            }
            if all(x in mapping for x in opt_norm):
                mapped = [mapping[opt_norm[0]], mapping[opt_norm[1]]]
                # Reorder to ["True","False"] if needed and adjust correct index
                if mapped == ["True", "False"]:
                    m["options"] = mapped
                elif mapped == ["False", "True"]:
                    m["options"] = ["True", "False"]
                    # adjust single-correct index (already enforced single-correct below)
                    if len(correct) == 1:
                        correct = [0 if correct[0] == 1 else 1]
                        m["correct"] = correct
                else:
                    # If somehow mapped to other words, reject
                    if audit is not None:
                        audit["reasons"].append("true_false options not canonical True/False after mapping")
                    continue
            else:
                if audit is not None:
                    audit["reasons"].append("true_false options not coercible")
                continue

        elif qtype == "compare_contrast":
            # 1) Stem must look comparative
            if not _looks_comparative(stem):
                if salvage_non_contrast:
                    salv = _salvage_compare_to_mcq(m)
                    if salv is not None:
                        out.append(salv)
                        if audit is not None:
                            audit["downgraded"] += 1
                            audit["reasons"].append("compare_contrast: non-comparative stem -> downgraded")
                        continue
                if audit is not None:
                    audit["reasons"].append("compare_contrast: non-comparative stem")
                continue
            # 2) At least 4 options
            if len(options) < 4:
                if audit is not None:
                    audit["reasons"].append("compare_contrast: <4 options")
                continue
            # 3) Two-clause options
            if not all(isinstance(o, str) and _is_two_clause(o) for o in options):
                if audit is not None:
                    audit["reasons"].append("compare_contrast: options not two-clause")
                continue
            # 4) Forbid All/None/Both/Neither
            bad = r"\b(all of the above|none of the above|both|neither)\b"
            if any(re.search(bad, str(o), re.I) for o in options):
                if audit is not None:
                    audit["reasons"].append("compare_contrast: forbidden distractor")
                continue

        elif qtype == "assertion_reason":
            # Require A: ... and R: ... in stem
            if not re.search(r"\bA:\b.*\bR:\b", stem, re.S):
                if audit is not None:
                    audit["reasons"].append("assertion_reason: missing A:/R: in stem")
                continue
            # Exactly 4 canonical options and single correct
            if len(options) != 4 or len(correct) != 1:
                if audit is not None:
                    audit["reasons"].append("assertion_reason: need 4 options and single correct")
                continue
            patterns = [
                "Both A and R are true, and R explains A",
                "Both A and R are true, but R does not explain A",
                "A is true, but R is false",
                "A is false, but R is true",
            ]
            # Each canonical phrase should appear at least once across options
            if not all(any(re.search(re.escape(p), str(o), re.I) for o in options) for p in patterns):
                if audit is not None:
                    audit["reasons"].append("assertion_reason: options not canonical AR4 set")
                continue

        elif qtype == "sequence_ordering":
            if len(correct) != 1 or len(options) < 4:
                if audit is not None:
                    audit["reasons"].append("sequence_ordering: need >=4 options, single correct")
                continue
            if not all(isinstance(o, str) and _is_permutation(o) for o in options):
                if audit is not None:
                    audit["reasons"].append("sequence_ordering: options not permutations")
                continue

        elif qtype == "matching_matrix":
            if len(correct) != 1 or len(options) < 4:
                if audit is not None:
                    audit["reasons"].append("matching_matrix: need >=4 options, single correct")
                continue
            if not all(isinstance(o, str) and _is_mapping(o) for o in options):
                if audit is not None:
                    audit["reasons"].append("matching_matrix: options not mappings")
                continue

        elif qtype == "k_type":
            # already enforced multi-correct above; ensure statements look sentence-like
            if len(options) < 4 or len(correct) < 2:
                if audit is not None:
                    audit["reasons"].append("k_type: require >=4 options and >=2 correct")
                continue
            if not all(len(str(o).split()) >= 3 for o in options):
                if audit is not None:
                    audit["reasons"].append("k_type: options too short/non-statement")
                continue

        elif qtype == "best_next_step":
            if len(correct) != 1 or len(options) < 4:
                if audit is not None:
                    audit["reasons"].append("best_next_step: need >=4 options, single correct")
                continue
            # rudimentary scenario heuristic
            if len(stem.split()) < 20 or not re.search(r"\b(best next step|should|first)\b", stem, re.I):
                if audit is not None:
                    audit["reasons"].append("best_next_step: stem not scenario/next-step style")
                continue

        elif qtype == "cause_effect":
            if len(correct) != 1 or len(options) < 4:
                if audit is not None:
                    audit["reasons"].append("cause_effect: need >=4 options, single correct")
                continue
            kws = ("cause", "causal", "causes")
            has_causal = sum(1 for o in options if any(k in str(o).lower() for k in kws)) >= 2
            has_skeptic = any(re.search(r"(no caus|correlation|association.*not.*caus)", str(o), re.I) for o in options)
            if not (has_causal and has_skeptic):
                if audit is not None:
                    audit["reasons"].append("cause_effect: options lack causal vs non-causal balance")
                continue

        elif qtype == "negative":
            if len(correct) != 1 or len(options) < 4:
                if audit is not None:
                    audit["reasons"].append("negative: need >=4 options, single correct")
                continue
            if not _looks_negative(stem):
                if audit is not None:
                    audit["reasons"].append("negative: missing NOT/EXCEPT cue in stem")
                continue
            if any(re.search(r"\b(all of the above|none of the above)\b", str(o), re.I) for o in options):
                if audit is not None:
                    audit["reasons"].append("negative: forbidden all/none distractor")
                continue

        # Set the qtype field to the selected type for consistency
        m["qtype"] = qtype
        out.append(m)
        if audit is not None:
            audit["kept"] += 1
    return out

def _build_mcq_prompt(chapter_title: str, chapter_text: str, settings: dict, exclude_norm_stems: set[str]) -> str:
    n = settings["n_qs"]
    qtype = settings.get("qtype", "multiple_choice")
    allow_multi = (qtype in ("list", "k_type"))
    diff = settings["difficulty_mix"]

    # Per-type guardrails and one-shot examples to steer formatting
    qtype_rules: dict[str, str] = {
        "multiple_choice": (
            "Stems ask for the best answer. Exactly one correct option. 4 options preferred. No 'All/None of the above'."
        ),
        "cloze": (
            "Fill-in-the-blank format. The STEM MUST contain at least one blank indicated by '_____'. "
            "Options are short tokens/phrases that could fill the blank. Exactly one correct index."
        ),
        "definition": (
            "Ask for the best definition of a given term OR the term that matches a given definition. "
            "Exactly one correct option. Avoid trivial wording differences."
        ),
        "list": (
            "List-style recognition. At least two options are correct (multiple correct indices). "
            "Avoid 'All/None of the above'."
        ),
        "true_false": (
            "Binary judgment. Options MUST be exactly ['True','False'] in that order. Exactly one correct index."
        ),
        "compare_contrast": (
            "Stems MUST explicitly compare two named entities using wording like 'X vs Y', 'between X and Y', or 'Which statement correctly contrasts X and Y?'. "
            "Include both entities by name in the STEM. Each option MUST be a two-clause comparison in the form 'X: … ; Y: …' (use ';', '—', or '–' as the divider). "
            "Exactly one correct option. Avoid 'All/None/Both/Neither' options."
        ),
        "assertion_reason": (
            "STEM must contain 'A: ...' (Assertion) and 'R: ...' (Reason). "
            "Options MUST be exactly these four patterns, in any order: "
            "1) Both A and R true, and R explains A; 2) Both A and R are true, but R does not explain A; "
            "3) A true, R false; 4) A false, R true. Exactly one correct."
        ),
        "sequence_ordering": (
            "Provide 4–6 labeled steps (e.g., A–D) in the stem. "
            "Options are different full-order permutations using '→' as divider. Exactly one correct."
        ),
        "matching_matrix": (
            "Provide two sets to match (A–D to 1–4). Each option is a complete mapping like 'A→2, B→4, C→1, D→3'. Exactly one correct."
        ),
        "k_type": (
            "Provide 4–6 standalone statements. Mark ALL that are true. Multiple correct indices (≥2). Avoid 'All/None of the above'."
        ),
        "best_next_step": (
            "Provide a 2–4 sentence scenario then ask: 'What is the best next step?' Exactly one correct option. Options must be plausible actions."
        ),
        "cause_effect": (
            "Given a study finding, pick the best causal interpretation: X→Y, Y→X, third variable, or correlation only. Exactly one correct."
        ),
        "negative": (
            "Include an explicit negative cue in the STEM (e.g., 'All of the following are TRUE EXCEPT:' or 'Which is NOT ...?'). "
            "Exactly one correct option, the exception. Avoid 'All/None of the above'."
        ),
    }

    qtype_examples: dict[str, str] = {
        "multiple_choice": (
            '{\n'
            '  "stem": "Which schedule of reinforcement typically produces the highest, most consistent response rate?",\n'
            '  "options": ["Fixed interval", "Variable interval", "Fixed ratio", "Variable ratio"],\n'
            '  "correct": [3],\n'
            '  "explanation": "Variable ratio schedules reinforce after an unpredictable number of responses, producing high, steady rates.",\n'
            '  "difficulty": "medium",\n'
            '  "tags": ["conditioning", "reinforcement"],\n'
            '  "source": "Section 3.2",\n'
            '  "qtype": "multiple_choice"\n'
            "}"
        ),
        "cloze": (
            '{\n'
            '  "stem": "The primary neurotransmitter at the neuromuscular junction is _____.",\n'
            '  "options": ["Dopamine", "Serotonin", "Acetylcholine", "GABA"],\n'
            '  "correct": [2],\n'
            '  "explanation": "Acetylcholine mediates synaptic transmission at the neuromuscular junction.",\n'
            '  "difficulty": "easy",\n'
            '  "tags": ["neuroscience", "neurotransmitters"],\n'
            '  "source": "Page 45",\n'
            '  "qtype": "cloze"\n'
            "}"
        ),
        "definition": (
            '{\n'
            '  "stem": "Which option best defines construct validity?",\n'
            '  "options": [\n'
            '    "The consistency of a measure across time",\n'
            '    "The extent to which a test measures the theoretical trait it intends to measure",\n'
            '    "Agreement between different raters",\n'
            '    "The representativeness of a sample"\n'
            '  ],\n'
            '  "correct": [1],\n'
            '  "explanation": "Construct validity reflects how well a test captures the intended theoretical construct.",\n'
            '  "difficulty": "medium",\n'
            '  "tags": ["measurement", "validity"],\n'
            '  "source": "Section 5.1",\n'
            '  "qtype": "definition"\n'
            "}"
        ),
        "list": (
            '{\n'
            '  "stem": "Which of the following are core components of working memory? (Select all that apply)",\n'
            '  "options": [\n'
            '    "Central executive",\n'
            '    "Phonological loop",\n'
            '    "Iconic store",\n'
            '    "Visuospatial sketchpad",\n'
            '    "Episodic buffer"\n'
            '  ],\n'
            '  "correct": [0,1,3,4],\n'
            '  "explanation": "Baddeley’s model includes central executive, phonological loop, visuospatial sketchpad, and episodic buffer.",\n'
            '  "difficulty": "medium",\n'
            '  "tags": ["memory"],\n'
            '  "source": "Section 7.3",\n'
            '  "qtype": "list"\n'
            "}"
        ),
        "true_false": (
            '{\n'
            '  "stem": "True or False: Classical conditioning involves learning the consequences of behavior.",\n'
            '  "options": ["True", "False"],\n'
            '  "correct": [1],\n'
            '  "explanation": "Learning consequences of behavior is operant conditioning; classical conditioning pairs stimuli.",\n'
            '  "difficulty": "easy",\n'
            '  "tags": ["conditioning"],\n'
            '  "source": "Page 112",\n'
            '  "qtype": "true_false"\n'
            "}"
        ),
        "compare_contrast": (
            '{\n'
            '  "stem": "Section 6.1 vs Section 6.2: which statement correctly contrasts them?",\n'
            '  "options": [\n'
            '    "6.1: assigns IP to members; 6.2: assigns IP to institution",\n'
            '    "6.1: institution holds IP for directed work; 6.2: member gives credit/disclaimer for incidental work",\n'
            '    "6.1: forbids all publication; 6.2: mandates publication",\n'
            '    "6.1: requires client consent for any publication; 6.2: never requires consent"\n'
            '  ],\n'
            '  "correct": [1],\n'
            '  "explanation": "6.1 covers institution IP for directed work; 6.2 covers member credit/disclaimer for incidental work.",\n'
            '  "difficulty": "medium",\n'
            '  "tags": ["Intellectual property"],\n'
            '  "source": "Section 6.1–6.2",\n'
            '  "qtype": "compare_contrast"\n'
            "}"
        ),
        "assertion_reason": (
            '{\n'
            '  "stem": "A: Section 6.1 assigns IP to the institution. R: Because directed work is owned by the employer.",\n'
            '  "options": [\n'
            '    "Both A and R are true, and R explains A",\n'
            '    "Both A and R are true, but R does not explain A",\n'
            '    "A is true, but R is false",\n'
            '    "A is false, but R is true"\n'
            '  ],\n'
            '  "correct": [0],\n'
            '  "explanation": "Directed work is typically employer-owned; R explains A.",\n'
            '  "difficulty": "medium", "tags": ["IP"], "source": "§6.1", "qtype": "assertion_reason"\n'
            "}"
        ),
        "sequence_ordering": (
            '{\n'
            '  "stem": "Put the disciplinary process in order (A–D): A: Notice → B: Investigation → C: Hearing → D: Sanction",\n'
            '  "options": ["A→B→C→D","A→C→B→D","B→A→C→D","A→B→D→C"],\n'
            '  "correct": [0],\n'
            '  "explanation": "Notice precedes investigation, then hearing, then sanction.",\n'
            '  "difficulty": "medium", "tags": ["process"], "source": "§X", "qtype":"sequence_ordering"\n'
            "}"
        ),
        "matching_matrix": (
            '{\n'
            '  "stem": "Match each subsection to its focus (A–D → 1–4).",\n'
            '  "options": [\n'
            '    "A→2, B→4, C→1, D→3", "A→3, B→4, C→1, D→2", "A→4, B→1, C→2, D→3", "A→2, B→1, C→4, D→3"\n'
            '  ],\n'
            '  "correct": [3],\n'
            '  "explanation": "Based on the subsection content.",\n'
            '  "difficulty": "medium", "tags":["mapping"], "source":"§X", "qtype":"matching_matrix"\n'
            "}"
        ),
        "k_type": (
            '{\n'
            '  "stem": "Which statements are true about Sections 6.1–6.2? (Select all that apply)",\n'
            '  "options": [\n'
            '    "6.1 covers directed work ownership.",\n'
            '    "6.2 requires disclaimers for incidental work.",\n'
            '    "6.1 mandates publication of all findings.",\n'
            '    "6.2 forbids giving credit."\n'
            '  ],\n'
            '  "correct": [0,1],\n'
            '  "explanation": "First two are accurate; the others are not.",\n'
            '  "difficulty":"medium","tags":["policy"],"source":"§6.1–6.2","qtype":"k_type"\n'
            "}"
        ),
        "best_next_step": (
            '{\n'
            '  "stem": "A client discloses X and Y during intake. The organization’s confidentiality policy applies, and the supervisor is unavailable. What is the best next step?",\n'
            '  "options": ["Proceed without documentation","Delay all action","Consult policy and document actions taken","Share details with peers informally"],\n'
            '  "correct": [2],\n'
            '  "explanation": "Consult the policy and document decisions.",\n'
            '  "difficulty":"medium","tags":["practice"],"source":"§Policy","qtype":"best_next_step"\n'
            "}"
        ),
        "cause_effect": (
            '{\n'
            '  "stem": "A cross-sectional study finds a strong association between A and B. What is the best supported inference?",\n'
            '  "options": ["A causes B","B causes A","A and B are caused by C","Association does not establish causality"],\n'
            '  "correct": [3],\n'
            '  "explanation": "Cross-sectional association cannot by itself establish direction.",\n'
            '  "difficulty":"easy","tags":["methods"],"source":"§Methods","qtype":"cause_effect"\n'
            "}"
        ),
        "negative": (
            '{\n'
            '  "stem": "All of the following are true of Section 6.1 EXCEPT:",\n'
            '  "options": ["Applies to directed work","Institution holds IP","Members always retain sole authorship","Covers employer-owned outputs"],\n'
            '  "correct": [2],\n'
            '  "explanation": "Members do not always retain sole authorship.",\n'
            '  "difficulty":"medium","tags":["policy"],"source":"§6.1","qtype":"negative"\n'
            "}"
        ),
    }

    style = """Write exam-quality MCQs.
- Use clear, unambiguous stems.
- Randomize plausible distractors.
- Prefer 4 options; more if needed. Avoid 'All/None of the above' unless pedagogically strong.
- Include a short explanation referencing the chapter content.
- Tag with section/topic; include difficulty from {diff}.
- {multi}

QTYPE-SPECIFIC RULES ({qtype}): {rules}

Follow the schema EXACTLY. Do not include any text outside JSON in your final output."""

    # Enforce cardinality based on setting
    multi_instruction = (
        "Every question MUST have MULTIPLE correct options (at least two). Do NOT create questions with a single correct answer."
        if allow_multi else
        "Every question MUST have EXACTLY ONE correct option."
    )
    # Add type preference hint
    type_hint = f"Focus on the '{qtype.replace('_',' ')}' question style when appropriate for MCQs."

    style = style.format(
        diff=", ".join(diff),
        multi=f"{multi_instruction}\n- {type_hint}",
        qtype=qtype,
        rules=qtype_rules.get(qtype, "Use standard MCQ formatting.")
    )

    # Schema hint for the model output (always include)
    schema_hint = """Return ONLY JSON in this exact schema:
list[MCQ] where MCQ = {
  "stem": str,
  "options": List[str],
  "correct": List[int],  # zero-based indices
  "explanation": str,
  "difficulty": "easy"|"medium"|"hard",
  "tags": List[str],
  "source": str,
    "qtype": "cloze"|"definition"|"list"|"true_false"|"multiple_choice"|"compare_contrast"|"assertion_reason"|"sequence_ordering"|"matching_matrix"|"k_type"|"best_next_step"|"cause_effect"|"negative"
}"""

    # Add custom instructions if provided
    custom_instructions = st.session_state.get("custom_mcq_instructions", "").strip()
    if custom_instructions:
        style += f"\n\nAdditional Instructions: {custom_instructions}"

    # Include a one-shot example for the selected type to steer formatting
    example_block = f"Example for type '{qtype}':\n{qtype_examples.get(qtype, qtype_examples['multiple_choice'])}"

    prompt = f"""
You are creating MCQs strictly from the chapter.

Chapter Title: {chapter_title}

Known question stems to avoid (normalized): {sorted(list(exclude_norm_stems))[:500]}

{style}

{schema_hint}

{example_block}

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
    exclude_norm_stems = _mcq_norm_stems(existing)
    prompt = _build_mcq_prompt(ch["title"], _chapter_excerpt_for_mcq(ch["text"]), s, exclude_norm_stems)


    with st.spinner("Generating MCQs…"):
        try:
            from google import genai as _genai_new
            client = _genai_new.Client(api_key=API_KEY)

            cfg = gx.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[MCQGen],
                temperature=float(s["temperature"]),
                system_instruction=f"Avoid generating MCQs whose stems are semantically similar to any of these normalized stems: {sorted(list(exclude_norm_stems))[:500]}"
            )

            # Debug header (no prompt or raw output printing)
            try:
                print("\n===== MCQ GEN DEBUG (generate_mcqs_for_active_chapter) =====")
            except Exception:
                pass

            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=cfg,
            )


            # Prefer parsed; fallback to JSON text
            mcqs = []
            raw_list = None
            try:
                if hasattr(resp, "parsed") and resp.parsed:
                    # Convert MCQGen items to plain dicts; enrich later
                    raw_list = [x.model_dump() for x in resp.parsed]
                    mcqs = list(raw_list)
                elif resp.text:
                    raw_list = json.loads(resp.text)
                    mcqs = list(raw_list)
                else:
                    st.error("No response received from API")
                    return
            except json.JSONDecodeError:
                st.warning("Received incomplete response. Trying to extract valid MCQs...")
                # Try to extract valid cards from partial JSON
                try:
                    # Look for complete card objects in the response
                    text = resp.text or ""
                    card_pattern = r'\{[^{}]*"q":[^{}]*"a":[^{}]*\}'
                    matches = re.findall(card_pattern, text)
                    partial_mcqs = []
                    for match in matches:
                        try:
                            partial_mcqs.append(json.loads(match))
                        except:
                            continue
                    if partial_mcqs:
                        mcqs = partial_mcqs
                        st.info(f"Recovered {len(mcqs)} MCQs from partial response")
                    else:
                        st.error("Could not recover any MCQs from response. Try reducing the number of questions.")
                        return
                except Exception as e2:
                    st.error(f"Could not parse response: {e2}")
                    return
            except Exception as e:
                st.error(f"Unexpected error parsing response: {e}")
                return

            # Log pre-filter raw list size
            try:
                from collections import Counter  # local import for minimal scope
                print(f"Raw MCQ count (pre-filter): {len(raw_list) if isinstance(raw_list, list) else 'N/A'}")
            except Exception:
                pass

            # Cleanup, type enforcement, dedupe, and save
            mcqs = [m for m in mcqs if m.get("stem") and m.get("options") and isinstance(m.get("correct"), list)]
            audit: dict = {}
            mcqs = _filter_mcqs_by_qtype(mcqs, st.session_state.mcq_settings.get("qtype", "multiple_choice"), audit=audit, salvage_non_contrast=True)
            mcqs = _assign_ids_and_hashes(mcqs)
            new_only = [m for m in mcqs if m["hash"] not in exclude_hashes]

            # Debug: final items first, then filtered items
            try:
                print("===== MCQ GEN DEBUG: FINAL NEW ITEMS (after dedupe) =====")
                print(json.dumps(new_only, ensure_ascii=False, indent=2))
                print(f"Final new items count: {len(new_only)}\n")
                print("===== MCQ GEN DEBUG: ITEMS KEPT AFTER QTYPE FILTER =====")
                print(json.dumps(mcqs, ensure_ascii=False, indent=2))
                print(f"Kept after qtype filter: {len(mcqs)} | Downgraded: {audit.get('downgraded', 0)}")
                # Rejection stats
                try:
                    from collections import Counter
                    total_in = audit.get("input", 0)
                    kept = audit.get("kept", 0)
                    reasons = Counter(audit.get("reasons", []))
                    print(f"QTYPE filter: kept {kept}/{total_in}")
                    print("Top rejection reasons:", reasons.most_common(10))
                except Exception:
                    pass
            except Exception:
                pass

            st.session_state.mcq_by_chapter.setdefault(st.session_state.active_chapter_index, []).extend(new_only)
            st.success(f"Added {len(new_only)} new MCQs (filtered {len(mcqs)-len(new_only)} duplicates). Kept {audit.get('kept', len(new_only))}, downgraded {audit.get('downgraded', 0)}.")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during MCQ generation: {e}")
            st.info("Try reducing the number of questions or adjusting the settings.")

def mcqs_from_flashcards(cards: list[dict], k: int=10) -> list[dict]:
    """Generate MCQs from existing flashcards"""
    
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
            "qtype": "multiple_choice",
        })
    
    
    return _assign_ids_and_hashes(out)

# -----------------------------------------------------------------------------
# 6. MCQ WIZARD NAVIGATION FUNCTIONS
# -----------------------------------------------------------------------------

def go_to_mcq_step(step: int):
    """Navigate to a specific MCQ wizard step"""
    st.session_state.mcq["step"] = max(0, min(2, step))

def next_mcq_step():
    """Navigate to next MCQ wizard step with validation"""
    step = st.session_state.mcq["step"]
    if step == 0 and not st.session_state.mcq["generated"]["items"]:
        st.session_state.mcq["warning"] = "Generate at least one MCQ to continue."
        return
    if step == 1 and not (st.session_state.mcq["test"]["submitted"] or st.session_state.mcq["test"]["responses"]):
        st.session_state.mcq["warning"] = "Submit the test or choose to continue without scoring."
        return
    st.session_state.mcq["warning"] = None
    go_to_mcq_step(step + 1)

def prev_mcq_step():
    """Navigate to previous MCQ wizard step"""
    step = st.session_state.mcq["step"]
    if step == 1 and st.session_state.mcq["test"]["responses"]:
        # Warn about losing test progress
        if not st.session_state.get("_mcq_back_confirmed", False):
            st.session_state.mcq["warning"] = "Going back will clear your current test answers."
            if st.button("Confirm: Clear answers and go back", key="confirm_back"):
                st.session_state["_mcq_back_confirmed"] = True
                st.session_state.mcq["test"] = {
                    "order": [], "responses": {}, "score": None, "submitted": False,
                    "current_q": 0, "show_explanations": False
                }
                go_to_mcq_step(step - 1)
            return
    st.session_state.mcq["warning"] = None
    st.session_state["_mcq_back_confirmed"] = False
    go_to_mcq_step(step - 1)

# -----------------------------------------------------------------------------
# 7. MCQ WIZARD STEP FUNCTIONS
# -----------------------------------------------------------------------------

def render_mcq_generate():
    """Render the Generate step of MCQ wizard"""
    active_chapter = st.session_state.chapters[st.session_state.active_chapter_index]
    
    with st.form("mcq_generate_form"):
        st.markdown("### Generation Settings")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            n_qs = st.number_input("Number of MCQs", 1, 30, st.session_state.mcq_settings["n_qs"])
        with col2:
            # Single MCQ type selector determines cardinality
            mcq_type_options = [
                "multiple_choice",  # Standard MCQ (single correct)
                "list",             # Multi-correct list-style MCQ
                "definition",
                "true_false",
                "compare_contrast",
                "cloze",
                "assertion_reason",
                "sequence_ordering",
                "matching_matrix",
                "k_type",
                "best_next_step",
                "cause_effect",
                "negative",
            ]
            mcq_type_labels = {
                "multiple_choice": "Standard MCQ",
                "list": "List (multiple correct)",
                "definition": "Definition",
                "true_false": "True/False",
                "compare_contrast": "Compare/Contrast",
                "cloze": "Cloze (fill-in-blank)",
                "assertion_reason": "Assertion–Reason",
                "sequence_ordering": "Sequence Ordering",
                "matching_matrix": "Matching (Matrix)",
                "k_type": "Multiple True/False (K-type)",
                "best_next_step": "Best Next Step (Scenario)",
                "cause_effect": "Cause vs. Correlation",
                "negative": "EXCEPT / NOT",
            }
            current_qtype = st.session_state.mcq_settings.get("qtype", "multiple_choice")
            qtype_sel = st.selectbox(
                "MCQ type:",
                mcq_type_options,
                index=mcq_type_options.index(current_qtype) if current_qtype in mcq_type_options else 0,
                format_func=lambda x: mcq_type_labels.get(x, x.replace('_',' ').title())
            )
        with col3:
            temperature = st.slider("Creativity (temp)", 0.0, 1.0, st.session_state.mcq_settings["temperature"], 0.05)
        with col4:
            choices = ["easy","medium","hard"]
            current = st.session_state.mcq_settings["difficulty_mix"]
            difficulty_mix = st.multiselect("Difficulty mix", choices, default=current)

        col_gen, col_from_flash = st.columns([1, 1])
        
        with col_gen:
            generate_clicked = st.form_submit_button("Generate MCQs from Chapter", type="primary")
            
        with col_from_flash:
            flashcards = st.session_state.flashcards_by_chapter.get(st.session_state.active_chapter_index, [])
            convert_clicked = st.form_submit_button("Convert from Flashcards", disabled=not flashcards)

        # Handle form submissions
        if generate_clicked:
            if not difficulty_mix:
                st.error("Please select at least one difficulty level!")
            else:
                # Update settings
                st.session_state.mcq_settings.update({
                    "n_qs": n_qs,
                    "qtype": qtype_sel,
                    "temperature": temperature,
                    "difficulty_mix": difficulty_mix
                })
                
                # Set busy state and generate
                st.session_state.mcq["busy"] = True
                st.rerun()
                
        if convert_clicked:
            if flashcards:
                converted_mcqs = mcqs_from_flashcards(flashcards, min(10, len(flashcards)))
                st.session_state.mcq["generated"]["items"] = converted_mcqs
                st.session_state.mcq["generated"]["source_id"] = f"flashcards_ch_{st.session_state.active_chapter_index}"
                st.session_state.mcq["generated"]["timestamp"] = datetime.now(timezone.utc).isoformat()
                st.success(f"Converted {len(converted_mcqs)} flashcards to MCQs!")
                st.rerun()

    # Handle busy generation
    if st.session_state.mcq["busy"]:
        with st.spinner("Generating MCQs..."):
            try:
                generate_mcqs_wizard()
            except Exception as e:
                st.error(f"Error during MCQ generation: {e}")
            finally:
                st.session_state.mcq["busy"] = False
                st.rerun()

    # Show current MCQs
    items = st.session_state.mcq["generated"]["items"]
    if items:
        st.markdown(f"**{len(items)} MCQs generated**")
        
        # Preview of generated items
        with st.expander("Preview Generated MCQs", expanded=len(items) <= 3):
            for i, q in enumerate(items[:5]):  # Show first 5
                st.markdown(f"**Q{i+1}:** {q['stem']}")
                for j, opt in enumerate(q['options']):
                    marker = "✅" if j in q['correct'] else "○"
                    st.markdown(f"  {marker} {chr(65+j)}. {opt}")
            if len(items) > 5:
                st.markdown(f"... and {len(items) - 5} more questions")

def generate_mcqs_wizard():
    """Generate MCQs for the wizard (modified version of existing function)"""
    ch = st.session_state.chapters[st.session_state.active_chapter_index]
    s = st.session_state.mcq_settings
    
    # Use existing MCQs for deduplication
    existing = st.session_state.mcq["generated"]["items"]
    exclude_hashes = _mcq_dedupe_hashes(existing)
    exclude_norm_stems = _mcq_norm_stems(existing)
    prompt = _build_mcq_prompt(ch["title"], _chapter_excerpt_for_mcq(ch["text"]), s, exclude_norm_stems)

    from google import genai as _genai_new
    client = _genai_new.Client(api_key=API_KEY)

    cfg = gx.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[MCQGen],
        temperature=float(s["temperature"]),
        system_instruction=f"Avoid generating MCQs whose stems are semantically similar to any of these normalized stems: {sorted(list(exclude_norm_stems))[:500]}"
    )

    # Debug header (no prompt or raw output printing)
    try:
        print("\n===== MCQ GEN DEBUG (generate_mcqs_wizard) =====")
    except Exception:
        pass

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=cfg,
    )

    # Parse response
    mcqs = []
    raw_list = None
    try:
        if hasattr(resp, "parsed") and resp.parsed:
            # Convert MCQGen items to plain dicts; enrich later
            raw_list = [x.model_dump() for x in resp.parsed]
            mcqs = list(raw_list)
        elif resp.text:
            raw_list = json.loads(resp.text)
            mcqs = list(raw_list)
        else:
            raise Exception("No response received from API")
    except json.JSONDecodeError:
        st.warning("Received incomplete response. Trying to extract valid MCQs…")
        text = resp.text or ""
        try:
            raw_list = json.loads(text[text.find("["):text.rfind("]")+1])
            mcqs = list(raw_list)
        except Exception:
            raise Exception("Could not parse MCQs from the response.")

    # (No parsed-before-filtering print per user request)

    # Log pre-filter raw list size
    try:
        from collections import Counter  # local import for minimal scope
        print(f"Raw MCQ count (pre-filter): {len(raw_list) if isinstance(raw_list, list) else 'N/A'}")
    except Exception:
        pass

    # Cleanup, type enforcement, dedupe, and save
    mcqs = [m for m in mcqs if m.get("stem") and m.get("options") and isinstance(m.get("correct"), list)]
    audit: dict = {}
    mcqs = _filter_mcqs_by_qtype(mcqs, st.session_state.mcq_settings.get("qtype", "multiple_choice"), audit=audit, salvage_non_contrast=True)
    mcqs = _assign_ids_and_hashes(mcqs)
    new_only = [m for m in mcqs if m["hash"] not in exclude_hashes]

    # Debug: final items first, then filtered items
    try:
        print("===== MCQ GEN DEBUG: FINAL NEW ITEMS (after dedupe) =====")
        print(json.dumps(new_only, ensure_ascii=False, indent=2))
        print(f"Final new items count: {len(new_only)}\n")
        print("===== MCQ GEN DEBUG: ITEMS KEPT AFTER QTYPE FILTER =====")
        print(json.dumps(mcqs, ensure_ascii=False, indent=2))
        print(f"Kept after qtype filter: {len(mcqs)} | Downgraded: {audit.get('downgraded', 0)}")
        # Rejection stats
        try:
            from collections import Counter
            total_in = audit.get("input", 0)
            kept = audit.get("kept", 0)
            reasons = Counter(audit.get("reasons", []))
            print(f"QTYPE filter: kept {kept}/{total_in}")
            print("Top rejection reasons:", reasons.most_common(10))
        except Exception:
            pass
    except Exception:
        pass

    # Store in wizard state
    st.session_state.mcq["generated"]["items"] = new_only
    st.session_state.mcq["generated"]["source_id"] = f"chapter_{st.session_state.active_chapter_index}"
    st.session_state.mcq["generated"]["params"] = dict(s)
    st.session_state.mcq["generated"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    st.success(f"Generated {len(new_only)} new MCQs! (kept {audit.get('kept', len(new_only))}, downgraded {audit.get('downgraded', 0)})")

def render_mcq_test():
    """Render the Test step of MCQ wizard"""
    items = st.session_state.mcq["generated"]["items"]
    test_state = st.session_state.mcq["test"]
    
    if not items:
        st.warning("No MCQs available for testing. Please generate some first.")
        return

    # Initialize test session if needed
    if not test_state["order"]:
        # Keep questions in original display order, but shuffle options within each question
        q_order = list(range(len(items)))  # Keep original order: [0, 1, 2, 3, ...]
        # No longer shuffling: random.shuffle(q_order)
        
        shuffled_items = []
        for qi in q_order:
            q = dict(items[qi])
            q["original_index"] = qi
            
            # Still shuffle options within each question for quiz variety
            opt_perm = list(range(len(q["options"])))
            random.shuffle(opt_perm)
            q["options"] = [q["options"][i] for i in opt_perm]
            
            # Map correct indices through permutation
            corr = set(q["correct"])
            q["correct"] = [i for i, old in enumerate(opt_perm) if old in corr]
            q["_opt_perm"] = opt_perm
            shuffled_items.append(q)
        
        test_state["order"] = q_order
        test_state["shuffled_items"] = shuffled_items
        test_state["responses"] = {}
        test_state["current_q"] = 0
        test_state["submitted"] = False
        test_state["score"] = None

    # Ensure additional state exists
    if "checked" not in test_state:
        test_state["checked"] = {}

    # Show current question
    current_idx = test_state["current_q"]
    shuffled_items = test_state["shuffled_items"]
    total_questions = len(shuffled_items)
    
    if current_idx < total_questions:
        q = shuffled_items[current_idx]
        
        st.markdown(f"**Question {current_idx + 1} of {total_questions}**")
        st.progress((current_idx + 1) / total_questions)
        
        st.markdown(f"**{q['stem']}**")
    # UI uses per-question cardinality: radio for single-correct, checkboxes for multi-correct
    multi = len(q["correct"]) > 1

    # Has this question been checked already?
    already_checked = bool(test_state["checked"].get(current_idx))

    with st.form(f"question_form_{current_idx}"):
        user_response = None

        if multi:
            st.info("Multiple answers may be correct. Select all that apply:")
            selections = []
            for i, opt in enumerate(q["options"]):
                if st.checkbox(
                    f"{chr(65+i)}. {opt}",
                    key=f"opt_{current_idx}_{i}",
                    disabled=already_checked,
                ):
                    selections.append(i)
            user_response = sorted(selections) if selections else None
        else:
            options_with_letters = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(q["options"])]
            selected = st.radio(
                "Choose one:",
                options_with_letters,
                index=None,
                key=f"radio_{current_idx}",
                disabled=already_checked,
            )
            if selected is not None:
                selected_index = options_with_letters.index(selected)
                user_response = [selected_index]

        # Feedback section (only after checking)
        if already_checked:
            # Use the saved response if available
            saved_resp = test_state["responses"].get(current_idx)
            is_correct = saved_resp is not None and set(saved_resp) == set(q["correct"])
            if is_correct:
                st.success("Correct!")
            else:
                st.error("Incorrect.")
            correct_letters = ", ".join(chr(65+i) for i in q["correct"]) if q.get("correct") else ""
            if correct_letters:
                st.caption(f"Correct answer(s): {correct_letters}")
            if q.get("explanation"):
                st.info(q["explanation"])

        # Primary action button: Check Answer -> Next Question / Finish Quiz
        next_is_finish = current_idx >= (total_questions - 1)
        action_label = ("Finish Quiz & View Results" if next_is_finish else "Next Question") if already_checked else "Check Answer"

        if st.form_submit_button(action_label, type="primary"):
            if not already_checked:
                # First click: check answer and show feedback
                if user_response is None:
                    st.warning("Please select an answer before checking.")
                else:
                    test_state["responses"][current_idx] = user_response
                    test_state["checked"][current_idx] = True
                    st.rerun()
            else:
                # Second click: move to next or finish
                if next_is_finish:
                    submit_mcq_test()
                    # Auto-navigate to the detailed results view
                    st.session_state.mcq["step"] = 2
                    st.rerun()
                else:
                    test_state["current_q"] = current_idx + 1
                    st.rerun()
    
    # Show completion status
    if test_state["submitted"]:
        st.markdown("---")
        st.subheader("Test Complete! 🎊")
        
        score = test_state["score"]
        total = len(shuffled_items)
        percentage = round(score * 100 / total) if total > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{score}/{total}")
        with col2:
            st.metric("Percentage", f"{percentage}%")
        with col3:
            st.metric("Questions", total)
        
        st.progress(score / total if total > 0 else 0)
        
        # Performance feedback
        if percentage >= 90:
            st.success("Excellent work! You've mastered this material! 🌟")
        elif percentage >= 70:
            st.success("Good job! You have a solid understanding. 👍")
        elif percentage >= 50:
            st.warning("Not bad, but there's room for improvement. Keep studying! 📚")
        else:
            st.error("You might want to review this chapter more thoroughly. 🔄")

def submit_mcq_test():
    """Calculate score and mark test as submitted"""
    test_state = st.session_state.mcq["test"]
    shuffled_items = test_state["shuffled_items"]
    responses = test_state["responses"]
    
    score = 0
    for i, q in enumerate(shuffled_items):
        user_answer = responses.get(i)
        correct_answer = set(q["correct"])
        if user_answer and set(user_answer) == correct_answer:
            score += 1
    
    test_state["score"] = score
    test_state["submitted"] = True

def render_mcq_review():
    """Render the Review step of MCQ wizard"""
    items = st.session_state.mcq["generated"]["items"]
    test_state = st.session_state.mcq["test"]
    
    if not items:
        st.warning("No MCQs available for review.")
        return
    
    # Test Results Summary (if test was taken)
    if test_state["submitted"]:
        score = test_state["score"]
        total = len(test_state["shuffled_items"])
        percentage = round(score * 100 / total) if total > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Score", f"{score}/{total}")
        with col2:
            st.metric("Percentage", f"{percentage}%")
        with col3:
            st.metric("Total Questions", total)

    # Detailed Review
    st.markdown("#### Detailed Question Review")
    
    for i, q in enumerate(items):
        # Determine if user got this question correct (for emoji in title)
        emoji = ""
        if test_state["submitted"] and test_state.get("shuffled_items"):
            shuffled_items = test_state["shuffled_items"]
            for si, sq in enumerate(shuffled_items):
                if sq.get("original_index") == i:
                    user_response = test_state["responses"].get(si)
                    if user_response is not None:
                        # Map user response back through option permutation to original indices
                        opt_perm = sq.get("_opt_perm", list(range(len(q["options"]))))
                        original_user_response = [opt_perm[idx] for idx in user_response]
                        correct_answer = set(q["correct"])
                        if set(original_user_response) == correct_answer:
                            emoji = "✅ "
                        else:
                            emoji = "❌ "
                    break
        
        # Show full question text in the expander title with emoji (no truncation)
        with st.expander(f"{emoji}Q{i+1}: {q['stem']}"):
            st.markdown(f"**Question:** {q['stem']}")
            
            # Show options; if user responded, mark their selected options with ✅ if correct, ❌ if incorrect
            st.markdown("**Options:**")

            # Determine user's original (unshuffled) selected option indices for this question, if available
            selected_original_response = None
            if test_state["submitted"] and test_state.get("shuffled_items"):
                shuffled_items = test_state["shuffled_items"]
                for si, sq in enumerate(shuffled_items):
                    if sq.get("original_index") == i:
                        user_response = test_state["responses"].get(si)
                        if user_response is not None:
                            opt_perm = sq.get("_opt_perm", list(range(len(q["options"]))))
                            selected_original_response = {opt_perm[idx] for idx in user_response}
                        break

            # Render options with appropriate markers
            if selected_original_response is not None:
                for j, opt in enumerate(q['options']):
                    # Determine the marker for this option
                    if j in q['correct']:
                        # This is a correct option - always show ✅
                        marker = "✅"
                    elif j in selected_original_response:
                        # User selected this incorrect option - show ❌
                        marker = "❌"
                    else:
                        # Neither correct nor selected by user - show ○
                        marker = "○"
                    
                    # Bold the correct options
                    if j in q['correct']:
                        st.markdown(f"{marker} **{chr(65+j)}. {opt}**")
                    else:
                        st.markdown(f"{marker} {chr(65+j)}. {opt}")
            else:
                # Fallback: highlight correct options when no user selection available
                for j, opt in enumerate(q['options']):
                    marker = "✅" if j in q['correct'] else "○"
                    # Bold the correct options
                    if j in q['correct']:
                        st.markdown(f"{marker} **{chr(65+j)}. {opt}**")
                    else:
                        st.markdown(f"{marker} {chr(65+j)}. {opt}")
            
            if q.get('explanation'):
                st.markdown(f"**Explanation:** {q['explanation']}")
            
            # Show user response if test was taken
            if test_state["submitted"] and test_state.get("shuffled_items"):
                shuffled_items = test_state["shuffled_items"]
                # Find this question in shuffled items
                for si, sq in enumerate(shuffled_items):
                    if sq.get("original_index") == i:
                        user_response = test_state["responses"].get(si)
                        if user_response is not None:
                            # Map user response back through option permutation
                            opt_perm = sq.get("_opt_perm", list(range(len(q["options"]))))
                            original_user_response = [opt_perm[idx] for idx in user_response]
                            
                            correct_letters = [chr(65+idx) for idx in q['correct']]
                            st.markdown(f"**Correct Answer:** **{', '.join(correct_letters)}**")
                        break
            
            # Metadata
            qtype_display = (q.get('qtype') or 'multiple_choice').replace('_',' ').title()
            st.caption(f"**Type:** {qtype_display} | **Difficulty:** {q.get('difficulty', 'medium').title()} | **Source:** {q.get('source', 'Unknown')}")

    # Navigation: Back button placed immediately after the last question
    if st.button("◀ Back to MCQs", key="back_from_results"):
        st.session_state.mcq["step"] = 0
        st.rerun()

# --- UI and State Management ---
def reset_mcq_test_state():
    """Resets the MCQ test state for a new quiz."""
    st.session_state.mcq["test"] = {
    "order": [], "responses": {}, "score": None, "submitted": False,
    "current_q": 0, "show_explanations": False, "shuffled_items": [],
    "checked": {}
    }

def reset_chat_session():
    """Resets the conversation tree for a new chat."""
    st.session_state.conv_tree = ConvTree()
    st.session_state.pending_user_node_id = None
    st.session_state.editing_msg_id = None
    st.session_state.editing_content = ""

def reset_mcq_wizard():
    """Resets the MCQ wizard state."""
    st.session_state.mcq = {
        "step": 0,
    "generated": {"items": [], "seed": None, "source_id": None, "params": {}, "timestamp": None},
    "test": {"order": [], "responses": {}, "score": None, "submitted": False, "current_q": 0, "show_explanations": False, "shuffled_items": [], "checked": {}},
        "review": {"filters": {}, "edits": {}, "export_format": "csv"},
        "busy": False,
        "warning": None
    }

def reset_all_for_new_chapter():
    """Resets both chat and MCQ wizard when changing chapters."""
    reset_chat_session()
    reset_mcq_wizard()

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
if "conversational_style" not in st.session_state: st.session_state.conversational_style = "Concise"
if "document_reliance" not in st.session_state: st.session_state.document_reliance = 3

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
    "temperature": 0.85
    }

# --- MCQ Session State ---
if "mcq_by_chapter" not in st.session_state:
    st.session_state.mcq_by_chapter = {}  # {chapter_index: [mcq, ...]}

if "mcq_settings" not in st.session_state:
    st.session_state.mcq_settings = {
        "n_qs": 10,
        "difficulty_mix": ["easy","medium","hard"],  # chosen set
    "temperature": 0.85,
        "show_expl_immediately": True,
        "timer_seconds": 0,  # 0 = untimed
    "qtype": "multiple_choice",
    }

if "mcq_sessions" not in st.session_state:
    st.session_state.mcq_sessions = {}  # keyed by chapter index

# New MCQ Wizard State
if "mcq" not in st.session_state:
    st.session_state.mcq = {
        "step": 0,  # 0=Generate, 1=Test, 2=Review
        "generated": {           # data produced in Generate
            "items": [],         # list[dict]: question, options, answer, meta
            "seed": None,        # to reproduce a generation
            "source_id": None,   # chapter / doc / selection identifier
            "params": {},        # difficulty, n_items, etc.
            "timestamp": None,
        },
        "test": {                # data collected in Test
            "order": [],         # index order after shuffle
            "responses": {},     # q_idx -> selected_option / free text
            "score": None,       # computed after submit
            "submitted": False,
            "current_q": 0,      # current question pointer
            "show_explanations": False
        },
        "review": {              # preferences in Review
            "filters": {},       # correct/incorrect tags, difficulty
            "edits": {},         # q_idx -> edited payload
            "export_format": "csv"
        },
        "busy": False,           # lock while long ops run
        "warning": None          # transient message to surface at top
    }

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
        # Clear stale auto text when a new PDF is uploaded
        new_name = getattr(uploaded_file, "name", None)
        prev_name = st.session_state.get("last_pdf_name")
        if new_name and new_name != prev_name:
            st.session_state.pop("auto_chapter_text", None)
        st.session_state["last_pdf_name"] = new_name
        
        st.header("Step 2: Define Chapters")
        
        # Option 1 + Download side-by-side (wider buttons; download shown disabled until available)
        col_opt1, col_dl, _ = st.columns([2, 2, 6])
        with col_opt1:
            auto_clicked_top = st.button("Automatically Extract Chapter Definitions", use_container_width=True)
        with col_dl:
            current_text = (
                (st.session_state.get("chapter_definitions_text", "").strip())
                or (st.session_state.get("auto_chapter_text", "").strip())
            )
            st.download_button(
                label="Download .txt Chapter Parsing File",
                data=(current_text.encode("utf-8") if current_text else b""),
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_chapters.txt",
                mime="text/plain",
                key="download_chapters_txt_top",
                disabled=not bool(current_text),
                use_container_width=True,
            )
        
        # Option to upload chapter definitions from a text file
        uploaded_definitions_file = st.file_uploader(
            "If available, upload a .txt file for chapter parsing",
            type="txt",
        )
        
        # Handle Option 1 click (auto extraction)
        if 'auto_clicked_top' in locals() and auto_clicked_top:
            with st.spinner("Extracting chapters with Gemini..."):
                try:
                    extracted = extract_chapters_with_gemini(uploaded_file, client)
                    # Build lines from validated/sorted extracted list
                    lines = [f"{ch['title']}, {ch['start_page']}-{ch['end_page']}" for ch in extracted]
                    text_out = "\n".join(lines)
                    if not text_out:
                        st.warning("No chapters were extracted. You can still enter them manually.")
                    else:
                        st.session_state.auto_chapter_text = text_out
                        st.success(f"Extracted {len(lines)} chapter(s). You can edit them, then click 'Process Document and Chapters' or download the .txt.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Automatic extraction failed: {e}")
        
        with st.form("chapter_form"):
            # Pre-populate with uploaded file content if available (and any auto extracted text)
            default_text = st.session_state.get("auto_chapter_text", "")
            if uploaded_definitions_file is not None and not default_text:
                try:
                    default_text = uploaded_definitions_file.read().decode("utf-8")
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")
            
            chapter_definitions_text = st.text_area(
                "Detected Chapter Parsing (Manual Input Available):",
                value=default_text,
                height=150,
                placeholder="For example:\n\nChapter 1: The Beginning, 1-20\nChapter 2: The Middle, 21-55",
                key="chapter_definitions_text",
            )

            col_process = st.columns([1])[0]
            with col_process:
                submitted = st.form_submit_button("Process Document and Chapters", type="primary")

        if submitted and st.session_state.get("chapter_definitions_text"):
            with st.spinner("Processing document... This may take a moment."):
                parsed_chapters = parse_chapter_definitions(st.session_state["chapter_definitions_text"]) 
                if parsed_chapters:
                    # Validate and sort manual input against total pages
                    as_ranges = []
                    for ch in parsed_chapters:
                        pages = ch["pages"]
                        as_ranges.append({
                            "title": ch["title"],
                            "start_page": min(pages),
                            "end_page": max(pages),
                        })
                    total_pages = None
                    try:
                        with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as _doc:
                            total_pages = len(_doc)
                    except Exception:
                        pass
                    norm, warns = validate_and_sort_chapter_ranges(as_ranges, total_pages=total_pages)
                    for w in warns:
                        st.warning(w)
                    # Rebuild parsed_chapters from validated ranges
                    parsed_chapters = [
                        {"title": ch["title"], "pages": list(range(ch["start_page"], ch["end_page"] + 1))}
                        for ch in norm
                    ]

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
                    reset_chat_session()  # Initialize the chat for the first chapter
                    # Clear stale auto text after successful processing
                    st.session_state.pop("auto_chapter_text", None)
                    st.rerun()
                else:
                    st.error("No valid chapter definitions found. Please check the format.")

        

# STATE 2: CHAT VIEW
else:
    # --- Sidebar for Chapter Selection and Settings ---
    with st.sidebar:

        # Document info at the top
        if st.session_state.uploaded_file_info:
            st.info(f"**Document:**\n{st.session_state.uploaded_file_info['name']}")

        st.header("Session Setup")

        chapter_titles = [ch['title'] for ch in st.session_state.chapters]
        
        # This widget's state is controlled by st.session_state.active_chapter_index
        selected_chapter_index = st.selectbox(
            "Select Chapter to Focus On",
            options=range(len(chapter_titles)),
            format_func=lambda i: chapter_titles[i],
            key="active_chapter_index",
            on_change=reset_all_for_new_chapter # Reset both chat and MCQ when chapter changes
        )
        
        # Display info about the selected chapter
        if st.session_state.chapters:
            selected_chapter = st.session_state.chapters[selected_chapter_index]
            
            # Calculate chapter info
            page_range = selected_chapter['pages']
            toc_start = min(page_range)
            toc_end = max(page_range)
            num_pages = len(page_range)
            
            # Check for text truncation
            chapter_text = selected_chapter.get('text', '')
            text_length = len(chapter_text)
            was_truncated = text_length >= CHAPTER_CHAR_LIMIT
            
            # Display chapter information
            st.caption(f"""
            • PDF Page Range: {toc_start}-{toc_end} ({num_pages} pages)
            \n• Text Length: {text_length:,} characters{' (truncated)' if was_truncated else ''}
            """)
        
        st.markdown("---")

        # Conversation Style Selection
        st.subheader("Conversation Style")
        selected_style = st.selectbox(
            "How the AI revision assistant interacts:",
            options=list(CONVERSATIONAL_STYLES.keys()),
            key="conversational_style",
            help="Different styles will change how the AI responds to you in the Conversation tab"
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
        
        # st.markdown("---")

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
        st.markdown("### Generation Settings")
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
                "cloze": "Fill-in-the-blank _____ format",
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
            s["temperature"] = st.slider(
                "Temperature (How creative should the AI be?)",
                0.0,
                2.0,
                s["temperature"],
                0.05,
                key="flash_temp_slider",
            )
            difficulty_options = ["mixed","easy","medium","hard"]
            current = s["difficulty"]
            s["difficulty"] = st.selectbox(
                "Difficulty:", 
                difficulty_options,
                index=difficulty_options.index(current),
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
        # If in Quiz or Results mode, render that view exclusively for the MCQs tab
        if st.session_state.mcq.get("step") == 1:
            st.markdown("### 📝 Quiz Mode")
            render_mcq_test()
            # Navigation back to MCQs list/settings
            col1, col2 = st.columns(2)
            with col1:
                if st.button("◀ Back to MCQs", key="back_to_mcqs_fullpage"):
                    st.session_state.mcq["step"] = 0
                    st.rerun()
            # Stop here to avoid rendering the generation UI below
            st.stop()

        elif st.session_state.mcq.get("step") == 2:
            st.markdown("### 📊 Quiz Results")
            render_mcq_review()
            # Stop here to avoid rendering the generation UI below
            st.stop()

        # Generation settings in two columns at the top
        st.markdown("### Generation Settings")
        col1, col2 = st.columns([1,1])
        s = st.session_state.mcq_settings
        
        with col1:
            s["n_qs"] = st.slider("How many MCQs?", 1, 30, s["n_qs"])
            
            # Single MCQ type selector (also determines single vs multiple correct)
            mcq_type_options = [
                "multiple_choice",  # Standard MCQ (single correct)
                "list",             # Multi-correct list-style MCQ
                "definition",
                "true_false",
                "compare_contrast",
                "cloze",
                "assertion_reason",
                "sequence_ordering",
                "matching_matrix",
                "k_type",
                "best_next_step",
                "cause_effect",
                "negative",
            ]
            mcq_type_labels = {
                "multiple_choice": "Standard MCQ",
                "list": "List (multiple correct)",
                "definition": "Definition",
                "true_false": "True/False",
                "compare_contrast": "Compare/Contrast",
                "cloze": "Cloze (fill-in-blank)",
                "assertion_reason": "Assertion–Reason",
                "sequence_ordering": "Sequence Ordering",
                "matching_matrix": "Matching (Matrix)",
                "k_type": "Multiple True/False (K-type)",
                "best_next_step": "Best Next Step (Scenario)",
                "cause_effect": "Cause vs. Correlation",
                "negative": "EXCEPT / NOT",
            }
            current_qtype = s.get("qtype", "multiple_choice")
            s["qtype"] = st.selectbox(
                "MCQ type:",
                mcq_type_options,
                index=mcq_type_options.index(current_qtype) if current_qtype in mcq_type_options else 0,
                format_func=lambda x: mcq_type_labels.get(x, x.replace('_',' ').title())
            )

        with col2:
            s["temperature"] = st.slider(
                "Temperature (How creative should the AI be?)",
                0.0,
                2.0,
                s["temperature"],
                0.05,
                key="mcq_temp_slider",
            )
            
            # Difficulty mix selection
            difficulty_choices = ["easy","medium","hard"]
            current_mix = s["difficulty_mix"] if s["difficulty_mix"] else ["medium"]
            s["difficulty_mix"] = st.multiselect(
                "Difficulty mix:",
                difficulty_choices,
                default=current_mix
            )

        # Previously generated MCQs upload section
        imported = st.file_uploader("**(Optional)** Upload JSON file to import existing MCQs", type="json", key="mcq_uploader")
        if imported:
            try:
                raw = json.load(imported)
                # Support both {"items": [...]} and plain list
                items = raw.get("items") if isinstance(raw, dict) else raw
                if not isinstance(items, list):
                    raise ValueError("JSON must be a list of MCQs or an object with an 'items' list")

                def _coerce_int_list(v):
                    if isinstance(v, list):
                        return [int(x) for x in v if isinstance(x, (int, float))]
                    return []

                seen_hashes = _mcq_dedupe_hashes(st.session_state.mcq_by_chapter.get(st.session_state.active_chapter_index, []))
                normalized = []
                skipped = 0
                reindexed = 0
                downgraded_basic = 0
                out_of_range_fixed = 0
                for it in items:
                    if not isinstance(it, dict):
                        skipped += 1
                        continue
                    stem = (it.get("stem") or "").strip()
                    options = it.get("options") or []
                    if not stem or not isinstance(options, list) or len(options) < 2:
                        # Drop trivially invalid (need at least 2 options; generation prefers >=4, but allow 2 for TF)
                        skipped += 1
                        continue
                    # Normalize qtype
                    qtype = (it.get("qtype") or "multiple_choice").strip()
                    if qtype == "basic":
                        qtype = "multiple_choice"
                        downgraded_basic += 1
                    # Fix common typos (extra spaces etc.)
                    if qtype.replace(" ", "") == "multiplechoice":
                        qtype = "multiple_choice"
                    # Coerce correct indices
                    correct = _coerce_int_list(it.get("correct"))
                    if not correct:
                        # Attempt to salvage: if original had 'answer' and it's an option string
                        ans = it.get("answer") or it.get("a")
                        if ans and isinstance(ans, str) and ans in options:
                            correct = [options.index(ans)]
                    # Convert 1-based indices if any equals len(options) (common import mistake) and 0 not present
                    if correct and max(correct) >= len(options):
                        # Try shifting to 0-based if all between 1 and len(options)
                        if all(1 <= c <= len(options) for c in correct):
                            correct = [c - 1 for c in correct]
                            reindexed += 1
                        else:
                            # Drop out-of-range values
                            new_correct = [c for c in correct if 0 <= c < len(options)]
                            if new_correct:
                                out_of_range_fixed += 1
                                correct = new_correct
                            else:
                                skipped += 1
                                continue
                    # Final guard: single or multi list not empty
                    if not correct:
                        skipped += 1
                        continue
                    explanation = (it.get("explanation") or "").strip()
                    difficulty = (it.get("difficulty") or "medium").lower()
                    if difficulty not in {"easy","medium","hard"}:
                        difficulty = "medium"
                    tags = it.get("tags") or []
                    if not isinstance(tags, list):
                        tags = []
                    source = (it.get("source") or "").strip()
                    m = {
                        "stem": stem,
                        "options": [str(o).strip() for o in options],
                        "correct": correct,
                        "explanation": explanation,
                        "difficulty": difficulty,
                        "tags": tags,
                        "source": source,
                        "qtype": qtype or "multiple_choice",
                        "id": it.get("id",""),
                        "hash": it.get("hash",""),
                        "meta": it.get("meta"),
                    }
                    # Assign hash/id if missing
                    if not m["hash"]:
                        m["hash"] = _stem_hash(m["stem"])
                    if not m["id"]:
                        m["id"] = hashlib.md5((m["hash"] + "|" + "|".join(m["options"]) ).encode()).hexdigest()
                    if m["hash"] in seen_hashes:
                        continue
                    seen_hashes.add(m["hash"])
                    normalized.append(m)

                if not normalized:
                    st.error("No valid MCQs found in file after normalization; ensure each item has a stem, >=2 options, and valid correct indices.")
                else:
                    ch_idx = st.session_state.active_chapter_index
                    st.session_state.mcq_by_chapter.setdefault(ch_idx, [])
                    st.session_state.mcq_by_chapter[ch_idx].extend(normalized)
                    st.success(
                        f"Imported {len(normalized)} MCQs (skipped {skipped}; basic→multiple_choice {downgraded_basic}; reindexed {reindexed}; trimmed out-of-range {out_of_range_fixed})."
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

        # Custom instructions for MCQ generation
        if "custom_mcq_instructions" not in st.session_state:
            st.session_state.custom_mcq_instructions = ""
        
        # Optional compare/contrast entity hints
        entity_hint = ""
        if s.get("qtype") == "compare_contrast":
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                ent_a = st.text_input("Entity A (optional)", key="cc_entity_a")
            with col_e2:
                ent_b = st.text_input("Entity B (optional)", key="cc_entity_b")
            if ent_a and ent_b:
                entity_hint = f" When making compare/contrast MCQs, focus on: {ent_a} vs {ent_b}. Use both names in the stem and two-clause options."

        st.session_state.custom_mcq_instructions = st.text_area(
            "**(Optional)** Add specific instructions to guide MCQ generation:",
            value=(st.session_state.custom_mcq_instructions + entity_hint) if entity_hint else st.session_state.custom_mcq_instructions,
            placeholder="e.g., Focus on application questions, include scenario-based questions, avoid trivial details, etc.",
            height=80,
            help="These instructions will be included in the prompt to help generate more targeted MCQs"
        )

        # Generation button and MCQ count at the bottom
        st.markdown("")
        mcqs = st.session_state.mcq_by_chapter.get(st.session_state.active_chapter_index, [])
        
        col_gen, col_randomise, col_quiz, col_empty = st.columns([2, 1.7, 2, 6.3], gap="small")
        with col_gen:
            if st.button("Generate (More) MCQs", type="primary"):
                # Validate settings before generation
                if s["n_qs"] <= 0:
                    st.error("Please set the number of MCQs to generate to at least 1!")
                elif not s["difficulty_mix"]:
                    st.error("Please select at least one difficulty level!")
                else:
                    generate_mcqs_for_active_chapter()
        
        with col_randomise:
            if mcqs and st.button("Shuffle Questions", help="Shuffle the order of MCQs"):
                random.shuffle(mcqs)
                st.session_state.mcq_by_chapter[st.session_state.active_chapter_index] = mcqs
                st.success("MCQs shuffled!")
                st.rerun()
        
        with col_quiz:
            if mcqs and st.button("Start MCQ Quiz", type="primary", help="Start an interactive quiz with these questions"):
                # Initialize quiz state and go to test mode
                st.session_state.mcq["generated"]["items"] = mcqs.copy()
                st.session_state.mcq["step"] = 1
                reset_mcq_test_state()
                st.rerun()
        
        # MCQ count display
        st.markdown(f"**{len(mcqs)} MCQs in this chapter**")

        # Display generated MCQs with enhanced formatting
        for i, q in enumerate(mcqs):
            difficulty = q.get('difficulty', 'medium').title()
            source = q.get('source', 'Unknown')
            
            # Show difficulty and source as the main title
            difficulty_emoji = {'Easy': '🟢', 'Medium': '🟡', 'Hard': '🔴'}.get(difficulty, '⚫')
            
            with st.expander(f"{difficulty_emoji} {i+1}. {difficulty} - {source}"):
                # Show the actual question
                st.markdown(f"**Question:** {q['stem']}")
                
                # Show options with letters
                st.markdown("**Options:**")
                for j, opt in enumerate(q['options']):
                    st.markdown(f"{chr(65+j)}. {opt}")
                
                # Answer in nested expander
                with st.expander("🔍 Show Answer", expanded=False):
                    correct_letters = [chr(65+idx) for idx in q['correct']]
                    if len(correct_letters) == 1:
                        st.markdown(f"**Correct Answer:** {correct_letters[0]}")
                    else:
                        st.markdown(f"**Correct Answers:** {', '.join(correct_letters)}")
                    
                    # Show the correct option text(s)
                    for idx in q['correct']:
                        if idx < len(q['options']):
                            st.markdown(f"✅ {q['options'][idx]}")
                    
                    if q.get('explanation'):
                        st.markdown(f"**Explanation:** {q['explanation']}")
                
                # Enhanced metadata display
                tags_display = ', '.join(q.get('tags', [])) if q.get('tags') else 'None'
                qtype_display = (q.get('qtype') or 'multiple_choice').replace('_',' ').title()
                st.caption(f"**Type:** {qtype_display} | **Tags:** {tags_display}")
                
                # Optional per-MCQ delete
                if st.button("Delete this MCQ", key=f"del_mcq_{i}"):
                    del mcqs[i]
                    st.session_state.mcq_by_chapter[st.session_state.active_chapter_index] = mcqs
                    st.rerun()
        
        # Export buttons
        if mcqs:
            st.markdown("---")
            st.markdown("#### Export Options")
            colA, colB, colC, colD = st.columns([1, 0.95, 1, 4.05], gap="small")
            with colA:
                export_data = {
                    "items": mcqs,
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
                st.download_button(
                    "Download as JSON",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"{active_chapter['title']}_mcqs.json",
                    mime="application/json",
                    help="Download MCQs in JSON format for importing into this app or other applications"
                )
            with colB:
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
                    "Download as CSV",
                    data=output.getvalue().encode("utf-8"),
                    file_name=f"{active_chapter['title']}_mcqs.csv",
                    mime="text/csv",
                    help="Download MCQs in CSV format for importing into other applications"
                )
            with colC:
                if st.button("Delete All", type="secondary", help="Delete all MCQs for this chapter"):
                    # Delete all MCQs immediately
                    st.session_state.mcq_by_chapter[st.session_state.active_chapter_index] = []
                    st.success("All MCQs deleted!")
                    st.rerun()
