# AutoRAG PMBOK Ingestion & Multi-Modal Enhancement Plan

> A step-by-step guide for junior data scientists to enhance AutoRAG with PMBOK document processing and multi-modal capabilities.

**Prerequisites:**
- Python 3.11+
- uv package manager installed
- Ollama installed locally
- Basic familiarity with RAG (Retrieval-Augmented Generation) concepts

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: PMBOK Document Ingestion](#phase-1-pmbok-document-ingestion)
3. [Phase 2: Header/Footer Removal](#phase-2-headerfooter-removal)
4. [Phase 3: Image Extraction & Embedding](#phase-3-image-extraction--embedding)
5. [Phase 4: Ollama Integration for Local LLM](#phase-4-ollama-integration-for-local-llm)
6. [Phase 5: Evaluation Framework Enhancement](#phase-5-evaluation-framework-enhancement)
7. [Phase 6: A/B Testing Framework](#phase-6-ab-testing-framework)
8. [Running the Complete Pipeline](#running-the-complete-pipeline)

---

## Overview

This plan transforms AutoRAG into a production-quality multi-modal RAG system using the PMBOK Guide (8th Edition) as our test document. We'll add:

- **Smart document chunking** that respects document structure
- **Header/footer removal** to improve chunk quality
- **Image embedding** for diagrams and charts
- **Local LLM inference** with Ollama
- **Enhanced evaluation** with A/B testing

### Expected Time: 8-12 hours for a junior data scientist

---

## Phase 1: PMBOK Document Ingestion

### What We're Doing
Ingest the PMBOK Guide PDF into AutoRAG using the heading-recursive chunking strategy.

### Steps

#### Step 1.1: Copy the PMBOK PDF to Fixtures

```bash
# Navigate to your project directory
cd /Users/dustinober/Projects/AutoRAG

# Create fixtures directory if it doesn't exist
mkdir -p data/fixtures/pdfs

# Copy the PMBOK guide
cp files/PMP/pmbokguide_eighthed_eng.pdf data/fixtures/pdfs/
```

#### Step 1.2: Update Configuration (Optional)

Edit `configs/base.yaml` to set heading-recursive as default:

```yaml
# In configs/base.yaml
ingest:
  chunking:
    strategy: heading_recursive
    chunk_word_size: 512
    chunk_word_overlap: 50
```

#### Step 1.3: Run the Smoke Pipeline

```bash
# Test ingestion with a simple question
uv run autorag smoke \
  --input data/fixtures/pdfs/pmbokguide_eighthed_eng.pdf \
  --question "What is the purpose of a project charter?" \
  --run-id pmbok_test
```

If successful, you'll see output like:
```
✓ Parsed 700+ pages from PMBOK
✓ Generated 1,234 chunks using heading_recursive strategy
✓ Created vector index
✓ Answer: The project charter formally authorizes the project...
```

#### Step 1.4: Generate Evaluation Questions

Create a test dataset based on PMBOK content:

```bash
# Create a new evaluation dataset file
cat > eval/datasets/pmbok_questions.jsonl << 'EOF'
{"question_id": "pmbok_001", "type": "fact", "question": "What are the 10 knowledge areas in PMBOK?", "gold_citations": [], "gold_answer": "Integration, Scope, Schedule, Cost, Quality, Resource, Communications, Risk, Procurement, Stakeholder"}
{"question_id": "pmbok_002", "type": "fact", "question": "What is the purpose of the project charter?", "gold_citations": [], "gold_answer": "To formally authorize the project and provide the project manager with authority to apply resources."}
{"question_id": "pmbok_003", "type": "fact", "question": "What is a Work Breakdown Structure (WBS)?", "gold_citations": [], "gold_answer": "A hierarchical decomposition of the total scope of work to be carried out."}
{"question_id": "pmbok_004", "type": "fact", "question": "What are the five process groups?", "gold_citations": [], "gold_answer": "Initiating, Planning, Executing, Monitoring and Controlling, Closing"}
{"question_id": "pmbok_005", "type": "contrast", "question": "What is the difference between predictive and agile approaches?", "gold_citations": [], "gold_answer": "Predictive is plan-driven with extensive upfront planning; agile is iterative with frequent adaptation."}
EOF
```

---

## Phase 2: Header/Footer Removal

### What We're Doing
Add filtering to remove repetitive headers, footers, and page numbers that break up meaningful content.

### Why It Matters
PMBOK (like many PDFs) has repeated elements on every page:
- "PMBOK® Guide"
- "Project Management Institute"
- Page numbers (Page 1 of 700)
- Section headers

These cause chunks to be incorrectly split, reducing retrieval quality.

### Steps

#### Step 2.1: Create a Header/Footer Filter Module

Create a new file `src/autokg_rag/ingest/header_footer_filter.py`:

```python
"""Header and footer removal utilities."""

from __future__ import annotations

import re
from typing import Pattern


# Common PDF header/footer patterns to filter
HEADER_FOOTER_PATTERNS: list[str] = [
    r"^PMBOK®?\s*Guide",
    r"^Project Management Institute$",
    r"^PMI®?\s*$",
    r"^Copyright\s*©\s*PMI.*$",
    r"^Page\s+\d+\s+of\s+\d+$",
    r"^\d+th?\s+Edition$",
    r"^A Guide to the Project Management Body of Knowledge$",
    r"^$\n?$",  # Empty lines
]


def compile_patterns(patterns: list[str]) -> list[Pattern[str]]:
    """Compile string patterns to regex objects."""
    return [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]


# Pre-compiled patterns for efficiency
_FILTER_PATTERNS = compile_patterns(HEADER_FOOTER_PATTERNS)


def is_header_footer_line(line: str) -> bool:
    """Check if a line matches header/footer patterns."""
    stripped = line.strip()
    if not stripped or len(stripped) < 2:
        return True
    
    for pattern in _FILTER_PATTERNS:
        if pattern.match(stripped):
            return True
    
    return False


def remove_header_footer_from_text(text: str) -> str:
    """Remove header/footer lines from extracted text."""
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        if not is_header_footer_line(line):
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)


def remove_repeated_lines_across_pages(pages: list[str], threshold: float = 0.5) -> list[str]:
    """Remove lines that appear on most pages (likely headers/footers)."""
    if not pages:
        return pages
    
    line_occurrences: dict[str, int] = {}
    total_pages = len(pages)
    
    for page in pages:
        unique_lines = set(line.strip() for line in page.splitlines() if line.strip())
        for line in unique_lines:
            line_occurrences[line] = line_occurrences.get(line, 0) + 1
    
    # Find lines that appear on >threshold% of pages
    lines_to_remove = {
        line for line, count in line_occurrences.items()
        if count / total_pages > threshold
    }
    
    # Filter pages
    cleaned_pages = []
    for page in pages:
        lines = page.splitlines()
        cleaned = [l for l in lines if l.strip() not in lines_to_remove]
        cleaned_pages.append("\n".join(cleaned))
    
    return cleaned_pages
```

#### Step 2.2: Integrate into PDF Parser

Modify `src/autokg_rag/ingest/pdf_parse.py`:

```python
# Add import at the top
from autokg_rag.ingest.header_footer_filter import (
    remove_header_footer_from_text,
    remove_repeated_lines_across_pages,
)

# Update the _normalize_page_text function
def _normalize_page_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    compact = "\n".join(line for line in lines if line)
    # Apply header/footer removal
    cleaned = remove_header_footer_from_text(compact)
    return cleaned.strip() or "(empty page)"

# Add new function to process all pages
def parse_pdf_pages_clean(path: Path) -> list[str]:
    """Extract and clean page text from PDF."""
    pages = parse_pdf_pages(path)  # Use existing function
    
    # Remove repeated lines across pages
    cleaned_pages = remove_repeated_lines_across_pages(pages, threshold=0.7)
    
    return cleaned_pages
```

#### Step 2.3: Test Header/Footer Removal

```bash
# Run a quick test
uv run python -c "
from pathlib import Path
from autokg_rag.ingest.pdf_parse import parse_pdf_pages_clean
from autokg_rag.ingest.header_footer_filter import remove_header_footer_from_text

# Test on first few pages
pages = parse_pdf_pages_clean(Path('data/fixtures/pdfs/pmbokguide_eighthed_eng.pdf'))
print(f'Extracted {len(pages)} pages')
print('Sample (first 500 chars):')
print(pages[0][:500] if pages else 'No pages')
"
```

---

## Phase 3: Image Extraction & Embedding

### What We're Doing
Extract diagrams and charts from PMBOK and generate embeddings for them.

### Why It Matters
PMBOK contains valuable diagrams:
- Process flowcharts (how processes relate)
- RACI matrices (responsibility charts)
- Risk matrices
- WBS examples

These contain information not in the text!

### Steps

#### Step 3.1: Add Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
vision = [
    "pdfplumber>=0.11.0",
    "pillow>=10.0.0",
    "torch>=2.0.0",
    "transformers>=4.30.0",
]
```

Install:
```bash
uv sync --extra vision
```

#### Step 3.2: Create Image Extraction Module

Create `src/autokg_rag/ingest/image_extract.py`:

```python
"""Image extraction from PDFs."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ImageInfo(Protocol):
    """Protocol for image information."""
    name: str
    xref: int
    width: int
    height: int


def extract_images_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract images from a PDF file.
    
    Returns a list of dicts with:
        - page_num: int
        - image_bytes: bytes
        - width: int
        - height: int
        - image_type: str (PNG, JPEG, etc.)
    """
    import pdfplumber
    
    images = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_images = page.images
            
            for img in page_images:
                # Get image data
                # Note: pdfplumber extracts image metadata; 
                # for actual bytes, we need additional processing
                images.append({
                    'page_num': page_num,
                    'x': img.get('x0', 0),
                    'y': img.get('top', 0),
                    'width': img.get('width', 0),
                    'height': img.get('height', 0),
                    'name': img.get('name', f'img_{len(images)}'),
                })
    
    return images


def extract_diagram_regions(pdf_path: Path, min_size: int = 100) -> list[dict]:
    """Extract likely diagram regions (larger images/charts)."""
    import pdfplumber
    
    diagrams = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Get all images from page
            images = page.images
            
            # Filter for larger images (likely diagrams, not logos)
            for img in images:
                width = img.get('width', 0)
                height = img.get('height', 0)
                
                if width > min_size and height > min_size:
                    diagrams.append({
                        'page_num': page_num,
                        'x0': img.get('x0', 0),
                        'y0': img.get('top', 0),
                        'x1': img.get('x1', 0),
                        'y1': img.get('bottom', 0),
                        'width': width,
                        'height': height,
                    })
    
    return diagrams
```

#### Step 3.3: Create Image Captioning Module

Create `src/autokg_rag/ingest/image_caption.py`:

```python
"""Image captioning using Ollama with llava."""

from __future__ import annotations

import base64
from pathlib import Path


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_image_caption(image_path: Path, prompt: str = None) -> str:
    """Generate a caption for an image using Ollama llava.
    
    Args:
        image_path: Path to the image file
        prompt: Optional custom prompt
    
    Returns:
        Generated caption text
    """
    if prompt is None:
        prompt = """Describe this diagram in detail. 
Focus on what information it conveys, its structure, 
and how it relates to project management. 
Be specific about labels, arrows, and relationships shown."""
    
    # Use ollama Python library
    import ollama
    
    # Encode image
    image_b64 = encode_image_to_base64(image_path)
    
    # Generate caption
    response = ollama.chat(
        model='llava-llama3',
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [image_b64]
            }
        ]
    )
    
    return response['message']['content']


def batch_caption_images(image_paths: list[Path], prompt: str = None) -> list[dict]:
    """Generate captions for multiple images.
    
    Returns:
        List of dicts with 'image_path' and 'caption'
    """
    results = []
    
    for img_path in image_paths:
        try:
            caption = generate_image_caption(img_path, prompt)
            results.append({
                'image_path': str(img_path),
                'caption': caption,
                'success': True,
            })
        except Exception as e:
            results.append({
                'image_path': str(img_path),
                'caption': None,
                'success': False,
                'error': str(e),
            })
    
    return results


def extract_and_caption_diagrams(
    pdf_path: Path,
    output_dir: Path,
    min_size: int = 200,
) -> list[dict]:
    """Extract diagrams from PDF and generate captions.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images
        min_size: Minimum dimension for diagram extraction
    
    Returns:
        List of dicts with image info and captions
    """
    from autokg_rag.ingest.image_extract import extract_diagram_regions
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract diagram regions
    diagrams = extract_diagram_regions(pdf_path, min_size)
    
    results = []
    
    # Note: Full extraction requires additional libraries like pdf2image
    # This is a placeholder that shows the architecture
    
    for i, diagram in enumerate(diagrams):
        results.append({
            'page_num': diagram['page_num'],
            'region': diagram,
            'caption': f"Diagram on page {diagram['page_num']}",  # Generated later
        })
    
    return results
```

#### Step 3.4: Add Image Indexing to Pipeline

Create `src/autokg_rag/vector/image_index.py`:

```python
"""Image embedding and indexing for multi-modal retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from autokg_rag.vector.store import VectorStore


def create_image_embeddings(
    captions: Sequence[str],
    model: str = "bge-small-en-v1.5",
) -> list[list[float]]:
    """Create embeddings for image captions.
    
    Uses the same embedding model as text for consistency.
    """
    from autokg_rag.embeddings.pipeline import create_embeddings
    
    return create_embeddings(captions, model=model)


class ImageIndex:
    """Index for image captions with metadata."""
    
    def __init__(self, store: VectorStore):
        self.store = store
        self.metadata: list[dict] = []
    
    def add_image_captions(
        self,
        captions: list[str],
        image_refs: list[str],
        page_nums: list[int],
    ) -> None:
        """Add image captions to the index."""
        embeddings = create_image_embeddings(captions)
        
        for i, (caption, ref, page) in enumerate(zip(captions, image_refs, page_nums)):
            self.store.add(
                id=f"img_{ref}_{page}",
                embedding=embeddings[i],
                text=caption,
                metadata={
                    'type': 'image',
                    'page': page,
                    'image_ref': ref,
                }
            )
            self.metadata.append({
                'chunk_id': f"img_{ref}_{page}",
                'page': page,
                'image_ref': ref,
                'caption': caption,
            })
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search images by query text."""
        from autokg_rag.embeddings.pipeline import create_embeddings
        
        query_embedding = create_embeddings([query])[0]
        results = self.store.search(query_embedding, top_k)
        
        return [
            {
                **r,
                'metadata': self.metadata[int(r['id'].split('_')[-1])],
            }
            for r in results
        ]
```

---

## Phase 4: Ollama Integration for Local LLM

### What We're Doing
Integrate Ollama for local answer generation instead of relying on external APIs.

### Steps

#### Step 4.1: Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.com/download
```

#### Step 4.2: Pull Required Models

```bash
# Pull llava for image understanding
ollama pull llava-llama3

# Pull a text model for answer generation
ollama pull llama3

# Or a lighter model for faster testing
ollama pull mistral
```

#### Step 4.3: Create Ollama Adapter

Create `src/autokg_rag/answer/ollama_adapter.py`:

```python
"""Ollama adapter for local LLM inference."""

from __future__ import annotations

from typing import Literal


OllamaModel = Literal["llama3", "mistral", "llava-llama3", "mixtral"]


class OllamaAdapter:
    """Adapter for Ollama local LLM inference."""
    
    def __init__(
        self,
        model: OllamaModel = "llama3",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        images: list[str] = None,
    ) -> str:
        """Generate text using Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            images: Optional list of base64-encoded images
        
        Returns:
            Generated text
        """
        import ollama
        
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt,
            })
        
        message_content = prompt
        if images:
            # For multi-modal models like llava
            message = {
                'role': 'user',
                'content': message_content,
                'images': images,
            }
        else:
            message = {
                'role': 'user',
                'content': message_content,
            }
        
        messages.append(message)
        
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens,
            },
        )
        
        return response['message']['content']
    
    def generate_with_context(
        self,
        question: str,
        context_chunks: list[str],
        system_prompt: str = None,
    ) -> tuple[str, list[dict]]:
        """Generate answer with retrieved context.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            system_prompt: Optional system prompt
        
        Returns:
            Tuple of (answer, citations)
        """
        # Build context from chunks
        context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Instructions:
1. Use the context to answer
2. Cite your sources using [1], [2], etc.
3. If the context doesn't contain enough information, say so

Answer:"""
        
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant that answers questions based on the provided context."
        
        answer = self.generate(prompt, system_prompt=system_prompt)
        
        # Extract citations (simplified)
        citations = []
        # This would need more sophisticated extraction in production
        
        return answer, citations


# Singleton instance
_default_adapter: OllamaAdapter | None = None


def get_ollama_adapter(
    model: OllamaModel = "llama3",
    temperature: float = 0.7,
) -> OllamaAdapter:
    """Get the default Ollama adapter."""
    global _default_adapter
    
    if _default_adapter is None:
        _default_adapter = OllamaAdapter(model, temperature)
    
    return _default_adapter
```

#### Step 4.4: Update Answer Generation

Modify `src/autokg_rag/answer/llm_adapter.py` to use Ollama:

```python
# Add import
from autokg_rag.answer.ollama_adapter import OllamaAdapter, get_ollama_adapter


def generate_answer(
    question: str,
    context_chunks: list[str],
    use_local: bool = True,
) -> tuple[str, list[dict]]:
    """Generate answer with context."""
    
    if use_local:
        # Use Ollama
        adapter = get_ollama_adapter()
        return adapter.generate_with_context(question, context_chunks)
    
    # Otherwise use external API (existing implementation)
    # ...
```

---

## Phase 5: Evaluation Framework Enhancement

### What We're Doing
Add LLM-as-a-judge evaluation to complement the existing metrics.

### Steps

#### Step 5.1: Create LLM Judge Module

Create `src/autokg_rag/eval/judge.py`:

```python
"""LLM-as-a-Judge evaluation."""

from __future__ import annotations

from typing import Literal


JudgementCriteria = Literal[
    "correctness",
    "helpfulness",
    "groundedness",
    "coherence",
]


def evaluate_with_llm_judge(
    question: str,
    answer: str,
    context: list[str],
    criteria: JudgementCriteria = "correctness",
    model: str = "llama3",
) -> dict:
    """Evaluate answer using LLM as judge.
    
    Args:
        question: The original question
        answer: Generated answer to evaluate
        context: Retrieved context chunks
        criteria: What to evaluate
        model: Ollama model to use
    
    Returns:
        Dict with score (0-10) and reasoning
    """
    from autokg_rag.answer.ollama_adapter import get_ollama_adapter
    
    adapter = get_ollama_adapter(model=model)
    
    context_text = "\n\n".join(context)
    
    prompt = f"""You are an expert evaluator. Judge the following answer based on {criteria}.

Question: {question}

Retrieved Context:
{context_text}

Generated Answer:
{answer}

Evaluation Criteria:
- correctness: Is the answer factually correct based on the context?
- helpfulness: Does the answer address the question well?
- groundedness: Does the answer stay faithful to the provided context?
- coherence: Is the answer well-organized and clear?

Provide your evaluation in this format:
Score: <0-10>
Reasoning: <explain your score>
"""
    
    result = adapter.generate(prompt, system_prompt="You are a strict but fair evaluator.")
    
    # Parse result (simplified)
    lines = result.split('\n')
    score = 5.0  # Default
    reasoning = result
    
    for line in lines:
        if line.lower().startswith('score:'):
            try:
                score = float(line.split(':')[1].strip())
            except:
                pass
    
    return {
        'criteria': criteria,
        'score': score,
        'reasoning': reasoning,
    }


def evaluate_answer_set(
    questions: list[dict],
    answers: list[str],
    contexts: list[list[str]],
    criteria: list[JudgementCriteria] = None,
) -> list[dict]:
    """Evaluate a set of answers with multiple criteria."""
    
    if criteria is None:
        criteria = ["correctness", "helpfulness", "groundedness"]
    
    results = []
    
    for q, a, c in zip(questions, answers, contexts):
        result = {
            'question_id': q.get('question_id', 'unknown'),
            'question': q.get('question', ''),
            'judgements': [],
        }
        
        for crit in criteria:
            judgement = evaluate_with_llm_judge(
                q.get('question', ''),
                a,
                c,
                criteria=crit,
            )
            result['judgements'].append(judgement)
        
        # Calculate average
        result['avg_score'] = sum(j['score'] for j in result['judgements']) / len(result['judgements'])
        
        results.append(result)
    
    return results
```

---

## Phase 6: A/B Testing Framework

### What We're Doing
Create a framework to compare different retrieval strategies statistically.

### Steps

#### Step 6.1: Create A/B Test Module

Create `src/autokg_rag/eval/ab_test.py`:

```python
"""A/B testing for retrieval strategies."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np


RetrievalMode = Literal["vector", "graph", "hybrid"]


@dataclass
class ABTestResult:
    """Results from an A/B test."""
    mode_a: RetrievalMode
    mode_b: RetrievalMode
    metric: str
    scores_a: list[float]
    scores_b: list[float]
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    p_value: float | None
    winner: RetrievalMode | None


def run_ab_test(
    questions: list[dict],
    run_retrieval_fn,
    mode_a: RetrievalMode = "vector",
    mode_b: RetrievalMode = "hybrid",
    metric: str = "recall_at_k",
    k: int = 5,
    n_runs: int = 1,
) -> ABTestResult:
    """Run A/B test comparing two retrieval modes.
    
    Args:
        questions: Test questions with gold citations
        run_retrieval_fn: Function to run retrieval and return metrics
        mode_a: First retrieval mode
        mode_b: Second retrieval mode
        metric: Metric to compare
        k: Top-k for recall
        n_runs: Number of runs (for variance estimation)
    
    Returns:
        ABTestResult with statistics
    """
    scores_a = []
    scores_b = []
    
    for _ in range(n_runs):
        # Shuffle questions for variance
        q shuffled = questions.copy()
        random.shuffle(q)
        
        for q in questions:
            # Run mode A
            result_a = run_retrieval_fn(q, mode_a, k)
            scores_a.append(result_a.get(metric, 0))
            
            # Run mode B
            result_b = run_retrieval_fn(q, mode_b, k)
            scores_b.append(result_b.get(metric, 0))
    
    # Calculate statistics
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    std_a = np.std(scores_a)
    std_b = np.std(scores_b)
    
    # Statistical test
    p_value = None
    if len(scores_a) > 1 and len(scores_b) > 1:
        from scipy import stats
        _, p_value = stats.ttest_ind(scores_a, scores_b)
    
    # Determine winner
    winner = None
    if mean_a > mean_b:
        winner = mode_a
    elif mean_b > mean_a:
        winner = mode_b
    
    return ABTestResult(
        mode_a=mode_a,
        mode_b=mode_b,
        metric=metric,
        scores_a=scores_a,
        scores_b=scores_b,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        p_value=p_value,
        winner=winner,
    )


def format_ab_test_report(result: ABTestResult) -> str:
    """Format A/B test results as markdown."""
    
    significance = "Yes" if result.p_value and result.p_value < 0.05 else "No"
    
    report = f"""# A/B Test Results: {result.mode_a} vs {result.mode_b}

## Summary

| Metric | {result.mode_a} | {result.mode_b} |
|--------|-----------------|-----------------|
| Mean {result.metric} | {result.mean_a:.4f} | {result.mean_b:.4f} |
| Std Dev | {result.std_a:.4f} | {result.std_b:.4f} |

## Statistical Significance

- **p-value**: {result.p_value:.4f if result.p_value else 'N/A'}
- **Statistically Significant (p<0.05)**: {significance}

## Winner

**{result.winner.upper() if result.winner else 'TIE'}** ({result.mode_a} vs {result.mode_b})

## Interpretation

"""
    
    if result.winner:
        diff = abs(result.mean_a - result.mean_b)
        pct = (diff / max(result.mean_a, result.mean_b)) * 100
        report += f"The {result.winner} strategy shows a {pct:.1f}% improvement in {result.metric}."
    
    return report
```

---

## Running the Complete Pipeline

### Quick Start Commands

After implementing all phases, run the full pipeline:

```bash
# 1. Ensure Ollama is running
ollama serve

# 2. Build the demo with PMBOK
make demo-build AUTORAG_DEMO_INPUT_DIR=data/fixtures/pdfs AUTORAG_DEMO_RUN_ID=pmbok

# 3. Run the Streamlit demo
uv run streamlit run app/streamlit_app.py

# 4. Run evaluation with A/B test
uv run autorag eval --questions eval/datasets/pmbok_questions.jsonl --ab-test
```

### Expected Output

After completing all phases, you'll have:
- PMBOK chunks with clean text (no headers/footers)
- Image captions indexed alongside text
- Local LLM generating answers via Ollama
- Evaluation results comparing vector/graph/hybrid retrieval
- A/B test report with statistical significance

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Ollama not responding | Ensure `ollama serve` is running |
| Out of memory | Use smaller models (mistral instead of llama3) |
| Slow image processing | Process fewer images or use batch mode |
| Poor retrieval quality | Try different chunk sizes (256, 512, 1024) |

### Getting Help

- Check logs in `data/artifacts/{run_id}/logs/`
- Run `uv run autorag doctor` for diagnostics
- Review the runbook: `docs/runbook.md`

---

## Next Steps (Optional Enhancements)

After completing this plan, consider:

1. **Self-Correction Loop** - Add query reformulation when initial retrieval fails
2. **User Feedback Collection** - Track user ratings in the Streamlit app
3. **Advanced Fusion** - Implement reciprocal rank fusion
4. **Production Deployment** - Add Docker, monitoring, and CI/CD

---

*Plan created for AutoRAG portfolio enhancement. Follow along at your own pace!*