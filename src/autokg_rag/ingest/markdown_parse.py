"""Markdown parsing utilities for clean text ingestion."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from autokg_rag.exceptions import IngestError


def sha256_for_file(path: Path) -> str:
    """Return stable SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_markdown_files(input_dir: Path) -> list[Path]:
    """Discover Markdown files under an input directory in stable order."""
    if not input_dir.exists():
        raise IngestError(f"Input path does not exist: {input_dir}")

    md_files = sorted(input_dir.rglob("*.md"))
    if not md_files:
        raise IngestError(f"No Markdown files found under {input_dir}")

    return md_files


def parse_markdown_sections(path: Path) -> list[dict[str, str]]:
    """Parse markdown file into sections based on headings.
    
    Returns a list of dicts with 'level', 'title', 'content', and 'full_path'.
    """
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    sections: list[dict[str, str]] = []
    current_section: dict[str, str] | None = None
    section_stack: list[tuple[int, str]] = []  # (level, title) stack for hierarchy
    
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    
    for line in lines:
        match = heading_pattern.match(line)
        if match:
            # Save previous section content
            if current_section is not None:
                sections.append(current_section)
            
            level = len(match.group(1))
            title = match.group(2).strip()
            
            # Build hierarchical path
            # Pop sections from stack that are at same or higher level
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            
            section_stack.append((level, title))
            full_path = " / ".join(s[1] for s in section_stack)
            
            current_section = {
                "level": level,
                "title": title,
                "content": "",
                "full_path": full_path,
            }
        elif current_section is not None:
            # Add content to current section
            current_section["content"] += line + "\n"
    
    # Don't forget the last section
    if current_section is not None:
        sections.append(current_section)
    
    return sections


def parse_markdown_to_chunks(
    path: Path,
    *,
    chunk_word_size: int = 400,
    chunk_word_overlap: int = 50,
) -> list[dict[str, str]]:
    """Parse markdown into chunks, respecting section boundaries.
    
    This is a higher-level function that produces chunks suitable for
    vector embedding while preserving section context.
    """
    sections = parse_markdown_sections(path)
    chunks: list[dict[str, str]] = []
    chunk_id = 0
    
    for section in sections:
        content = section["content"].strip()
        if not content:
            continue
        
        # Split content into words
        words = content.split()
        
        # If section is small enough, keep as single chunk
        if len(words) <= chunk_word_size:
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:06d}",
                "section_path": section["full_path"],
                "section_level": section["level"],
                "chunk_text": content,
            })
            chunk_id += 1
        else:
            # Split into overlapping chunks
            start = 0
            while start < len(words):
                end = min(start + chunk_word_size, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)
                
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id:06d}",
                    "section_path": section["full_path"],
                    "section_level": section["level"],
                    "chunk_text": chunk_text,
                })
                chunk_id += 1
                
                # Move start forward, accounting for overlap
                start = end - chunk_word_overlap
                if start >= len(words) - chunk_word_overlap:
                    break
    
    return chunks


def extract_title_from_markdown(path: Path) -> str:
    """Extract the first H1 heading as the document title."""
    content = path.read_text(encoding="utf-8")
    
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    
    # Fallback to filename
    return path.stem