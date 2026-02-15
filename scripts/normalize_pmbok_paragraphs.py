#!/usr/bin/env python3
"""
Normalize paragraph formatting in PMBOK markdown file.

Merges lines that belong to the same paragraph into single lines while
preserving markdown structure (headings, lists, code blocks, tables, blank lines).

This is critical for Graph RAG quality as broken paragraphs fragment semantic meaning.
"""

import re
from pathlib import Path


def is_heading(line: str) -> bool:
    """Check if line is a markdown heading."""
    return line.lstrip().startswith("#")


def is_list_item(line: str) -> bool:
    """Check if line is a markdown list item (bullet or numbered)."""
    stripped = line.lstrip()
    # Bullet list: -, *, +
    if re.match(r"^[-*+]\s", stripped):
        return True
    # Numbered list: 1., 2., etc.
    if re.match(r"^\d+\.\s", stripped):
        return True
    return False


def is_code_block_delimiter(line: str) -> bool:
    """Check if line is a code block delimiter (```)."""
    return line.lstrip().startswith("```")


def is_table_line(line: str) -> bool:
    """Check if line is part of a markdown table (contains |)."""
    return "|" in line


def is_blank_line(line: str) -> bool:
    """Check if line is blank or contains only whitespace."""
    return not line.strip()


def is_special_block_start(line: str) -> bool:
    """Check if line starts a special block that should not be merged."""
    return (
        is_heading(line)
        or is_list_item(line)
        or is_code_block_delimiter(line)
        or is_table_line(line)
        or is_blank_line(line)
    )


def normalize_paragraphs(content: str) -> str:
    """
    Normalize paragraphs by merging consecutive non-special lines.
    
    Preserves:
    - Headings (lines starting with #)
    - List items (lines starting with -, *, +, or numbered 1., 2.)
    - Code blocks (lines between ``` markers)
    - Tables (lines with | delimiters)
    - Blank lines (paragraph separators)
    
    Also handles multi-line bullet points by merging continuation lines
    with their preceding bullet item.
    """
    lines = content.split("\n")
    result = []
    current_paragraph = []
    in_code_block = False
    current_list_item = None  # Track the current list item for continuation merging
    
    for line in lines:
        # Track code block state
        if is_code_block_delimiter(line):
            # Flush any pending paragraph before code block
            if current_paragraph and not in_code_block:
                merged = " ".join(current_paragraph)
                result.append(merged)
                current_paragraph = []
            # Flush any pending list item
            if current_list_item is not None:
                result.append(current_list_item)
                current_list_item = None
            
            in_code_block = not in_code_block
            result.append(line)
            continue
        
        # Inside code blocks, preserve lines as-is
        if in_code_block:
            result.append(line)
            continue
        
        # Handle blank lines - end current paragraph and list item
        if is_blank_line(line):
            if current_paragraph:
                merged = " ".join(current_paragraph)
                result.append(merged)
                current_paragraph = []
            if current_list_item is not None:
                result.append(current_list_item)
                current_list_item = None
            result.append(line)
            continue
        
        # Handle headings - end current paragraph and list item, add heading as-is
        if is_heading(line):
            if current_paragraph:
                merged = " ".join(current_paragraph)
                result.append(merged)
                current_paragraph = []
            if current_list_item is not None:
                result.append(current_list_item)
                current_list_item = None
            result.append(line)
            continue
        
        # Handle list items - end current paragraph, start new list item
        if is_list_item(line):
            if current_paragraph:
                merged = " ".join(current_paragraph)
                result.append(merged)
                current_paragraph = []
            # Flush previous list item if any
            if current_list_item is not None:
                result.append(current_list_item)
            current_list_item = line
            continue
        
        # Handle table lines - end current paragraph and list item, add table line as-is
        if is_table_line(line):
            if current_paragraph:
                merged = " ".join(current_paragraph)
                result.append(merged)
                current_paragraph = []
            if current_list_item is not None:
                result.append(current_list_item)
                current_list_item = None
            result.append(line)
            continue
        
        # Check if this is a continuation line for a list item
        if current_list_item is not None:
            # Merge continuation line with the current list item
            current_list_item = current_list_item.rstrip() + " " + line.strip()
            continue
        
        # Regular text line - accumulate for paragraph merging
        current_paragraph.append(line.strip())
    
    # Flush any remaining paragraph
    if current_paragraph:
        merged = " ".join(current_paragraph)
        result.append(merged)
    
    # Flush any remaining list item
    if current_list_item is not None:
        result.append(current_list_item)
    
    return "\n".join(result)


def main():
    """Main entry point."""
    # Define paths
    markdown_path = Path("data/fixtures/pmbok/markdown/pmbokguide_eighthed_eng.md")
    
    # Read the file
    print(f"Reading {markdown_path}...")
    content = markdown_path.read_text(encoding="utf-8")
    
    # Count original lines
    original_lines = len(content.split("\n"))
    print(f"Original line count: {original_lines}")
    
    # Normalize paragraphs
    print("Normalizing paragraphs...")
    normalized_content = normalize_paragraphs(content)
    
    # Count normalized lines
    normalized_lines = len(normalized_content.split("\n"))
    print(f"Normalized line count: {normalized_lines}")
    print(f"Lines reduced: {original_lines - normalized_lines}")
    
    # Write back
    print(f"Writing normalized content to {markdown_path}...")
    markdown_path.write_text(normalized_content, encoding="utf-8")
    
    print("Done!")


if __name__ == "__main__":
    main()