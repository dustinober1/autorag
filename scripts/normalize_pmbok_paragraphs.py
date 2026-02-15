#!/usr/bin/env python3
"""
Normalize paragraph formatting in PMBOK markdown file.

Merges lines that belong to the same paragraph into single lines while
preserving markdown structure (headings, lists, code blocks, tables, blank lines).

This is critical for Graph RAG quality as broken paragraphs fragment semantic meaning.

Also removes PDF artifacts:
- Standalone page numbers (Arabic and Roman numerals)
- Running headers (page number + section title patterns)
- Repeated section headers that appear as running headers
- Section headers with trailing page numbers (e.g., "Section 1 – Introduction 5")
- Broken bold headers where text runs together (e.g., "GuideAffinity")
- Standalone "overview of the content." artifact lines
"""

import re
from pathlib import Path


# Known running header patterns from PMBOK PDF
RUNNING_HEADERS = [
    "The Standard for Project Management",
    "A Guide to the Project Management Body of Knowledge",
    "PMBOK® Guide",
    "PMBOK [®] Guide",
    "Table of Contents",
    "List of Figures and Tables",
    "Preface",
    "Section 1",
    "Section 2",
    "Section 3",
    "Section 4",
    "Section 5",
    "Appendix X1",
    "Appendix X2",
    "Appendix X3",
    "Appendix X4",
    "Appendix X5",
]

# Build regex pattern for running headers with page numbers
# Pattern: NUMBER + HEADER (e.g., "32 The Standard for Project Management")
HEADER_PATTERN_PART = "|".join(re.escape(h) for h in RUNNING_HEADERS)
RUNNING_HEADER_REGEX = re.compile(
    rf"^\d+\s+({HEADER_PATTERN_PART})(?:\s|$)"
)

# Pattern for section running headers like "Section 1 Introduction 5" or "Section 2 A System for Value Delivery"
SECTION_HEADER_REGEX = re.compile(
    r"^(Section\s+\d+)\s*[\-–—]\s*(.+?)(?:\s+\d+)?$"
)

# Pattern for section headers with trailing page numbers
# Example: "Section 1 – Introduction 5" -> "Section 1 – Introduction"
SECTION_HEADER_WITH_PAGE_REGEX = re.compile(r"^(Section\s+\d+\s*[\-–—]\s*.+)\s+\d+$")

# Pattern for broken bold headers where next word is attached without space
# Example: "**1.1 Structure of the PMBOK® GuideAffinity diagram.**" -> "**1.1 Structure of the PMBOK® Guide Affinity diagram.**"
BROKEN_BOLD_HEADER_REGEX = re.compile(r"\*\*(.+?Guide)([A-Z][a-z]+)")

# Pattern for standalone "overview of the content." lines that are PDF artifacts
STANDALONE_OVERVIEW_REGEX = re.compile(r"^\*\*overview of the content\.\*\*$")

# Pattern for standalone page numbers (Arabic numerals, 1-4 digits)
PAGE_NUMBER_REGEX = re.compile(r"^\d{1,4}$")

# Pattern for Roman numerals (commonly used in front matter)
ROMAN_NUMERAL_REGEX = re.compile(
    r"^(?:i{1,3}|iv|v|vi{1,3}|ix|x{1,3}|xi{1,3}|xiv|xv|xvi{1,3}|xix|xx{1,3}|xxi{1,3}|xxiv|xxv|xxvi{1,3}|xxix|xxx{1,3})$",
    re.IGNORECASE
)

# Pattern for bold page numbers like "**3**"
BOLD_PAGE_NUMBER_REGEX = re.compile(r"^\*\*\d{1,4}\*\*$")


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


def is_page_number(line: str) -> bool:
    """
    Check if line is a standalone page number (Arabic or Roman).
    
    Examples:
    - "39" (Arabic)
    - "v", "vii", "xii" (Roman numerals in front matter)
    - "**3**" (bold page numbers)
    """
    stripped = line.strip()
    if not stripped:
        return False
    
    # Check for plain Arabic page numbers (1-4 digits)
    if PAGE_NUMBER_REGEX.match(stripped):
        return True
    
    # Check for Roman numerals (front matter pages)
    if ROMAN_NUMERAL_REGEX.match(stripped):
        return True
    
    # Check for bold page numbers like "**3**"
    if BOLD_PAGE_NUMBER_REGEX.match(stripped):
        return True
    
    return False


def is_running_header(line: str) -> bool:
    """
    Check if line is a running header from PDF.
    
    Examples:
    - "32 The Standard for Project Management"
    - "4 The Standard for Project Management"
    - "viii Preface"
    - "Preface ix"
    - "xii Table of Contents"
    - "Table of Contents xiii"
    - "List of Figures and Tables xix"
    - "xix List of Figures and Tables"
    """
    stripped = line.strip()
    if not stripped:
        return False
    
    # Pattern: NUMBER + "The Standard for Project Management"
    if RUNNING_HEADER_REGEX.match(stripped):
        return True
    
    # Pattern: Roman numeral + "Preface" or "Preface" + Roman numeral
    if re.match(r"^[ivx]+\s+Preface$", stripped, re.IGNORECASE):
        return True
    if re.match(r"^Preface\s+[ivx]+$", stripped, re.IGNORECASE):
        return True
    
    # Pattern: Roman numeral + "Table of Contents" or vice versa
    if re.match(r"^[ivx]+\s+Table of Contents$", stripped, re.IGNORECASE):
        return True
    if re.match(r"^Table of Contents\s+[ivx]+$", stripped, re.IGNORECASE):
        return True
    
    # Pattern: Roman numeral + "List of Figures and Tables" or vice versa
    if re.match(r"^[ivx]+\s+List of Figures and Tables$", stripped, re.IGNORECASE):
        return True
    if re.match(r"^List of Figures and Tables\s+[ivx]+$", stripped, re.IGNORECASE):
        return True
    
    return False


def is_repeated_section_header(line: str) -> bool:
    """
    Check if line is a section header that appears as running header.
    
    These are lines like "Section 2 A System for Value Delivery" that appear
    repeatedly at page boundaries. They match the pattern:
    - "Section X Title" where X is a single digit and Title is text
    - NOT "Section X.Y" which are actual content headings
    
    Examples:
    - "Section 2 A System for Value Delivery"
    - "Section 3 Project Management Principles"
    - "Section 4 Project Life Cycles"
    """
    stripped = line.strip()
    if not stripped:
        return False
    
    # Match "Section X Title" pattern (X is a single digit, no decimal)
    # This distinguishes running headers from actual section headings like "Section 2.1"
    if re.match(r"^Section\s+\d\s+[A-Za-z]", stripped):
        # Make sure it's not a subsection (Section X.Y)
        if not re.match(r"^Section\s+\d+\.\d", stripped):
            return True
    
    return False


def is_pdf_artifact(line: str) -> bool:
    """
    Check if line is a PDF artifact that should be removed.
    
    This includes:
    - Standalone page numbers
    - Running headers
    - Repeated section headers
    """
    return (
        is_page_number(line)
        or is_running_header(line)
        or is_repeated_section_header(line)
    )


def remove_pdf_artifacts(content: str) -> tuple[str, int]:
    """
    Remove PDF artifacts from content.
    
    Returns:
        Tuple of (cleaned content, number of artifacts removed)
    """
    lines = content.split("\n")
    result = []
    artifacts_removed = 0
    
    for line in lines:
        if is_pdf_artifact(line):
            artifacts_removed += 1
            continue
        result.append(line)
    
    return "\n".join(result), artifacts_removed


def fix_section_headers_with_page_numbers(content: str) -> tuple[str, int]:
    """
    Fix section headers that have trailing page numbers.
    
    Example: "Section 1 – Introduction 5" -> "Section 1 – Introduction"
    
    Returns:
        Tuple of (fixed content, number of fixes applied)
    """
    lines = content.split("\n")
    result = []
    fixes = 0
    
    for line in lines:
        match = SECTION_HEADER_WITH_PAGE_REGEX.match(line.strip())
        if match:
            # Replace with the header without trailing page number
            fixed_line = match.group(1)
            result.append(fixed_line)
            fixes += 1
        else:
            result.append(line)
    
    return "\n".join(result), fixes


def fix_broken_bold_headers(content: str) -> tuple[str, int]:
    """
    Fix bold headers where the next word is attached without a space.
    
    Example: "**1.1 Structure of the PMBOK® GuideAffinity diagram.**" 
             -> "**1.1 Structure of the PMBOK® Guide Affinity diagram.**"
    
    Returns:
        Tuple of (fixed content, number of fixes applied)
    """
    # Use re.sub with a function to insert space between Guide and the next capitalized word
    def replace_func(match):
        full_match = match.group(0)  # The full match like "**1.1 Structure of the PMBOK® GuideAffinity diagram.**"
        # Insert space before the capitalized word that follows "Guide"
        return re.sub(r'(Guide)([A-Z][a-z])', r'\1 \2', full_match)
    
    # Count matches before replacement
    pattern = re.compile(r'\*\*.*?Guide[A-Z].*?\.\*\*')
    matches = pattern.findall(content)
    
    if not matches:
        return content, 0
    
    # Replace each occurrence
    fixed_content = pattern.sub(replace_func, content)
    
    fixes = len(matches)
    return fixed_content, fixes


def remove_standalone_overview_lines(content: str) -> tuple[str, int]:
    """
    Remove standalone "overview of the content." lines that are PDF artifacts.
    
    These appear as isolated lines with no surrounding context, indicating they
    are PDF artifacts from page breaks.
    
    Returns:
        Tuple of (fixed content, number of lines removed)
    """
    lines = content.split("\n")
    result = []
    removed = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if STANDALONE_OVERVIEW_REGEX.match(stripped):
            # Check if it's standalone (surrounded by blank lines or at file boundaries)
            prev_blank = not lines[i-1].strip() if i > 0 else True
            next_blank = not lines[i+1].strip() if i < len(lines)-1 else True
            
            if prev_blank or next_blank:
                removed += 1
                continue
        
        result.append(line)
    
    return "\n".join(result), removed


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
    
    # Phase 1: Remove PDF artifacts (page numbers, running headers)
    print("Removing PDF artifacts...")
    cleaned_content, artifacts_removed = remove_pdf_artifacts(content)
    print(f"PDF artifacts removed: {artifacts_removed}")
    
    # Phase 2: Fix section headers with trailing page numbers
    print("Fixing section headers with trailing page numbers...")
    cleaned_content, section_fixes = fix_section_headers_with_page_numbers(cleaned_content)
    print(f"Section header page numbers removed: {section_fixes}")
    
    # Phase 3: Fix broken bold headers
    print("Fixing broken bold headers...")
    cleaned_content, bold_fixes = fix_broken_bold_headers(cleaned_content)
    print(f"Broken bold headers fixed: {bold_fixes}")
    
    # Phase 4: Remove standalone "overview of the content." lines
    print("Removing standalone overview lines...")
    cleaned_content, overview_removed = remove_standalone_overview_lines(cleaned_content)
    print(f"Standalone overview lines removed: {overview_removed}")
    
    # Count lines after artifact removal
    cleaned_lines = len(cleaned_content.split("\n"))
    print(f"Lines after artifact removal: {cleaned_lines}")
    
    # Phase 5: Normalize paragraphs
    print("Normalizing paragraphs...")
    normalized_content = normalize_paragraphs(cleaned_content)
    
    # Count normalized lines
    normalized_lines = len(normalized_content.split("\n"))
    print(f"Normalized line count: {normalized_lines}")
    print(f"Total lines reduced: {original_lines - normalized_lines}")
    
    # Write back
    print(f"Writing normalized content to {markdown_path}...")
    markdown_path.write_text(normalized_content, encoding="utf-8")
    
    print("Done!")


if __name__ == "__main__":
    main()
