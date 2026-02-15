# PMBOK Structure Parsing Rules

This rule file defines the logic for parsing and interpreting the structure of PMBOK (Project Management Body of Knowledge) markdown files within this project.

## Header Identification Logic

The PMBOK markdown files use a mixed convention for headers. Parsers and agents must recognize the following patterns as structural headers:

### 1. Standard Markdown Headers
Lines starting with one or more hash characters (`#`) are standard headers.
- **Pattern:** `^#{1,6}\s+(.+)$`
- **Example:** `# Introduction`
- **Note:** Be aware of potential OCR/formatting artifacts where text might be concatenated (e.g., `##### IntroductionProject Life Cycles`).

### 2. Bolded Numbered Headers
Lines that are bolded and start with a section number are treated as headers, often representing subsections not marked with `#`.
- **Pattern:** `^\*\*\d+(?:\.\d+)*\s+.*\*\*\s*$`
- **Example:** `**1.1 Purpose of The Standard for Project Management**`
- **Logic:** Treat these as headers. The hierarchy level can be inferred from the depth of the numbering (e.g., `1.1` is level 2, `1.1.1` is level 3) or relative to the preceding standard header.

### 3. Section Dividers
Bolded lines explicitly naming a "Section" indicate high-level divisions.
- **Pattern:** `^\*\*Section\s+\d+\*\*\s*$`
- **Example:** `**Section 2**`
- **Logic:** These represent top-level or near-top-level divisions in the document structure.

## Parsing Hierarchy

When constructing a hierarchical tree or table of contents from the markdown:

1. **Precedence:** Standard Markdown headers (`#`) generally define the primary structure.
2. **Integration:** Bolded Numbered Headers should be nested under the most recent Standard Markdown Header.
3. **Fallback:** If a text block starts with a Bolded Numbered Header, it initiates a new logical section even if a standard markdown header is absent.

## Content Extraction

- **Text Content:** All text following a header (Standard or Bolded) until the next header of equal or higher precedence belongs to that section.
- **Lists:** Standard markdown lists (`-`, `1.`) should be preserved as content within their respective sections.
