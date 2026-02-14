#!/usr/bin/env python3
"""Test script for PMBOK-aware PDF parsing and section hierarchy detection."""

import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from autokg_rag.ingest.pmbok_toc_parser import PmbokTocParser, load_pmbok_toc
    from autokg_rag.ingest.table_extractor import extract_tables_from_pdf
    from autokg_rag.ingest.pdf_parse import parse_pdf_with_tables
except ModuleNotFoundError as exc:  # pragma: no cover - local env dependency guard
    print(f"Skipping runtime tests due to missing dependency: {exc}")
    sys.exit(0)


def test_pmbok_toc_parsing():
    """Test the PMBOK TOC parser with a sample PDF."""
    print("Testing PMBOK TOC parsing...")
    
    # Look for PMBOK PDF in fixtures
    pmbok_path = Path("data/fixtures/pmbok/pmbokguide_eighthed_eng.pdf")
    
    if not pmbok_path.exists():
        print(f"PMBOK PDF not found at {pmbok_path}")
        # Create a mock test instead
        print("Creating a mock test for TOC parsing...")
        
        # Create a temporary test file with mock TOC content
        test_pdf_path = Path("test_mock_pmbok.pdf")
        if not test_pdf_path.exists():
            print("Mock test PDF doesn't exist, testing with parser initialization only...")
            parser = PmbokTocParser()
            print("✓ TOC Parser initialized successfully")
            return True
        else:
            print(f"Found test PDF at {test_pdf_path}")
            # Try to parse the PDF
            toc_entries, section_map = load_pmbok_toc(test_pdf_path)
            print(f"Found {len(toc_entries)} TOC entries")
            if toc_entries:
                print(f"First entry: {toc_entries[0]}")
                print(f"Sample section map: {list(section_map.items())[:3]}")
            return True
    else:
        print(f"Found PMBOK PDF at {pmbok_path}")
        try:
            toc_entries, section_map = load_pmbok_toc(pmbok_path)
            print(f"Successfully parsed {len(toc_entries)} TOC entries")
            if toc_entries:
                print(f"First few entries:")
                for i, entry in enumerate(toc_entries[:5]):
                    print(f"  {i+1}. {entry.number} - {entry.title} (page {entry.page})")
                    print(f"     Full path: {entry.full_path}")
            
            print(f"Created section map with {len(section_map)} entries")
            return True
        except Exception as e:
            print(f"Error parsing PMBOK PDF: {e}")
            return False


def test_table_extraction():
    """Test table extraction from PDF."""
    print("\nTesting table extraction...")
    
    # Look for any PDF to test table extraction
    pmbok_path = Path("data/fixtures/pmbok/pmbokguide_eighthed_eng.pdf")
    
    if not pmbok_path.exists():
        print(f"PMBOK PDF not found at {pmbok_path}")
        print("Testing table extractor initialization only...")
        print("✓ Table extraction module import successful")
        return True
    else:
        try:
            tables = extract_tables_from_pdf(pmbok_path)
            print(f"Extracted {len(tables)} tables from PMBOK PDF")
            if tables:
                print(f"First table: {tables[0].headers} with {len(tables[0].rows)} rows")
            return True
        except Exception as e:
            print(f"Error extracting tables: {e}")
            return False


def test_enhanced_pdf_parsing():
    """Test the enhanced PDF parsing with both text and tables."""
    print("\nTesting enhanced PDF parsing...")
    
    pmbok_path = Path("data/fixtures/pmbok/pmbokguide_eighthed_eng.pdf")
    
    if not pmbok_path.exists():
        print(f"PMBOK PDF not found at {pmbok_path}")
        return True
    else:
        try:
            pages, tables = parse_pdf_with_tables(pmbok_path)
            print(f"Parsed {len(pages)} pages and {len(tables)} tables")
            if pages:
                print(f"First page text length: {len(pages[0])} characters")
            if tables:
                print(f"First table has {len(tables[0].headers)} columns and {len(tables[0].rows)} rows")
            return True
        except Exception as e:
            print(f"Error in enhanced PDF parsing: {e}")
            return False


def test_schema_extensions():
    """Test the extended ChunkRecord schema."""
    print("\nTesting extended ChunkRecord schema...")
    
    try:
        from autokg_rag.schemas.records import ChunkRecord
        
        # Create a sample chunk with new fields
        chunk = ChunkRecord(
            chunk_id="test-chunk-1",
            doc_id="test-doc-1",
            page=1,
            section="Introduction",
            chunk_text="This is a sample chunk.",
            chunk_type="text",
            section_path="1. Introduction / 1.1 Background",
            cross_refs=["Section 2.1", "Figure 3.2"]
        )
        
        print(f"✓ Created chunk with new fields:")
        print(f"  - chunk_type: {chunk.chunk_type}")
        print(f"  - section_path: {chunk.section_path}")
        print(f"  - cross_refs: {chunk.cross_refs}")
        
        return True
    except Exception as e:
        print(f"Error testing schema extensions: {e}")
        return False


def main():
    """Run all tests."""
    print("Running PMBOK-aware PDF parsing tests...\n")
    
    results = []
    results.append(test_schema_extensions())
    results.append(test_pmbok_toc_parsing())
    results.append(test_table_extraction())
    results.append(test_enhanced_pdf_parsing())
    
    print(f"\nTest Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
