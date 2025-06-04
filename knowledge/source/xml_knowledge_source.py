import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
import re

from knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class XMLKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries XML file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess XML file content with robust error handling."""
        content_dict = {}
        for file_path in self.safe_file_paths:
            try:
                # First attempt: Standard XML parsing
                tree = ET.parse(file_path)
                root = tree.getroot()
                content = self._extract_xml_content(root)
                content_dict[file_path] = content if content.strip() else "XML document is empty or contains no readable text."
                
            except ET.ParseError as e:
                # Second attempt: Try to fix common XML issues
                try:
                    content = self._parse_malformed_xml(file_path)
                    content_dict[file_path] = content if content.strip() else "XML document processed but contains no readable text."
                except Exception:
                    # Third attempt: Extract as much text as possible
                    try:
                        content = self._extract_text_from_broken_xml(file_path)
                        content_dict[file_path] = f"Warning: XML parsing failed, extracted text content: {content}"
                    except Exception:
                        content_dict[file_path] = f"Error parsing XML file: {str(e)}"
                        
            except FileNotFoundError:
                content_dict[file_path] = f"Error: File '{file_path}' not found"
            except Exception as e:
                content_dict[file_path] = f"Error reading XML document: {str(e)}"
                
        return content_dict

    def _parse_malformed_xml(self, file_path: Path) -> str:
        """Attempt to parse malformed XML by cleaning it first."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        # Remove content after the first complete root element
        # Find the first opening tag
        root_tag_match = re.search(r'<([^/?!\s>]+)', content)
        if root_tag_match:
            root_tag = root_tag_match.group(1)
            # Find the matching closing tag
            closing_tag = f"</{root_tag}>"
            closing_pos = content.find(closing_tag)
            if closing_pos != -1:
                # Cut content at the end of the closing tag
                clean_content = content[:closing_pos + len(closing_tag)]
                
                # Parse the cleaned content
                root = ET.fromstring(clean_content)
                return self._extract_xml_content(root)
        
        raise Exception("Could not clean malformed XML")

    def _extract_text_from_broken_xml(self, file_path: Path) -> str:
        """Extract text content from broken XML using regex."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        # Remove XML tags and extract text content
        text_content = re.sub(r'<[^>]+>', ' ', content)
        # Clean up whitespace
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        return text_content[:1000] + "..." if len(text_content) > 1000 else text_content

    def _extract_xml_content(self, element) -> str:
        """Recursively extract text content from XML elements."""
        content = ""
        
        # Add element tag name and attributes
        if element.tag:
            content += f"{element.tag}: "
        
        # Add attributes if present
        if element.attrib:
            attrs = ", ".join([f"{k}={v}" for k, v in element.attrib.items()])
            content += f"[{attrs}] "
        
        # Add element text
        if element.text and element.text.strip():
            content += element.text.strip() + " "
        
        # Process child elements recursively
        for child in element:
            content += self._extract_xml_content(child)
        
        # Add tail text if present
        if element.tail and element.tail.strip():
            content += element.tail.strip() + " "
        
        content += "\n"
        return content

    def add(self) -> None:
        """
        Add XML file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        content_str = (
            str(self.content) if isinstance(self.content, dict) else self.content
        )
        new_chunks = self._chunk_text(content_str)
        self.chunks.extend(new_chunks)
        self._save_documents()

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]