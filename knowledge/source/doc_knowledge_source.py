from pathlib import Path
from typing import Dict, List

from knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class DOCKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries DOCX file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess DOCX file content."""
        docx = self._import_docx()

        content = {}

        for path in self.safe_file_paths:
            text = ""
            path = self.convert_to_path(path)
            doc = docx.Document(path)
            for paragraph in doc.paragraphs:
                if paragraph.text:
                    text += paragraph.text + "\n"
            content[path] = text
        return content

    def _import_docx(self):
        """Dynamically import python-docx."""
        try:
            import docx

            return docx
        except ImportError:
            raise ImportError(
                "python-docx is not installed. Please install it with: pip install python-docx"
            )

    def add(self) -> None:
        """
        Add DOCX file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        for _, text in self.content.items():
            new_chunks = self._chunk_text(text)
            self.chunks.extend(new_chunks)
        self._save_documents()

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]