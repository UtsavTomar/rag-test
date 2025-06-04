from pathlib import Path
from typing import Dict, List

from knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class YAMLKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries YAML file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess YAML file content."""
        yaml = self._import_yaml()

        content = {}

        for path in self.safe_file_paths:
            text = ""
            path = self.convert_to_path(path)
            with open(path, 'r', encoding='utf-8') as file:
                yaml_data = yaml.safe_load(file)
                text = self._extract_text_from_yaml(yaml_data)
            content[path] = text
        return content

    def _import_yaml(self):
        """Dynamically import PyYAML."""
        try:
            import yaml

            return yaml
        except ImportError:
            raise ImportError(
                "PyYAML is not installed. Please install it with: pip install PyYAML"
            )

    def _extract_text_from_yaml(self, data) -> str:
        """Extract text content from YAML data structure."""
        text = ""
        
        if isinstance(data, dict):
            for key, value in data.items():
                text += f"{key}: "
                text += self._extract_text_from_yaml(value)
                text += "\n"
        elif isinstance(data, list):
            for item in data:
                text += self._extract_text_from_yaml(item)
                text += "\n"
        elif data is not None:
            text += str(data) + " "
        
        return text

    def add(self) -> None:
        """
        Add YAML file content to the knowledge source, chunk it, compute embeddings,
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