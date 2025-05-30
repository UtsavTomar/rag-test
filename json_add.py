from knowledge.storage.knowledge_storage import KnowledgeStorage
from knowledge.source.json_knowledge_source import JSONKnowledgeSource
from datetime import datetime
from uuid import uuid4

# Or use a meaningful prefix
collection_name = f"dataset_{datetime.now().strftime('%Y%m%d')}_{str(uuid4())[:8]}"
print(collection_name)
knowledge_storage = KnowledgeStorage(collection_name=collection_name)
knowledge_storage.initialize_knowledge_storage()
json_source = JSONKnowledgeSource(
    file_paths=["all-datasets.json"],
    storage=knowledge_storage
)

json_source.load_content()
json_source.add()

