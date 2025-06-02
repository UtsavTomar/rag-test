from knowledge.storage.knowledge_storage import KnowledgeStorage
from knowledge.source.json_knowledge_source import JSONKnowledgeSource
from knowledge.source.string_knowledge_source import StringKnowledgeSource
from knowledge.source.crew_docling_source import CrewDoclingSource
from knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
from typing import List
from uuid import uuid4


def json_to_vector(file_paths: List[str]) -> str:
    """
    Convert JSON files to vector database.
    
    Args:
        file_paths: List of JSON file paths to process
        
    Returns:
        database_id: Unique identifier for the created database
    """
    database_id = str(uuid4())
    knowledge_storage = KnowledgeStorage(database_id=database_id)
    knowledge_storage.initialize_knowledge_storage()
    
    json_source = JSONKnowledgeSource(
        file_paths=file_paths,  # Use the parameter instead of hardcoded value
        storage=knowledge_storage
    )

    json_source.load_content()
    json_source.add()

    return database_id

def string_to_vector(content: str) -> str:
    """
    Convert string files to vector database.
    
    Args:
        file_paths: List of string file paths to process
        
    Returns:
        database_id: Unique identifier for the created database
    """
    database_id = str(uuid4())
    knowledge_storage = KnowledgeStorage(database_id=database_id)
    knowledge_storage.initialize_knowledge_storage()
    
    string_source = StringKnowledgeSource(
        content=content,  # Use the parameter instead of hardcoded value
        storage=knowledge_storage
    )

    string_source.add()

    return database_id

def docling_to_vector(file_paths: List[str]) -> str:
    """
    Convert Docling files to vector database.
    
    Args:
        file_paths: List of Docling file paths to process
        
    Returns:
        database_id: Unique identifier for the created database
    """
    database_id = str(uuid4())
    knowledge_storage = KnowledgeStorage(database_id=database_id)
    knowledge_storage.initialize_knowledge_storage()
    
    docling_source = CrewDoclingSource(
        file_paths=file_paths, 
        storage=knowledge_storage
    )

    docling_source.add()

    return database_id

def text_file_to_vector(file_paths: List[str]) -> str:
    """
    Convert text files to vector database.
    
    Args:
        file_paths: List of text file paths to process
        
    Returns:
        database_id: Unique identifier for the created database
    """
    database_id = str(uuid4())
    knowledge_storage = KnowledgeStorage(database_id=database_id)
    knowledge_storage.initialize_knowledge_storage()
    
    text_file_source = TextFileKnowledgeSource(
        file_paths=file_paths, 
        storage=knowledge_storage
    )

    text_file_source.add()

    return database_id

def pdf_to_vector(file_paths: List[str]) -> str:
    """
    Convert PDF files to vector database.
    
    Args:
        file_paths: List of PDF file paths to process
        
    Returns:
        database_id: Unique identifier for the created database
    """
    database_id = str(uuid4())
    knowledge_storage = KnowledgeStorage(database_id=database_id)
    knowledge_storage.initialize_knowledge_storage()
    
    pdf_source = PDFKnowledgeSource(
        file_paths=file_paths, 
        storage=knowledge_storage
    )

    pdf_source.add()

    return database_id

def csv_to_vector(file_paths: List[str]) -> str:
    """
    Convert CSV files to vector database.
    
    Args:
        file_paths: List of CSV file paths to process
        
    Returns:
        database_id: Unique identifier for the created database
    """
    database_id = str(uuid4())
    knowledge_storage = KnowledgeStorage(database_id=database_id)
    knowledge_storage.initialize_knowledge_storage()
    
    csv_source = CSVKnowledgeSource(
        file_paths=file_paths, 
        storage=knowledge_storage
    )

    csv_source.add()

    return database_id

def excel_to_vector(file_paths: List[str]) -> str:
    """
    Convert Excel files to vector database.
    
    Args:
        file_paths: List of Excel file paths to process
        
    Returns:
        database_id: Unique identifier for the created database
    """
    database_id = str(uuid4())
    knowledge_storage = KnowledgeStorage(database_id=database_id)
    knowledge_storage.initialize_knowledge_storage()
    
    excel_source = ExcelKnowledgeSource(
        file_paths=file_paths, 
        storage=knowledge_storage
    )

    excel_source.add()

    return database_id

if __name__ == "__main__":
    file_paths = ["test.xlsx"]
    excel_to_vector(file_paths)
