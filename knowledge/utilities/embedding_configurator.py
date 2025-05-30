import os
import numpy as np
from typing import Any, Dict, Optional, List, Union, cast
from abc import ABC, abstractmethod


Documents = Union[str, List[str]]
Embeddings = np.ndarray


class EmbeddingFunction(ABC):
    
    @abstractmethod
    def __call__(self, input: Documents) -> Embeddings:
        """
        Embed the input documents and return numpy array
        
        Args:
            input: Single string or list of strings to embed
            
        Returns:
            numpy array of embeddings with shape (n_docs, embedding_dim)
        """
        pass


def validate_embedding_function(embedding_function: EmbeddingFunction) -> bool:
    """Validate that the embedding function works correctly"""
    try:
        test_result = embedding_function(["test"])
        return isinstance(test_result, np.ndarray) and len(test_result.shape) == 2
    except Exception:
        return False


class EmbeddingConfigurator:
    def __init__(self):
        self.embedding_functions = {
            "openai": self._configure_openai,
            "azure": self._configure_azure,
            "ollama": self._configure_ollama,
            "vertexai": self._configure_vertexai,
            "google": self._configure_google,
            "cohere": self._configure_cohere,
            "voyageai": self._configure_voyageai,
            "bedrock": self._configure_bedrock,
            "huggingface": self._configure_huggingface,
            "watson": self._configure_watson,
            "custom": self._configure_custom,
        }

    def configure_embedder(
        self,
        embedder_config: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingFunction:
        """Configures and returns an embedding function based on the provided config."""
        if embedder_config is None:
            return self._create_default_embedding_function()

        provider = embedder_config.get("provider")
        config = embedder_config.get("config", {})
        model_name = config.get("model") if provider != "custom" else None

        if provider not in self.embedding_functions:
            raise Exception(
                f"Unsupported embedding provider: {provider}, supported providers: {list(self.embedding_functions.keys())}"
            )

        embedding_function = self.embedding_functions[provider]
        return (
            embedding_function(config)
            if provider == "custom"
            else embedding_function(config, model_name)
        )

    @staticmethod
    def _create_default_embedding_function():
        class OpenAIEmbeddingFunction(EmbeddingFunction):
            def __init__(self, api_key=None, model_name="text-embedding-3-small"):
                try:
                    import openai
                except ImportError:
                    raise ImportError("OpenAI package not installed. Install with: pip install openai")
                
                self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
                self.model_name = model_name
            
            def __call__(self, input: Documents) -> Embeddings:
                if isinstance(input, str):
                    input = [input]
                
                response = self.client.embeddings.create(
                    input=input,
                    model=self.model_name
                )
                
                embeddings = [item.embedding for item in response.data]
                return np.array(embeddings)

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

    @staticmethod
    def _configure_openai(config, model_name):
        class OpenAIEmbeddingFunction(EmbeddingFunction):
            def __init__(self, api_key=None, model_name="text-embedding-3-small", 
                        api_base=None, api_type=None, api_version=None,
                        default_headers=None, dimensions=None, deployment_id=None,
                        organization_id=None):
                try:
                    import openai
                except ImportError:
                    raise ImportError("OpenAI package not installed. Install with: pip install openai")
                
                client_kwargs = {}
                if api_key:
                    client_kwargs['api_key'] = api_key
                if api_base:
                    client_kwargs['base_url'] = api_base
                if organization_id:
                    client_kwargs['organization'] = organization_id
                if default_headers:
                    client_kwargs['default_headers'] = default_headers
                
                self.client = openai.OpenAI(**client_kwargs)
                self.model_name = model_name
                self.dimensions = dimensions
            
            def __call__(self, input: Documents) -> Embeddings:
                if isinstance(input, str):
                    input = [input]
                
                kwargs = {'input': input, 'model': self.model_name}
                if self.dimensions:
                    kwargs['dimensions'] = self.dimensions
                
                response = self.client.embeddings.create(**kwargs)
                embeddings = [item.embedding for item in response.data]
                return np.array(embeddings)

        return OpenAIEmbeddingFunction(
            api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            model_name=model_name,
            api_base=config.get("api_base", None),
            api_type=config.get("api_type", None),
            api_version=config.get("api_version", None),
            default_headers=config.get("default_headers", None),
            dimensions=config.get("dimensions", None),
            deployment_id=config.get("deployment_id", None),
            organization_id=config.get("organization_id", None),
        )

    @staticmethod
    def _configure_azure(config, model_name):
        class AzureOpenAIEmbeddingFunction(EmbeddingFunction):
            def __init__(self, api_key=None, api_base=None, api_type="azure", 
                        api_version=None, model_name=None, default_headers=None,
                        dimensions=None, deployment_id=None, organization_id=None):
                try:
                    import openai
                except ImportError:
                    raise ImportError("OpenAI package not installed. Install with: pip install openai")
                
                self.client = openai.AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=api_base,
                    api_version=api_version,
                    azure_deployment=deployment_id or model_name
                )
                self.model_name = deployment_id or model_name
                self.dimensions = dimensions
            
            def __call__(self, input: Documents) -> Embeddings:
                if isinstance(input, str):
                    input = [input]
                
                kwargs = {'input': input, 'model': self.model_name}
                if self.dimensions:
                    kwargs['dimensions'] = self.dimensions
                
                response = self.client.embeddings.create(**kwargs)
                embeddings = [item.embedding for item in response.data]
                return np.array(embeddings)

        return AzureOpenAIEmbeddingFunction(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            api_type=config.get("api_type", "azure"),
            api_version=config.get("api_version"),
            model_name=model_name,
            default_headers=config.get("default_headers"),
            dimensions=config.get("dimensions"),
            deployment_id=config.get("deployment_id"),
            organization_id=config.get("organization_id"),
        )

    @staticmethod
    def _configure_ollama(config, model_name):
        class OllamaEmbeddingFunction(EmbeddingFunction):
            def __init__(self, url="http://localhost:11434/api/embeddings", model_name=None):
                self.url = url
                self.model_name = model_name
            
            def __call__(self, input: Documents) -> Embeddings:
                import requests
                
                if isinstance(input, str):
                    input = [input]
                
                all_embeddings = []
                for text in input:
                    response = requests.post(
                        self.url,
                        json={"model": self.model_name, "prompt": text}
                    )
                    response.raise_for_status()
                    embedding = response.json()["embedding"]
                    all_embeddings.append(embedding)
                
                return np.array(all_embeddings)

        return OllamaEmbeddingFunction(
            url=config.get("url", "http://localhost:11434/api/embeddings"),
            model_name=model_name,
        )

    @staticmethod
    def _configure_vertexai(config, model_name):
        class GoogleVertexEmbeddingFunction(EmbeddingFunction):
            def __init__(self, model_name=None, api_key=None, project_id=None, region=None):
                try:
                    import google.auth
                    from google.cloud import aiplatform
                except ImportError:
                    raise ImportError("Google Cloud AI Platform not installed. Install with: pip install google-cloud-aiplatform")
                
                self.model_name = model_name
                self.project_id = project_id
                self.region = region
                
                if project_id and region:
                    aiplatform.init(project=project_id, location=region)
            
            def __call__(self, input: Documents) -> Embeddings:
                from google.cloud import aiplatform
                
                if isinstance(input, str):
                    input = [input]
                
                model = aiplatform.TextEmbeddingModel.from_pretrained(self.model_name)
                embeddings = model.get_embeddings(input)
                
                return np.array([emb.values for emb in embeddings])

        return GoogleVertexEmbeddingFunction(
            model_name=model_name,
            api_key=config.get("api_key"),
            project_id=config.get("project_id"),
            region=config.get("region"),
        )

    @staticmethod
    def _configure_google(config, model_name):
        class GoogleGenerativeAiEmbeddingFunction(EmbeddingFunction):
            def __init__(self, model_name=None, api_key=None, task_type=None):
                try:
                    import google.generativeai as genai
                except ImportError:
                    raise ImportError("Google GenerativeAI not installed. Install with: pip install google-generativeai")
                
                genai.configure(api_key=api_key)
                self.model_name = model_name
                self.task_type = task_type
            
            def __call__(self, input: Documents) -> Embeddings:
                import google.generativeai as genai
                
                if isinstance(input, str):
                    input = [input]
                
                embeddings = []
                for text in input:
                    result = genai.embed_content(
                        model=self.model_name,
                        content=text,
                        task_type=self.task_type
                    )
                    embeddings.append(result['embedding'])
                
                return np.array(embeddings)

        return GoogleGenerativeAiEmbeddingFunction(
            model_name=model_name,
            api_key=config.get("api_key"),
            task_type=config.get("task_type"),
        )

    @staticmethod
    def _configure_cohere(config, model_name):
        class CohereEmbeddingFunction(EmbeddingFunction):
            def __init__(self, model_name=None, api_key=None):
                try:
                    import cohere
                except ImportError:
                    raise ImportError("Cohere not installed. Install with: pip install cohere")
                
                self.client = cohere.Client(api_key)
                self.model_name = model_name
            
            def __call__(self, input: Documents) -> Embeddings:
                if isinstance(input, str):
                    input = [input]
                
                response = self.client.embed(
                    texts=input,
                    model=self.model_name
                )
                
                return np.array(response.embeddings)

        return CohereEmbeddingFunction(
            model_name=model_name,
            api_key=config.get("api_key"),
        )

    @staticmethod
    def _configure_voyageai(config, model_name):
        class VoyageAIEmbeddingFunction(EmbeddingFunction):
            def __init__(self, model_name=None, api_key=None):
                try:
                    import voyageai
                except ImportError:
                    raise ImportError("VoyageAI not installed. Install with: pip install voyageai")
                
                self.client = voyageai.Client(api_key=api_key)
                self.model_name = model_name
            
            def __call__(self, input: Documents) -> Embeddings:
                if isinstance(input, str):
                    input = [input]
                
                result = self.client.embed(
                    texts=input,
                    model=self.model_name
                )
                
                return np.array(result.embeddings)

        return VoyageAIEmbeddingFunction(
            model_name=model_name,
            api_key=config.get("api_key"),
        )

    @staticmethod
    def _configure_bedrock(config, model_name):
        class AmazonBedrockEmbeddingFunction(EmbeddingFunction):
            def __init__(self, session=None, model_name=None):
                try:
                    import boto3
                except ImportError:
                    raise ImportError("Boto3 not installed. Install with: pip install boto3")
                
                self.session = session or boto3.Session()
                self.client = self.session.client('bedrock-runtime')
                self.model_name = model_name or "amazon.titan-embed-text-v1"
            
            def __call__(self, input: Documents) -> Embeddings:
                import json
                
                if isinstance(input, str):
                    input = [input]
                
                embeddings = []
                for text in input:
                    body = json.dumps({"inputText": text})
                    response = self.client.invoke_model(
                        body=body,
                        modelId=self.model_name,
                        accept='application/json',
                        contentType='application/json'
                    )
                    
                    result = json.loads(response['body'].read())
                    embeddings.append(result['embedding'])
                
                return np.array(embeddings)

        # Allow custom model_name override with backwards compatibility
        kwargs = {"session": config.get("session")}
        if model_name is not None:
            kwargs["model_name"] = model_name
        return AmazonBedrockEmbeddingFunction(**kwargs)

    @staticmethod
    def _configure_huggingface(config, model_name):
        class HuggingFaceEmbeddingServer(EmbeddingFunction):
            def __init__(self, url=None):
                self.url = url
            
            def __call__(self, input: Documents) -> Embeddings:
                import requests
                
                if isinstance(input, str):
                    input = [input]
                
                response = requests.post(
                    self.url,
                    json={"inputs": input}
                )
                response.raise_for_status()
                
                return np.array(response.json())

        return HuggingFaceEmbeddingServer(
            url=config.get("api_url"),
        )

    @staticmethod
    def _configure_watson(config, model_name):
        try:
            import ibm_watsonx_ai.foundation_models as watson_models
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
        except ImportError as e:
            raise ImportError(
                "IBM Watson dependencies are not installed. Please install them to use Watson embedding."
            ) from e

        class WatsonEmbeddingFunction(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                if isinstance(input, str):
                    input = [input]

                embed_params = {
                    EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
                    EmbedParams.RETURN_OPTIONS: {"input_text": True},
                }

                embedding = watson_models.Embeddings(
                    model_id=config.get("model"),
                    params=embed_params,
                    credentials=Credentials(
                        api_key=config.get("api_key"), url=config.get("api_url")
                    ),
                    project_id=config.get("project_id"),
                )

                try:
                    embeddings = embedding.embed_documents(input)
                    return cast(Embeddings, np.array(embeddings))
                except Exception as e:
                    print("Error during Watson embedding:", e)
                    raise e

        return WatsonEmbeddingFunction()

    @staticmethod
    def _configure_custom(config):
        custom_embedder = config.get("embedder")
        if isinstance(custom_embedder, EmbeddingFunction):
            try:
                validate_embedding_function(custom_embedder)
                return custom_embedder
            except Exception as e:
                raise ValueError(f"Invalid custom embedding function: {str(e)}")
        elif callable(custom_embedder):
            try:
                instance = custom_embedder()
                if isinstance(instance, EmbeddingFunction):
                    validate_embedding_function(instance)
                    return instance
                raise ValueError(
                    "Custom embedder does not create an EmbeddingFunction instance"
                )
            except Exception as e:
                raise ValueError(f"Error instantiating custom embedder: {str(e)}")
        else:
            raise ValueError(
                "Custom embedder must be an instance of `EmbeddingFunction` or a callable that creates one"
            )