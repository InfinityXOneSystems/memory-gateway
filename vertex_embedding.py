"""Compatibility shim inside the hyphenated folder to expose the real module.

This file re-exports the implementation from services/memory_gateway/vertex_embedding.py
so imports that expect the hyphenated path succeed.
"""
try:
    from services.memory_gateway.vertex_embedding import VertexEmbedder, get_embedder  # type: ignore
except Exception:  # pragma: no cover - best-effort shim
    # If the underscore package isn't available, raise informative import error
    raise

__all__ = ["VertexEmbedder", "get_embedder"]
from typing import List
import os
import logging

from google.cloud import aiplatform

logger = logging.getLogger("vertex-embedding")


class VertexEmbedder:
    def __init__(self):
        project = os.getenv("VERTEX_PROJECT_ID") or os.getenv("FIRESTORE_PROJECT_ID")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        model = os.getenv("VERTEX_EMBEDDING_MODEL", "textembedding-gecko@001")
        self.project = project
        self.location = location
        self.model = model
        # Ensure credentials available
        creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds and not project:
            raise RuntimeError("Google credentials not found: set GOOGLE_APPLICATION_CREDENTIALS or VERTEX_PROJECT_ID")
        try:
            aiplatform.init(project=project, location=location)
        except Exception as e:
            logger.error(f"Failed to init Vertex aiplatform: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Try multiple Vertex client entry points for compatibility across SDK versions
        # 1) aiplatform.gapic.EmbeddingServiceClient (newer)
        # 2) aiplatform.gapic.EmbeddingsClient (older)
        # 3) google.genai client as a last resort
        name = f"projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}"
        # Try aiplatform gapic embedding service
        try:
            gapic = getattr(aiplatform, 'gapic', None)
            if gapic is not None and hasattr(gapic, 'EmbeddingServiceClient'):
                client = gapic.EmbeddingServiceClient()
                # try the newer EmbedContentRequest/EmbedContentResponse pattern
                if hasattr(gapic, 'EmbedContentRequest'):
                    req_cls = getattr(gapic, 'EmbedContentRequest')
                    # build instances list
                    instances = [{'content': t} for t in texts]
                    req = req_cls(model=name, instances=instances)
                    resp = client.embed_content(request=req)
                    # resp may contain .responses with embedding values
                    embeddings = []
                    for r in getattr(resp, 'responses', []):
                        if hasattr(r, 'embedding'):
                            embeddings.append(list(r.embedding))
                        elif hasattr(r, 'values'):
                            embeddings.append(list(r.values))
                    if embeddings:
                        return embeddings
                else:
                    # fallback to a generic embed call
                    resp = client.embed(request={"model": name, "instances": texts})
                    embeddings = [inst.values for inst in getattr(resp, 'responses', [])]
                    if embeddings:
                        return embeddings
        except Exception:
            logger.debug('EmbeddingServiceClient attempt failed', exc_info=True)

        try:
            if hasattr(aiplatform, 'gapic') and hasattr(aiplatform.gapic, 'EmbeddingsClient'):
                client = aiplatform.gapic.EmbeddingsClient()
                response = client.embed_text(request={"model": name, "instances": texts})
                embeddings = [inst.embedding for inst in response.predictions]
                return embeddings
        except Exception:
            logger.debug('EmbeddingsClient attempt failed', exc_info=True)

        # Fallback to google-genai if available
        try:
            from google import genai
            client = genai.GenerationClient()
            # genai requires a different flow; try the embeddings API if available
            if hasattr(client, 'embeddings'):
                resp = client.embeddings.create(model=self.model, input=texts)
                return [e.embedding for e in resp.data]
        except Exception:
            logger.debug('google.genai fallback failed', exc_info=True)

        raise RuntimeError('No compatible Vertex embeddings client available')


def get_embedder():
    return VertexEmbedder()
