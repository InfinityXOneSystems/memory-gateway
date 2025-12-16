import os
import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, status, Request
from pydantic import BaseModel, Field
from google.cloud import firestore
from google.cloud import pubsub_v1
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
import google.auth.transport.requests
import google.oauth2.id_token
# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger("memory-gateway")
# --------------------------------------------------
# AUTH
# --------------------------------------------------

GOOGLE_ISSUERS = [
    "https://accounts.google.com",
    "accounts.google.com"
]

security = HTTPBearer()

def verify_google_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        request = google.auth.transport.requests.Request()
        id_info = google.oauth2.id_token.verify_oauth2_token(token, request)
        if id_info["iss"] not in GOOGLE_ISSUERS:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid issuer.")
        # Optionally, check audience, email, etc. here
        return id_info
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")

# --------------------------------------------------
# ENV
# --------------------------------------------------

PROJECT_ID = os.getenv("GCP_PROJECT")
SERVICE_NAME = "memory-gateway"
MEMORY_COL = "infinity_memory"
AUDIT_COL = "infinity_audit"

# --------------------------------------------------
# INIT
# --------------------------------------------------

app = FastAPI(
    title="Infinity XOS Memory Gateway",
    version="1.0.0",
    docs_url="/docs"
)

db = firestore.Client()
publisher = pubsub_v1.PublisherClient() if PROJECT_ID else None

# --------------------------------------------------
# MODELS
# --------------------------------------------------

class MemoryWrite(BaseModel):
    agent_id: str
    scope: str
    importance: int = Field(ge=1, le=10)
    confidence: float = Field(ge=0.0, le=1.0)
    content: Dict[str, Any]
    tags: List[str] = []
    source: Optional[str] = None

class MemoryQuery(BaseModel):
    query: str
    scope: Optional[str] = None
    limit: int = 10

class MemorySummary(BaseModel):
    scope: str
    max_tokens: int = 1200

# --------------------------------------------------
# UTILS
# --------------------------------------------------

def now():
    return datetime.now(timezone.utc)

def audit(event: str, payload: Dict[str, Any]):
    logger.info(f"AUDIT: {event} | {payload}")
    db.collection(AUDIT_COL).add({
        "event": event,
        "payload": payload,
        "timestamp": now(),
        "service": SERVICE_NAME
    })

def publish(topic: str, data: Dict[str, Any]):
    if not publisher:
        logger.warning("Pub/Sub publisher not initialized; skipping publish.")
        return
    try:
        publisher.publish(
            publisher.topic_path(PROJECT_ID, topic),
            str(data).encode("utf-8")
        )
        logger.info(f"Published to topic {topic}: {data}")
    except Exception as e:
        logger.error(f"Failed to publish to topic {topic}: {e}")

# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
def health():
    logger.info("Health check called.")
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "time": now().isoformat()
    }

# --------------------------------------------------
# WRITE MEMORY
# --------------------------------------------------

@app.post("/memory/write")
def write_memory(m: MemoryWrite, user=Depends(verify_google_token)):
    mem_id = str(uuid.uuid4())
    record = {
        "id": mem_id,
        "agent_id": m.agent_id,
        "scope": m.scope,
        "importance": m.importance,
        "confidence": m.confidence,
        "content": m.content,
        "tags": m.tags,
        "source": m.source,
        "created_at": now(),
        "rotatable": True
    }
    try:
        db.collection(MEMORY_COL).document(mem_id).set(record)
        logger.info(f"Memory written: {mem_id} by {m.agent_id}")
    except Exception as e:
        logger.error(f"Failed to write memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to write memory.")
    audit("memory_write", record)
    publish("memory-write", record)
    return {"id": mem_id, "status": "stored"}

# --------------------------------------------------
# SEARCH MEMORY
# --------------------------------------------------

@app.post("/memory/search")
def search_memory(q: MemoryQuery, user=Depends(verify_google_token)):
    ref = db.collection(MEMORY_COL)
    if q.scope:
        ref = ref.where("scope", "==", q.scope)

    results = []
    try:
        for d in ref.limit(q.limit).stream():
            data = d.to_dict()
            if q.query.lower() in str(data.get("content", "")).lower():
                results.append(data)
        logger.info(f"Memory search: query='{q.query}' scope='{q.scope}' count={len(results)}")
    except Exception as e:
        logger.error(f"Failed to search memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to search memory.")
    audit("memory_search", {"query": q.query, "count": len(results)})
    return {"results": results}

# --------------------------------------------------
# SUMMARIZE MEMORY
# --------------------------------------------------

@app.post("/memory/summarize")
def summarize(req: MemorySummary, user=Depends(verify_google_token)):
    try:
        docs = list(
            db.collection(MEMORY_COL)
            .where("scope", "==", req.scope)
            .stream()
        )
        if not docs:
            logger.warning(f"No memory found for scope: {req.scope}")
            raise HTTPException(404, "No memory found")
        combined = " ".join(str(d.to_dict()["content"]) for d in docs)
        summary = combined[:req.max_tokens]
        db.collection(MEMORY_COL).add({
            "agent_id": "memory-gateway",
            "scope": req.scope,
            "importance": 10,
            "confidence": 1.0,
            "rotatable": False,
            "content": {"summary": summary},
            "created_at": now()
        })
        logger.info(f"Memory summarized for scope: {req.scope}")
    except Exception as e:
        logger.error(f"Failed to summarize memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to summarize memory.")
    audit("memory_summary", {"scope": req.scope})
    publish("memory-summary", {"scope": req.scope})
    return {"status": "summarized"}

# --------------------------------------------------
# REHYDRATE
# --------------------------------------------------

@app.post("/memory/rehydrate")
def rehydrate(user=Depends(verify_google_token)):
    try:
        memories = [
            d.to_dict()
            for d in db.collection(MEMORY_COL)
            .where("rotatable", "==", False)
            .stream()
        ]
        logger.info(f"Rehydrated system memory count: {len(memories)}")
    except Exception as e:
        logger.error(f"Failed to rehydrate memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to rehydrate memory.")
    audit("rehydrate", {"count": len(memories)})
    return {"system_memory": memories}

# --------------------------------------------------
# PRUNE
# --------------------------------------------------

@app.post("/memory/prune")
def prune(scope: str, hours: int = 720, user=Depends(verify_google_token)):
    cutoff = now().timestamp() - hours * 3600
    removed = 0
    try:
        for d in db.collection(MEMORY_COL).where("scope", "==", scope).stream():
            rec = d.to_dict()
            if rec["importance"] < 5 and rec["created_at"].timestamp() < cutoff:
                d.reference.delete()
                removed += 1
        logger.info(f"Pruned {removed} records from scope: {scope}")
    except Exception as e:
        logger.error(f"Failed to prune memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to prune memory.")
    audit("prune", {"scope": scope, "removed": removed})
    return {"removed": removed}
