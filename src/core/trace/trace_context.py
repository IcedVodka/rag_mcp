#!/usr/bin/env python3
"""
Trace Context - Request Tracing and Observability

Provides trace_id generation, stage recording, and timing for
both ingestion and query pipelines.
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TraceContext:
    """
    Context for tracing request flow through the system.
    
    Used to track timing, stages, and metadata for both
    ingestion and query operations.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_type: str = "query"  # "query" or "ingestion"
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    
    def record_stage(
        self,
        name: str,
        method: Optional[str] = None,
        provider: Optional[str] = None,
        details: Optional[dict] = None
    ) -> None:
        """
        Record a stage in the trace.
        
        Args:
            name: Stage name (e.g., "dense_retrieval", "rerank")
            method: Method/strategy used
            provider: Provider/backend used
            details: Additional details
        """
        stage = {
            "name": name,
            "timestamp": time.time(),
            "elapsed_ms": self.elapsed_ms(),
        }
        if method is not None:
            stage["method"] = method
        if provider is not None:
            stage["provider"] = provider
        if details is not None:
            stage["details"] = details
        self.stages.append(stage)
    
    def finish(self) -> None:
        """Mark the trace as finished."""
        self.finished_at = time.time()
    
    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """
        Get elapsed time in milliseconds.
        
        Args:
            stage_name: Optional stage name to get elapsed time up to
            
        Returns:
            Elapsed time in milliseconds
        """
        if stage_name:
            for stage in self.stages:
                if stage["name"] == stage_name:
                    return stage["elapsed_ms"]
            return 0.0
        
        end_time = self.finished_at or time.time()
        return (end_time - self.started_at) * 1000
    
    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        result = {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.elapsed_ms(),
            "stages": self.stages,
        }
        return result
