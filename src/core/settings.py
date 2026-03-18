#!/usr/bin/env python3
"""
Settings Module - Configuration Loading and Validation

Provides centralized configuration management with validation,
typing, and environment variable substitution.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class SettingsError(Exception):
    """Base exception for settings-related errors."""
    pass


class ConfigurationError(SettingsError):
    """Raised when configuration is invalid or missing required fields."""
    pass


class ValidationError(SettingsError):
    """Raised when configuration validation fails."""
    pass


def _substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in configuration values.
    
    Supports ${VAR_NAME} syntax. If the variable is not set and no
    default is provided, the original placeholder is kept.
    
    Args:
        value: Configuration value (string, dict, list, etc.)
        
    Returns:
        Value with environment variables substituted
    """
    if isinstance(value, str):
        pattern = r'\$\{([^}]+)\}'
        
        def replace(match: re.Match) -> str:
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default = var_expr.split(':', 1)
            else:
                var_name, default = var_expr, None
            
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            else:
                # Return original placeholder if not set
                return match.group(0)
        
        return re.sub(pattern, replace, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    else:
        return value


@dataclass
class LLMSettings:
    """LLM configuration settings."""
    provider: str
    openai: dict[str, Any] = field(default_factory=dict)
    dashscope: dict[str, Any] = field(default_factory=dict)
    anthropic: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionLLMSettings:
    """Vision LLM configuration settings."""
    provider: str
    enabled: bool = False
    openai: dict[str, Any] = field(default_factory=dict)
    dashscope: dict[str, Any] = field(default_factory=dict)
    anthropic: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingSettings:
    """Embedding configuration settings."""
    provider: str
    openai: dict[str, Any] = field(default_factory=dict)
    dashscope: dict[str, Any] = field(default_factory=dict)
    anthropic: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreSettings:
    """Vector store configuration settings."""
    provider: str
    chroma: dict[str, Any] = field(default_factory=dict)
    qdrant: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitterSettings:
    """Text splitter configuration settings."""
    strategy: str
    recursive: dict[str, Any] = field(default_factory=dict)
    semantic: dict[str, Any] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalSettings:
    """Retrieval configuration settings."""
    dense: dict[str, Any] = field(default_factory=dict)
    sparse: dict[str, Any] = field(default_factory=dict)
    hybrid: dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerSettings:
    """Reranker configuration settings."""
    backend: str
    cross_encoder: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionSettings:
    """Ingestion pipeline configuration settings."""
    batch_size: int = 32
    integrity_check: dict[str, Any] = field(default_factory=dict)
    images: dict[str, Any] = field(default_factory=dict)
    chunk_refiner: dict[str, Any] = field(default_factory=dict)
    metadata_enricher: dict[str, Any] = field(default_factory=dict)
    image_captioner: dict[str, Any] = field(default_factory=dict)


@dataclass
class BM25Settings:
    """BM25 index configuration settings."""
    index_path: str = "data/db/bm25"
    k1: float = 1.5
    b: float = 0.75


@dataclass
class EvaluationSettings:
    """Evaluation configuration settings."""
    provider: str
    ragas: dict[str, Any] = field(default_factory=dict)
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservabilitySettings:
    """Observability configuration settings."""
    logging: dict[str, Any] = field(default_factory=dict)
    tracing: dict[str, Any] = field(default_factory=dict)
    dashboard: dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageSettings:
    """Storage configuration settings."""
    data_dir: str = "data"
    db_dir: str = "data/db"
    image_dir: str = "data/images"
    document_dir: str = "data/documents"


@dataclass
class Settings:
    """
    Main settings container for the Smart Knowledge Hub.
    
    This dataclass holds all configuration settings for the application.
    It is populated by loading and parsing the YAML configuration file.
    """
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    splitter: SplitterSettings
    retrieval: RetrievalSettings
    reranker: RerankerSettings
    ingestion: IngestionSettings
    bm25: BM25Settings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    storage: StorageSettings
    vision_llm: Optional[VisionLLMSettings] = None


def load_settings(path: Union[str, Path]) -> Settings:
    """
    Load settings from a YAML configuration file.
    
    Args:
        path: Path to the YAML configuration file
        
    Returns:
        Settings object with all configuration loaded
        
    Raises:
        ConfigurationError: If the file cannot be read or parsed
    """
    path = Path(path)
    
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to read configuration file: {e}")
    
    if raw_config is None:
        raise ConfigurationError("Configuration file is empty")
    
    # Substitute environment variables
    config = _substitute_env_vars(raw_config)
    
    # Validate required fields
    validate_settings(config)
    
    # Build Settings object
    settings = Settings(
        llm=LLMSettings(**config['llm']),
        vision_llm=VisionLLMSettings(**config.get('vision_llm', {})) if 'vision_llm' in config else None,
        embedding=EmbeddingSettings(**config['embedding']),
        vector_store=VectorStoreSettings(**config['vector_store']),
        splitter=SplitterSettings(**config['splitter']),
        retrieval=RetrievalSettings(**config['retrieval']),
        reranker=RerankerSettings(**config['reranker']),
        ingestion=IngestionSettings(**config['ingestion']),
        bm25=BM25Settings(**config['bm25']),
        evaluation=EvaluationSettings(**config['evaluation']),
        observability=ObservabilitySettings(**config['observability']),
        storage=StorageSettings(**config['storage']),
    )
    
    logger.info(f"Settings loaded from: {path}")
    return settings


def validate_settings(config: dict[str, Any]) -> None:
    """
    Validate that required configuration fields are present.
    
    Args:
        config: Parsed configuration dictionary
        
    Raises:
        ValidationError: If required fields are missing
    """
    required_nested_fields = [
        ('llm', 'provider'),
        ('embedding', 'provider'),
        ('vector_store', 'provider'),
        ('splitter', 'strategy'),
    ]
    
    missing = []
    
    for section, field in required_nested_fields:
        if section not in config:
            missing.append(f"{section}.{field}")
        elif not isinstance(config[section], dict):
            missing.append(f"{section}.{field}")
        elif field not in config[section]:
            missing.append(f"{section}.{field}")
        else:
            value = config[section][field]
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(f"{section}.{field}")
    
    if missing:
        raise ValidationError(
            f"Missing required configuration fields: {', '.join(missing)}"
        )
