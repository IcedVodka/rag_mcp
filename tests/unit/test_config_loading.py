#!/usr/bin/env python3
"""
Configuration Loading Tests

Tests for settings loading, validation, and environment variable substitution.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.settings import (
    load_settings,
    validate_settings,
    _substitute_env_vars,
    Settings,
    ConfigurationError,
    ValidationError,
)


class TestEnvVarSubstitution:
    """Test environment variable substitution."""
    
    def test_substitute_simple_var(self, monkeypatch) -> None:
        """Test simple environment variable substitution."""
        monkeypatch.setenv('TEST_VAR', 'test_value')
        result = _substitute_env_vars('${TEST_VAR}')
        assert result == 'test_value'
    
    def test_substitute_with_default(self, monkeypatch) -> None:
        """Test substitution with default value."""
        result = _substitute_env_vars('${UNSET_VAR:default_value}')
        assert result == 'default_value'
    
    def test_substitute_unset_var(self) -> None:
        """Test that unset variables keep placeholder."""
        result = _substitute_env_vars('${DEFINITELY_UNSET_VAR_12345}')
        assert result == '${DEFINITELY_UNSET_VAR_12345}'
    
    def test_substitute_in_dict(self, monkeypatch) -> None:
        """Test substitution within dictionary."""
        monkeypatch.setenv('API_KEY', 'secret123')
        config = {
            'api_key': '${API_KEY}',
            'other': 'value'
        }
        result = _substitute_env_vars(config)
        assert result['api_key'] == 'secret123'
        assert result['other'] == 'value'
    
    def test_substitute_in_list(self, monkeypatch) -> None:
        """Test substitution within list."""
        monkeypatch.setenv('ITEM', 'item_value')
        config = ['${ITEM}', 'static']
        result = _substitute_env_vars(config)
        assert result == ['item_value', 'static']
    
    def test_substitute_nested(self, monkeypatch) -> None:
        """Test substitution in nested structures."""
        monkeypatch.setenv('OUTER', 'outer_val')
        monkeypatch.setenv('INNER', 'inner_val')
        config = {
            'level1': {
                'level2': ['${OUTER}', '${INNER}']
            }
        }
        result = _substitute_env_vars(config)
        assert result['level1']['level2'] == ['outer_val', 'inner_val']


class TestValidateSettings:
    """Test settings validation."""
    
    def test_valid_config(self) -> None:
        """Test validation with valid config."""
        config = {
            'llm': {'provider': 'openai'},
            'embedding': {'provider': 'openai'},
            'vector_store': {'provider': 'chroma'},
            'splitter': {'strategy': 'recursive'},
        }
        # Should not raise
        validate_settings(config)
    
    def test_missing_llm_provider(self) -> None:
        """Test validation fails when llm.provider is missing."""
        config = {
            'llm': {},
            'embedding': {'provider': 'openai'},
            'vector_store': {'provider': 'chroma'},
            'splitter': {'strategy': 'recursive'},
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_settings(config)
        assert 'llm.provider' in str(exc_info.value)
    
    def test_missing_embedding_provider(self) -> None:
        """Test validation fails when embedding.provider is missing."""
        config = {
            'llm': {'provider': 'openai'},
            'embedding': {},
            'vector_store': {'provider': 'chroma'},
            'splitter': {'strategy': 'recursive'},
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_settings(config)
        assert 'embedding.provider' in str(exc_info.value)
    
    def test_missing_vector_store_provider(self) -> None:
        """Test validation fails when vector_store.provider is missing."""
        config = {
            'llm': {'provider': 'openai'},
            'embedding': {'provider': 'openai'},
            'vector_store': {},
            'splitter': {'strategy': 'recursive'},
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_settings(config)
        assert 'vector_store.provider' in str(exc_info.value)
    
    def test_missing_splitter_strategy(self) -> None:
        """Test validation fails when splitter.strategy is missing."""
        config = {
            'llm': {'provider': 'openai'},
            'embedding': {'provider': 'openai'},
            'vector_store': {'provider': 'chroma'},
            'splitter': {},
        }
        with pytest.raises(ValidationError) as exc_info:
            validate_settings(config)
        assert 'splitter.strategy' in str(exc_info.value)


class TestLoadSettings:
    """Test loading settings from file."""
    
    def test_load_valid_config(self, tmp_path) -> None:
        """Test loading a valid configuration file."""
        config_content = """
llm:
  provider: openai
  openai:
    api_key: test-key
    model: gpt-4

embedding:
  provider: openai
  openai:
    model: text-embedding-3-small

vector_store:
  provider: chroma
  chroma:
    persist_directory: data/db/chroma

splitter:
  strategy: recursive
  recursive:
    chunk_size: 1000

retrieval:
  dense:
    top_k: 20
  sparse:
    top_k: 20
  hybrid:
    rrf_k: 60

reranker:
  backend: none

ingestion:
  batch_size: 32

bm25:
  index_path: data/db/bm25
  k1: 1.5
  b: 0.75

evaluation:
  provider: custom

observability:
  logging:
    level: INFO
  tracing:
    enabled: true

storage:
  data_dir: data
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        settings = load_settings(config_file)
        
        assert isinstance(settings, Settings)
        assert settings.llm.provider == "openai"
        assert settings.embedding.provider == "openai"
        assert settings.vector_store.provider == "chroma"
        assert settings.splitter.strategy == "recursive"
        assert settings.reranker.backend == "none"
    
    def test_file_not_found(self) -> None:
        """Test loading from non-existent file raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_settings("/nonexistent/path/config.yaml")
        assert "not found" in str(exc_info.value).lower()
    
    def test_empty_file(self, tmp_path) -> None:
        """Test loading empty file raises error."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        
        with pytest.raises(ConfigurationError) as exc_info:
            load_settings(config_file)
        assert "empty" in str(exc_info.value).lower()
    
    def test_invalid_yaml(self, tmp_path) -> None:
        """Test loading invalid YAML raises error."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(ConfigurationError) as exc_info:
            load_settings(config_file)
        assert "parse" in str(exc_info.value).lower() or "yaml" in str(exc_info.value).lower()
    
    def test_env_var_substitution_in_file(self, tmp_path, monkeypatch) -> None:
        """Test environment variables are substituted when loading."""
        monkeypatch.setenv('TEST_API_KEY', 'substituted_key')
        
        config_content = """
llm:
  provider: anthropic
  anthropic:
    api_key: ${TEST_API_KEY}
    model: claude-3-5-sonnet-20241022
    base_url: https://api.anthropic.com/v1

embedding:
  provider: dashscope
  dashscope:
    api_key: test-key
    model: text-embedding-v4
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1

vector_store:
  provider: chroma

splitter:
  strategy: recursive

retrieval:
  dense:
    top_k: 20
  sparse:
    top_k: 20
  hybrid:
    rrf_k: 60

reranker:
  backend: none

ingestion:
  batch_size: 32

bm25:
  index_path: data/db/bm25
  k1: 1.5
  b: 0.75

evaluation:
  provider: custom

observability:
  logging:
    level: INFO

storage:
  data_dir: data
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        settings = load_settings(config_file)
        
        assert settings.llm.anthropic['api_key'] == 'substituted_key'


class TestSettingsDataclass:
    """Test Settings dataclass structure."""
    
    def test_settings_structure(self, tmp_path) -> None:
        """Test that Settings has expected attributes."""
        config_content = """
llm:
  provider: openai
  openai:
    api_key: test-key
    model: gpt-4o
    base_url: https://api.openai.com/v1

embedding:
  provider: openai
  openai:
    api_key: test-key
    model: text-embedding-3-small
    base_url: https://api.openai.com/v1

vector_store:
  provider: qdrant
  qdrant:
    url: http://localhost:6333

splitter:
  strategy: semantic
  semantic:
    chunk_size: 500

retrieval:
  dense:
    top_k: 10
  sparse:
    top_k: 10
  hybrid:
    rrf_k: 60

reranker:
  backend: cross_encoder
  cross_encoder:
    model: cross-encoder/ms-marco

ingestion:
  batch_size: 64

bm25:
  index_path: data/db/bm25
  k1: 1.2
  b: 0.75

evaluation:
  provider: ragas
  ragas:
    metrics:
      - faithfulness

observability:
  logging:
    level: DEBUG
  tracing:
    enabled: true

storage:
  data_dir: /custom/data
  db_dir: /custom/data/db

vision_llm:
  provider: openai
  enabled: true
  openai:
    api_key: test-key
    model: gpt-4o
    base_url: https://api.openai.com/v1
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        settings = load_settings(config_file)
        
        # Test LLM
        assert settings.llm.provider == "openai"
        assert settings.llm.openai['model'] == "gpt-4o"
        
        # Test Embedding
        assert settings.embedding.provider == "openai"
        
        # Test Vector Store
        assert settings.vector_store.provider == "qdrant"
        
        # Test Splitter
        assert settings.splitter.strategy == "semantic"
        
        # Test Reranker
        assert settings.reranker.backend == "cross_encoder"
        
        # Test Ingestion
        assert settings.ingestion.batch_size == 64
        
        # Test BM25
        assert settings.bm25.k1 == 1.2
        
        # Test Evaluation
        assert settings.evaluation.provider == "ragas"
        
        # Test Storage
        assert settings.storage.data_dir == "/custom/data"
        
        # Test Vision LLM
        assert settings.vision_llm is not None
        assert settings.vision_llm.provider == "openai"
        assert settings.vision_llm.enabled is True
