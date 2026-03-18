#!/usr/bin/env python3
"""
Metadata Enricher - Chunk Metadata Enhancement Transform

Implements a two-stage metadata enrichment process:
1. Rule-based enrichment: Fast pattern-based metadata extraction
2. LLM-based enrichment: Optional intelligent metadata generation via LLM

Features:
- Rule-based title extraction (first line or first 50 chars)
- Rule-based summary generation (truncated first 100 chars)
- Rule-based tag extraction (keyword-based)
- Optional LLM enhancement for semantic title, summary, and tags
- Graceful degradation: falls back to rule results on LLM failure
- Configurable via settings (enable/disable LLM, prompt template path)
- Full trace integration for observability

Metadata Output Format:
{
    "title": str,           # Chunk title
    "summary": str,         # Chunk summary
    "tags": List[str],      # List of tags/keywords
    "enriched_by": str,     # "rule" or "llm"
    "enriched_at": str,     # ISO format timestamp
    "fallback_reason": str  # Optional: reason for fallback
}
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict

from core.types import Chunk
from core.settings import Settings
from core.trace.trace_context import TraceContext
from libs.llm.llm_factory import LLMFactory
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from .base_transform import BaseTransform, TransformConfigError

logger = logging.getLogger(__name__)


class MetadataEnricher(BaseTransform):
    """
    Metadata enrichment transform with rule-based and LLM enhancement.
    
    This transform enriches chunks with metadata through two stages:
    
    1. Rule-based enrichment (always applied as baseline):
       - Title: First line or first 50 characters of chunk
       - Summary: Truncated first 100 characters
       - Tags: Extracted keywords based on simple patterns
    
    2. LLM-based enrichment (optional, configurable):
       - Sends text to LLM for semantic metadata generation
       - Generates intelligent title, summary, and tags
       - Falls back to rule-based results on LLM failure
    
    Configuration options (via settings.ingestion.metadata_enricher):
    - enabled: bool - Whether to enable enrichment (default: True)
    - use_llm: bool - Whether to use LLM enhancement (default: False)
    - prompt_path: str - Path to LLM prompt template file
    
    Example:
        >>> settings = load_settings("config/settings.yaml")
        >>> enricher = MetadataEnricher(settings)
        >>> enriched_chunks = enricher.transform(chunks, trace)
    """
    
    # Common stop words to exclude from tags (English and Chinese)
    STOP_WORDS = {
        # English stop words
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'and', 'but', 'or', 'yet', 'so',
        'if', 'because', 'although', 'though', 'while', 'where', 'when',
        'that', 'which', 'who', 'whom', 'whose', 'what', 'this', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their',
        'there', 'than', 'then', 'only', 'also', 'just', 'even', 'back',
        'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
        'many', 'make', 'over', 'such', 'take', 'year', 'good', 'come',
        'could', 'state', 'most', 'us', 'much', 'know', 'water', 'long',
        'little', 'very', 'after', 'words', 'called', 'just', 'where',
        # Chinese stop words
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
        '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
        '着', '没有', '看', '好', '自己', '这', '那', '这些', '那些', '之',
        '与', '及', '等', '或', '但', '而', '如果', '因为', '所以', '虽然',
        '可以', '需要', '进行', '根据', '通过', '以及', '对于', '关于',
        '使用', '方法', '系统', '部分', '其中', '目前', '已经', '开始',
    }
    
    # Patterns for keyword extraction (rule-based tagging)
    KEYWORD_PATTERNS = [
        # Technical terms (camelCase, PascalCase, snake_case)
        (r'\b[A-Z][a-zA-Z]*(?:[A-Z][a-zA-Z]+)+\b', 'technical'),  # PascalCase (e.g., FastAPI, MySQL)
        (r'\b[a-z]+(?:_[a-z]+)+\b', 'technical'),  # snake_case
        (r'\b[a-z]+(?:[A-Z][a-z]+)+\b', 'technical'),  # camelCase
        # Acronyms
        (r'\b[A-Z]{2,}\b', 'acronym'),
        # Version numbers
        (r'\bv?\d+\.\d+(?:\.\d+)?\b', 'version'),
        # File extensions
        (r'\b\w+\.(?:py|js|ts|java|cpp|c|go|rs|yaml|json|xml|md)\b', 'file'),
        # API/Protocol terms
        (r'\b(?:API|REST|HTTP|HTTPS|JSON|XML|SQL|NoSQL|SDK|CLI|UI|URL)\b', 'protocol'),
    ]
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ) -> None:
        """
        Initialize the metadata enricher.
        
        Args:
            settings: Application settings containing metadata_enricher config
            llm: Optional pre-configured LLM instance. If not provided,
                 will be created from settings if use_llm is enabled.
            prompt_path: Optional path to prompt template. Overrides config.
            
        Raises:
            TransformConfigError: If configuration is invalid
        """
        super().__init__()
        
        self.settings = settings
        
        # Get metadata_enricher config from settings
        enricher_config = getattr(settings.ingestion, 'metadata_enricher', {})
        self.enabled = enricher_config.get('enabled', True)
        self.use_llm = enricher_config.get('use_llm', False)
        
        # Determine prompt path
        self.prompt_path = prompt_path or enricher_config.get('prompt_path')
        self._prompt_template: Optional[str] = None
        
        # Initialize LLM if needed
        self._llm: Optional[BaseLLM] = llm
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(settings)
                logger.info(f"Created LLM for metadata enrichment: {self._llm.model}")
            except Exception as e:
                logger.warning(f"Failed to create LLM for enrichment: {e}. "
                              "Will fall back to rule-based enrichment.")
                self._llm = None
        
        logger.debug(f"MetadataEnricher initialized: enabled={self.enabled}, use_llm={self.use_llm}")
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Transform chunks by enriching their metadata.
        
        Processing flow:
        1. If not enabled, return chunks unchanged
        2. Apply rule-based enrichment to each chunk
        3. If use_llm is enabled, send to LLM for enhancement
        4. On LLM failure, fall back to rule-based result
        
        Args:
            chunks: List of chunks to enrich
            trace: Optional trace context for recording stages
            
        Returns:
            List of enriched chunks
        """
        if not self.enabled:
            logger.debug("MetadataEnricher disabled, returning chunks unchanged")
            return chunks
        
        if not chunks:
            return []
        
        if trace:
            trace.record_stage(
                name="metadata_enrichment",
                method="rule_based" + ("+llm" if self.use_llm else ""),
                provider="MetadataEnricher"
            )
        
        enriched_chunks: List[Chunk] = []
        
        for chunk in chunks:
            try:
                enriched_chunk = self._enrich_chunk(chunk, trace)
                enriched_chunks.append(enriched_chunk)
            except Exception as e:
                logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
                # Fall back to original chunk on error, but still add rule-based metadata
                try:
                    fallback_chunk = self._create_fallback_chunk(chunk)
                    enriched_chunks.append(fallback_chunk)
                except Exception:
                    # Last resort: return original
                    enriched_chunks.append(chunk)
        
        if trace:
            trace.record_stage(
                name="metadata_enrichment_complete",
                details={
                    "input_count": len(chunks),
                    "output_count": len(enriched_chunks),
                    "llm_enabled": self.use_llm and self._llm is not None
                }
            )
        
        return enriched_chunks
    
    def _enrich_chunk(
        self,
        chunk: Chunk,
        trace: Optional[TraceContext] = None
    ) -> Chunk:
        """
        Enrich a single chunk with metadata.
        
        Args:
            chunk: Chunk to enrich
            trace: Optional trace context
            
        Returns:
            Enriched chunk (may be same instance or new)
        """
        # Stage 1: Rule-based enrichment (always applied as baseline)
        rule_based_metadata = self._rule_based_enrich(chunk)
        
        # Stage 2: LLM enhancement (optional, with fallback)
        if self.use_llm and self._llm:
            llm_metadata = self._llm_enrich(chunk, trace)
            if llm_metadata is not None:
                # Merge LLM metadata with rule-based as fallback
                final_metadata = {
                    **rule_based_metadata,
                    **llm_metadata,
                    'enriched_by': 'llm',
                    'rule_based_fallback': {
                        'title': rule_based_metadata['title'],
                        'summary': rule_based_metadata['summary'],
                        'tags': rule_based_metadata['tags']
                    }
                }
                enrichment_method = "llm_enhanced"
            else:
                # LLM failed, use rule-based
                final_metadata = rule_based_metadata
                final_metadata['fallback_reason'] = 'llm_failure'
                enrichment_method = "rule_based_fallback"
                if trace:
                    trace.record_stage(
                        name="metadata_enrichment_fallback",
                        details={"chunk_id": chunk.id, "reason": "llm_failure"}
                    )
        else:
            final_metadata = rule_based_metadata
            enrichment_method = "rule_based"
        
        # Create new chunk with enriched metadata
        enriched_metadata = chunk.metadata.copy()
        enriched_metadata['enrichment'] = {
            'method': enrichment_method,
            'metadata': final_metadata
        }
        
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=enriched_metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref
        )
    
    def _create_fallback_chunk(self, chunk: Chunk) -> Chunk:
        """Create a fallback chunk with rule-based metadata when enrichment fails."""
        rule_based_metadata = self._rule_based_enrich(chunk)
        rule_based_metadata['fallback_reason'] = 'enrichment_error'
        
        enriched_metadata = chunk.metadata.copy()
        enriched_metadata['enrichment'] = {
            'method': 'rule_based_error_fallback',
            'metadata': rule_based_metadata
        }
        
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=enriched_metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref
        )
    
    def _rule_based_enrich(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Apply rule-based enrichment to generate metadata.
        
        Rules:
        - Title: First line or first 50 characters
        - Summary: Truncated first 100 characters
        - Tags: Extracted keywords based on patterns
        - enriched_by: "rule"
        - enriched_at: Current timestamp
        
        Args:
            chunk: Chunk to enrich
            
        Returns:
            Dictionary containing enriched metadata
        """
        text = chunk.text.strip()
        
        if not text:
            return {
                'title': '',
                'summary': '',
                'tags': [],
                'enriched_by': 'rule',
                'enriched_at': datetime.utcnow().isoformat()
            }
        
        # Extract title: first line or first 50 chars
        title = self._extract_title(text)
        
        # Extract summary: first 100 chars
        summary = self._extract_summary(text)
        
        # Extract tags from text
        tags = self._extract_tags(text)
        
        return {
            'title': title,
            'summary': summary,
            'tags': tags,
            'enriched_by': 'rule',
            'enriched_at': datetime.utcnow().isoformat()
        }
    
    def _extract_title(self, text: str, max_length: int = 50) -> str:
        """
        Extract title from text.
        
        Strategy:
        1. Try first line if it's not too long
        2. Otherwise take first max_length characters
        3. Clean up markdown headers if present
        
        Args:
            text: Source text
            max_length: Maximum title length
            
        Returns:
            Extracted title
        """
        if not text:
            return ""
        
        # Get first line
        first_line = text.split('\n')[0].strip()
        
        # Remove markdown header markers
        first_line = re.sub(r'^#{1,6}\s*', '', first_line)
        
        # If first line is empty after cleaning, try to get first non-empty line
        if not first_line:
            for line in text.split('\n'):
                clean_line = re.sub(r'^#{1,6}\s*', '', line.strip())
                if clean_line:
                    first_line = clean_line
                    break
        
        # If first line is within limit, use it
        if 0 < len(first_line) <= max_length:
            return first_line
        
        # Otherwise truncate (for lines between max_length and max_length*2, or longer)
        if len(first_line) > max_length:
            # Try to break at a word boundary
            truncated = first_line[:max_length]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.5:
                truncated = truncated[:last_space]
            return truncated + "..."
        
        return first_line
    
    def _extract_summary(self, text: str, max_length: int = 100) -> str:
        """
        Extract summary from text.
        
        Strategy:
        1. Take first max_length characters
        2. Try to break at sentence boundary
        3. Clean up the text
        
        Args:
            text: Source text
            max_length: Maximum summary length
            
        Returns:
            Extracted summary
        """
        if not text:
            return ""
        
        # Normalize whitespace
        normalized = ' '.join(text.split())
        
        if len(normalized) <= max_length:
            return normalized
        
        # Truncate at max_length
        truncated = normalized[:max_length]
        
        # Try to find a sentence boundary
        sentence_endings = ['. ', '? ', '! ', '。', '？', '！']
        last_boundary = -1
        for ending in sentence_endings:
            pos = truncated.rfind(ending)
            if pos > last_boundary:
                last_boundary = pos
        
        if last_boundary > max_length * 0.5:
            truncated = truncated[:last_boundary + 1]
        else:
            # No good sentence boundary, break at word
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.5:
                truncated = truncated[:last_space]
        
        return truncated.strip() + "..."
    
    def _extract_tags(self, text: str, max_tags: int = 5) -> List[str]:
        """
        Extract tags from text using pattern matching.
        
        Strategy:
        1. Find technical terms (camelCase, PascalCase, snake_case, acronyms)
        2. Find capitalized proper nouns (e.g., Python, Starlette)
        3. Find frequent meaningful words (excluding stop words)
        4. Limit to max_tags unique tags
        
        Args:
            text: Source text
            max_tags: Maximum number of tags to return
            
        Returns:
            List of extracted tags
        """
        if not text:
            return []
        
        tags = []
        tag_sources = {}  # Track source pattern for each tag
        
        # Collect all candidates first, then prioritize
        # Priority: technical patterns > capitalized proper nouns > frequent words
        # Reserve space for capitalized words (e.g., Python) alongside pattern matches
        
        # Reserve slots: 3 for pattern tags, 2 for capitalized words
        pattern_limit = max(1, max_tags - 2)  # At least 1, leave room for capitalized
        
        # 1. Collect pattern-based tags (technical patterns) - with limit
        pattern_tags = []
        for pattern, tag_type in self.KEYWORD_PATTERNS:
            if len(pattern_tags) >= pattern_limit:
                break
            matches = re.findall(pattern, text)
            for match in matches:
                if len(pattern_tags) >= pattern_limit:
                    break
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                match = match.strip()
                if match and len(match) > 1 and match.lower() not in self.STOP_WORDS:
                    if match not in tag_sources:
                        tag_sources[match] = tag_type
                        pattern_tags.append(match)
        
        # 2. Collect capitalized words (proper nouns/technical terms like Python)
        capitalized_tags = []
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for word in capitalized:
            if len(pattern_tags) + len(capitalized_tags) >= max_tags:
                break
            if word not in tag_sources and word.lower() not in self.STOP_WORDS:
                tag_sources[word] = 'proper_noun'
                capitalized_tags.append(word)
        
        # 3. Add pattern tags first (higher priority), then capitalized
        tags = pattern_tags + capitalized_tags
        
        # 4. Fill remaining slots with frequent words if needed
        if len(tags) < max_tags:
            words = re.findall(r'\b[A-Za-z][a-z]{2,}\b', text)
            word_counts = {}
            for word in words:
                word_lower = word.lower()
                if word_lower not in self.STOP_WORDS and len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
            for word, count in sorted_words:
                if len(tags) >= max_tags:
                    break
                if word not in tag_sources and count >= 2:  # At least 2 occurrences
                    tag_sources[word] = 'frequent'
                    tags.append(word)
        
        # Sort alphabetically for consistency
        tags = sorted(tags, key=str.lower)
        
        return tags
    
    def _llm_enrich(
        self,
        chunk: Chunk,
        trace: Optional[TraceContext] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to generate high-quality metadata.
        
        Sends the text to LLM with a metadata extraction prompt and returns
        the structured metadata. Returns None on failure to allow
        fallback to rule-based results.
        
        Args:
            chunk: Chunk to enrich
            trace: Optional trace context
            
        Returns:
            Dictionary with 'title', 'summary', 'tags' or None if LLM call fails
        """
        if not self._llm:
            return None
        
        # Load prompt template if not already loaded
        if self._prompt_template is None:
            self._prompt_template = self._load_prompt()
            if self._prompt_template is None:
                logger.warning("No prompt template available for LLM enrichment")
                return None
        
        # Format prompt with text
        prompt = self._prompt_template.format(text=chunk.text[:2000])  # Limit input size
        
        try:
            messages = [ChatMessage(role="user", content=prompt)]
            
            if trace:
                trace.record_stage(
                    name="llm_enrichment_request",
                    provider=self._llm.__class__.__name__,
                    details={
                        "model": self._llm.model,
                        "chunk_id": chunk.id,
                        "text_length": len(chunk.text)
                    }
                )
            
            response = self._llm.chat(messages, temperature=0.3)
            
            if trace:
                trace.record_stage(
                    name="llm_enrichment_response",
                    details={
                        "chunk_id": chunk.id,
                        "prompt_tokens": response.usage.get("prompt_tokens") if response.usage else None,
                        "completion_tokens": response.usage.get("completion_tokens") if response.usage else None
                    }
                )
            
            # Parse response to extract metadata
            metadata = self._parse_llm_response(response)
            
            if metadata:
                metadata['enriched_by'] = 'llm'
                metadata['enriched_at'] = datetime.utcnow().isoformat()
            
            return metadata
            
        except Exception as e:
            logger.warning(f"LLM enrichment failed for chunk {chunk.id}: {e}")
            if trace:
                trace.record_stage(
                    name="llm_enrichment_error",
                    details={"chunk_id": chunk.id, "error": str(e)}
                )
            return None
    
    def _parse_llm_response(self, response: ChatResponse) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response to extract metadata.
        
        Expects response in one of these formats:
        1. JSON: {"title": "...", "summary": "...", "tags": [...]}
        2. Structured text with clear markers
        
        Args:
            response: LLM response
            
        Returns:
            Parsed metadata dictionary or None if parsing fails
        """
        content = response.content.strip()
        
        if not content:
            logger.warning("LLM returned empty enrichment response")
            return None
        
        # Try JSON parsing first
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                # Validate required fields
                title = data.get('title', '').strip()
                summary = data.get('summary', '').strip()
                tags = data.get('tags', [])
                
                if not isinstance(tags, list):
                    tags = [str(tags)] if tags else []
                
                # Ensure we have at least some content
                if title or summary or tags:
                    return {
                        'title': title or 'Untitled',
                        'summary': summary or 'No summary available',
                        'tags': [str(t) for t in tags if t][:10]  # Max 10 tags
                    }
        except json.JSONDecodeError:
            pass  # Try text parsing
        
        # Try text-based parsing
        try:
            metadata = self._parse_text_response(content)
            if metadata:
                return metadata
        except Exception as e:
            logger.warning(f"Failed to parse text response: {e}")
        
        logger.warning(f"Could not parse LLM response: {content[:200]}...")
        return None
    
    def _parse_text_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse text-based LLM response.
        
        Looks for patterns like:
        - Title: xxx
        - Summary: xxx
        - Tags: xxx, yyy, zzz
        
        Args:
            content: Response content
            
        Returns:
            Parsed metadata or None
        """
        title = ""
        summary = ""
        tags: List[str] = []
        
        lines = content.split('\n')
        current_field = None
        current_value: List[str] = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for field markers
            title_match = re.match(r'^(?:title|标题)[：:]\s*(.+)', line, re.IGNORECASE)
            summary_match = re.match(r'^(?:summary|摘要)[：:]\s*(.+)', line, re.IGNORECASE)
            tags_match = re.match(r'^(?:tags?|标签)[：:]\s*(.+)', line, re.IGNORECASE)
            
            if title_match:
                if current_field:
                    title, summary, tags = self._set_field_value(
                        title, summary, tags, current_field, ' '.join(current_value)
                    )
                current_field = 'title'
                current_value = [title_match.group(1)]
            elif summary_match:
                if current_field:
                    title, summary, tags = self._set_field_value(
                        title, summary, tags, current_field, ' '.join(current_value)
                    )
                current_field = 'summary'
                current_value = [summary_match.group(1)]
            elif tags_match:
                if current_field:
                    title, summary, tags = self._set_field_value(
                        title, summary, tags, current_field, ' '.join(current_value)
                    )
                current_field = 'tags'
                current_value = [tags_match.group(1)]
            elif current_field:
                current_value.append(line)
        
        # Don't forget the last field
        if current_field:
            title, summary, tags = self._set_field_value(
                title, summary, tags, current_field, ' '.join(current_value)
            )
        
        if title or summary or tags:
            return {
                'title': title or 'Untitled',
                'summary': summary or 'No summary available',
                'tags': tags if isinstance(tags, list) else [tags]
            }
        
        return None
    
    def _set_field_value(
        self,
        title: str,
        summary: str,
        tags: List[str],
        field: str,
        value: str
    ) -> tuple:
        """Set field value and return updated tuple."""
        value = value.strip()
        if field == 'title':
            title = value
        elif field == 'summary':
            summary = value
        elif field == 'tags':
            # Parse tags (comma, semicolon, or Chinese comma separated)
            tags = [t.strip() for t in re.split(r'[,;，；]', value) if t.strip()]
        return title, summary, tags
    
    def _load_prompt(self, prompt_path: Optional[str] = None) -> Optional[str]:
        """
        Load prompt template from file.
        
        Args:
            prompt_path: Path to prompt file. Uses instance prompt_path if not provided.
            
        Returns:
            Prompt template string or None if file not found
        """
        path = prompt_path or self.prompt_path
        
        if not path:
            # Use default prompt
            return self._get_default_prompt()
        
        try:
            full_path = Path(path)
            if not full_path.is_absolute():
                # Try relative to project root
                full_path = Path(__file__).parents[3] / path
            
            with open(full_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Verify template has {text} placeholder
            if '{text}' not in template:
                logger.warning(f"Prompt template missing {{text}} placeholder: {path}")
                return self._get_default_prompt()
            
            logger.debug(f"Loaded prompt template from: {full_path}")
            return template
            
        except FileNotFoundError:
            logger.warning(f"Prompt template file not found: {path}, using default")
            return self._get_default_prompt()
        except Exception as e:
            logger.warning(f"Failed to load prompt template: {e}, using default")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Get default prompt template for metadata extraction."""
        return """Please analyze the following text and extract metadata in JSON format.

Text:
{text}

Please provide the output in the following JSON format:
{{
    "title": "A concise title for this text (max 50 chars)",
    "summary": "A brief summary of the main content (max 100 chars)",
    "tags": ["tag1", "tag2", "tag3"]
}}

Requirements:
- Title should be descriptive and capture the main topic
- Summary should briefly describe the key points
- Tags should be 3-5 relevant keywords or concepts
- Return ONLY the JSON, no additional text"""


# Export for use in other modules
__all__ = ["MetadataEnricher"]
