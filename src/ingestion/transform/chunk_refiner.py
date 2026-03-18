#!/usr/bin/env python3
"""
Chunk Refiner - Text Cleaning and Enhancement Transform

Implements a two-stage chunk refinement process:
1. Rule-based refinement: Fast pattern-based cleaning (whitespace, headers, etc.)
2. LLM-based enhancement: Optional intelligent rewriting via LLM

Features:
- Rule-based noise removal without API calls
- Optional LLM enhancement for OCR error correction
- Graceful degradation: falls back to rule results on LLM failure
- Configurable via settings (enable/disable LLM, prompt template path)
- Full trace integration for observability
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Any

from core.types import Chunk
from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from libs.llm.llm_factory import LLMFactory
from libs.llm.base_llm import BaseLLM, ChatMessage
from .base_transform import BaseTransform, TransformConfigError

logger = logging.getLogger(__name__)


class ChunkRefiner(BaseTransform):
    """
    Chunk refinement transform with rule-based and LLM enhancement.
    
    This transform processes chunks through two stages:
    
    1. Rule-based refinement (always applied):
       - Removes excessive whitespace and normalizes line breaks
       - Strips common page headers/footers and page numbers
       - Removes HTML/Markdown formatting markers
       - Preserves code block formatting and indentation
    
    2. LLM-based enhancement (optional, configurable):
       - Sends text to LLM for intelligent refinement
       - Can fix OCR errors, improve readability
       - Falls back to rule-based results on LLM failure
    
    Configuration options (via settings.ingestion.chunk_refiner):
    - enabled: bool - Whether to enable refinement (default: True)
    - use_llm: bool - Whether to use LLM enhancement (default: False)
    - prompt_path: str - Path to LLM prompt template file
    
    Example:
        >>> settings = load_settings("config/settings.yaml")
        >>> refiner = ChunkRefiner(settings)
        >>> refined_chunks = refiner.transform(chunks, trace)
    """
    
    # Common patterns for noise detection (case-insensitive via re.IGNORECASE during compilation)
    HEADER_PATTERNS = [
        r'^\s*confidential\s*.*$',
        r'^\s*internal use only\s*.*$',
        r'^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$',
        r'^\s*chapter\s+\d+[:\.]?\s*.*$',
        r'^\s*appendix\s+[a-z]\s*.*$',
        r'^\s*section\s+\d+[.\d]*\s*.*$',
    ]
    
    FOOTER_PATTERNS = [
        r'^\s*copyright\s+.*$',
        r'^\s*all rights reserved\s*.*$',
        r'^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$',
        r'^\s*\d+\s*/\s*\d+\s*$',  # "3 / 10" page format
        r'^\s*doc\s*id[:\s]+\w+.*$',
        r'^\s*version[:\s]+\d+[.\d]*.*$',
        r'^\s*last updated?[:\s]+.*$',
        r'^.*\s+-\s+page\s+\d+.*$',  # "Title - Page 45" format
        r'^.*\s+page\s+\d+.*$',  # Any text ending with page number
    ]
    
    # HTML/XML comment pattern
    HTML_COMMENT_PATTERN = r'<!--.*?-->'
    
    # Common HTML/Markdown tags to remove
    HTML_TAG_PATTERNS = [
        r'</?[a-zA-Z][^>]*>',  # HTML tags like <p>, </div>, etc.
    ]
    
    # OCR error patterns (common substitutions)
    OCR_ERROR_PATTERNS = [
        (r'\b0(?=CR)', 'O'),      # 0CR -> OCR
        (r'(?<=O)CR\b', 'CR'),    # OCR (keep)
        (r'\b1(?=n\w)', 'i'),     # 1n -> in
        (r'(?<=[a-z])1(?=ng)', 'i'),  # s1gn -> sign
        (r'(?<=[a-z])0(?=[nrt])', 'o'),  # c0n -> con, d0 -> do, etc
        (r'(?<=[a-z])1(?=[szv])', 'l'),  # a1s -> als, a1so -> also
        (r'(?<=[a-z])5(?=[sz])', 's'),   # a5s -> ass
        (r'3', 'e'),  # common OCR: 3 -> e (when surrounded by letters)
    ]
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ) -> None:
        """
        Initialize the chunk refiner.
        
        Args:
            settings: Application settings containing chunk_refiner config
            llm: Optional pre-configured LLM instance. If not provided,
                 will be created from settings if use_llm is enabled.
            prompt_path: Optional path to prompt template. Overrides config.
            
        Raises:
            TransformConfigError: If configuration is invalid
        """
        super().__init__()
        
        self.settings = settings
        
        # Get chunk_refiner config from settings
        refiner_config = getattr(settings.ingestion, 'chunk_refiner', {})
        self.enabled = refiner_config.get('enabled', True)
        self.use_llm = refiner_config.get('use_llm', False)
        
        # Determine prompt path
        self.prompt_path = prompt_path or refiner_config.get('prompt_path')
        self._prompt_template: Optional[str] = None
        
        # Initialize LLM if needed
        self._llm: Optional[BaseLLM] = llm
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(settings)
                logger.info(f"Created LLM for chunk refinement: {self._llm.model}")
            except Exception as e:
                logger.warning(f"Failed to create LLM for refinement: {e}. "
                              "Will fall back to rule-based refinement.")
                self._llm = None
        
        # Compile regex patterns for efficiency (case-insensitive)
        self._header_regex = re.compile('|'.join(f'({p})' for p in self.HEADER_PATTERNS), 
                                         re.MULTILINE | re.IGNORECASE)
        self._footer_regex = re.compile('|'.join(f'({p})' for p in self.FOOTER_PATTERNS), 
                                         re.MULTILINE | re.IGNORECASE)
        self._html_comment_regex = re.compile(self.HTML_COMMENT_PATTERN, re.DOTALL)
        self._html_tag_regex = re.compile('|'.join(self.HTML_TAG_PATTERNS))
        
        logger.debug(f"ChunkRefiner initialized: enabled={self.enabled}, use_llm={self.use_llm}")
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Transform chunks through rule-based and optional LLM refinement.
        
        Processing flow:
        1. If not enabled, return chunks unchanged
        2. Apply rule-based refinement to each chunk
        3. If use_llm is enabled, send to LLM for enhancement
        4. On LLM failure, fall back to rule-based result
        
        Args:
            chunks: List of chunks to refine
            trace: Optional trace context for recording stages
            
        Returns:
            List of refined chunks
        """
        if not self.enabled:
            logger.debug("ChunkRefiner disabled, returning chunks unchanged")
            return chunks
        
        if not chunks:
            return []
        
        if trace:
            trace.record_stage(
                name="chunk_refinement",
                method="rule_based" + ("+llm" if self.use_llm else ""),
                provider="ChunkRefiner"
            )
        
        refined_chunks: List[Chunk] = []
        
        for chunk in chunks:
            try:
                refined_chunk = self._refine_chunk(chunk, trace)
                refined_chunks.append(refined_chunk)
            except Exception as e:
                logger.error(f"Failed to refine chunk {chunk.id}: {e}")
                # Fall back to original chunk on error
                refined_chunks.append(chunk)
        
        if trace:
            trace.record_stage(
                name="chunk_refinement_complete",
                details={
                    "input_count": len(chunks),
                    "output_count": len(refined_chunks)
                }
            )
        
        return refined_chunks
    
    def _refine_chunk(
        self,
        chunk: Chunk,
        trace: Optional[TraceContext] = None
    ) -> Chunk:
        """
        Refine a single chunk.
        
        Args:
            chunk: Chunk to refine
            trace: Optional trace context
            
        Returns:
            Refined chunk (may be same instance or new)
        """
        # Stage 1: Rule-based refinement (always applied)
        rule_based_text = self._rule_based_refine(chunk.text)
        
        # Stage 2: LLM enhancement (optional, with fallback)
        if self.use_llm and self._llm:
            llm_result = self._llm_refine(rule_based_text, trace)
            if llm_result is not None:
                final_text = llm_result
                refinement_method = "llm_enhanced"
            else:
                # LLM failed, fall back to rule-based
                final_text = rule_based_text
                refinement_method = "rule_based_fallback"
                if trace:
                    trace.record_stage(
                        name="chunk_refinement_fallback",
                        details={"chunk_id": chunk.id, "reason": "llm_failure"}
                    )
        else:
            final_text = rule_based_text
            refinement_method = "rule_based"
        
        # Create new chunk with refined text
        # Preserve all metadata and add refinement info
        refined_metadata = chunk.metadata.copy()
        refined_metadata['refinement'] = {
            'method': refinement_method,
            'original_length': len(chunk.text),
            'refined_length': len(final_text)
        }
        
        return Chunk(
            id=chunk.id,
            text=final_text,
            metadata=refined_metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref
        )
    
    def _rule_based_refine(self, text: str) -> str:
        """
        Apply rule-based refinement to text.
        
        Performs the following operations:
        1. Normalize line endings
        2. Remove HTML comments
        3. Strip HTML/Markdown tags (preserving code blocks)
        4. Remove page headers and footers
        5. Normalize whitespace
        6. Fix common OCR errors (optional)
        
        Args:
            text: Raw text to refine
            
        Returns:
            Refined text
        """
        if not text:
            return ""
        
        # Split into lines for processing
        lines = text.split('\n')
        processed_lines: List[str] = []
        in_code_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # Detect code block boundaries
            if stripped.startswith('```') or stripped.startswith('~~~'):
                in_code_block = not in_code_block
                processed_lines.append(line)
                continue
            
            if in_code_block:
                # Preserve code block content as-is
                processed_lines.append(line)
                continue
            
            # Skip empty lines (will normalize later)
            if not stripped:
                processed_lines.append('')  # Mark for potential removal
                continue
            
            # Skip headers and footers
            if self._is_header(stripped):
                continue
            if self._is_footer(stripped):
                continue
            
            # Remove HTML comments
            line = self._html_comment_regex.sub('', line)
            
            # Remove HTML tags
            line = self._html_tag_regex.sub('', line)
            
            # Basic whitespace normalization (preserve inline spaces)
            line = ' '.join(line.split())
            
            if line:
                processed_lines.append(line)
        
        # Rejoin and normalize excessive blank lines
        result = '\n'.join(processed_lines)
        result = self._normalize_blank_lines(result)
        
        # Strip leading/trailing whitespace
        result = result.strip()
        
        return result
    
    def _is_header(self, line: str) -> bool:
        """Check if line matches header patterns."""
        return bool(self._header_regex.match(line))
    
    def _is_footer(self, line: str) -> bool:
        """Check if line matches footer patterns."""
        return bool(self._footer_regex.match(line))
    
    def _normalize_blank_lines(self, text: str, max_consecutive: int = 2) -> str:
        """
        Normalize excessive blank lines.
        
        Args:
            text: Text with potential excessive newlines
            max_consecutive: Maximum number of consecutive blank lines to keep
            
        Returns:
            Text with normalized blank lines
        """
        # Replace 3+ consecutive newlines with max_consecutive newlines
        pattern = '\n{' + str(max_consecutive + 1) + ',}'
        replacement = '\n' * max_consecutive
        return re.sub(pattern, replacement, text)
    
    def _llm_refine(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """
        Use LLM to refine text.
        
        Sends the text to LLM with a refinement prompt and returns
        the improved version. Returns None on failure to allow
        fallback to rule-based results.
        
        Args:
            text: Text to refine (already rule-cleaned)
            trace: Optional trace context
            
        Returns:
            Refined text or None if LLM call fails
        """
        if not self._llm:
            return None
        
        # Load prompt template if not already loaded
        if self._prompt_template is None:
            self._prompt_template = self._load_prompt()
            if self._prompt_template is None:
                logger.warning("No prompt template available for LLM refinement")
                return None
        
        # Format prompt with text
        prompt = self._prompt_template.format(text=text)
        
        try:
            messages = [ChatMessage(role="user", content=prompt)]
            
            if trace:
                trace.record_stage(
                    name="llm_refinement_request",
                    provider=self._llm.__class__.__name__,
                    details={
                        "model": self._llm.model,
                        "text_length": len(text)
                    }
                )
            
            response = self._llm.chat(messages, temperature=0.3)
            refined_text = response.content.strip()
            
            if trace:
                trace.record_stage(
                    name="llm_refinement_response",
                    details={
                        "response_length": len(refined_text),
                        "prompt_tokens": response.usage.get("prompt_tokens") if response.usage else None,
                        "completion_tokens": response.usage.get("completion_tokens") if response.usage else None
                    }
                )
            
            # Validate response (should not be empty and should be reasonable length)
            if not refined_text:
                logger.warning("LLM returned empty refinement")
                return None
            
            # Allow some size variance but flag extreme changes
            size_ratio = len(refined_text) / max(len(text), 1)
            if size_ratio < 0.1 or size_ratio > 10:
                logger.warning(f"LLM refinement produced extreme size change: "
                              f"{len(text)} -> {len(refined_text)} (ratio: {size_ratio:.2f})")
                # Still return it, but warn
            
            return refined_text
            
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")
            if trace:
                trace.record_stage(
                    name="llm_refinement_error",
                    details={"error": str(e)}
                )
            return None
    
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
            logger.warning("No prompt path configured")
            return None
        
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
                return None
            
            logger.debug(f"Loaded prompt template from: {full_path}")
            return template
            
        except FileNotFoundError:
            logger.warning(f"Prompt template file not found: {path}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load prompt template: {e}")
            return None
    
    def _fix_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors in text.
        
        This is an experimental feature that attempts to correct
        common OCR misrecognitions without LLM assistance.
        
        Args:
            text: Text potentially containing OCR errors
            
        Returns:
            Text with some OCR errors corrected
        """
        result = text
        for pattern, replacement in self.OCR_ERROR_PATTERNS:
            result = re.sub(pattern, replacement, result)
        return result
