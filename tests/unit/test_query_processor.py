#!/usr/bin/env python3
"""
Query Processor Unit Tests

Test cases for QueryProcessor class including:
- Normal tokenization
- Stopword filtering
- Empty query handling
- Filters processing
- Fallback tokenization
"""

import pytest
from src.core.query_engine.query_processor import QueryProcessor, ProcessedQuery


class TestQueryProcessor:
    """Test suite for QueryProcessor"""

    @pytest.fixture
    def processor(self):
        """Create a QueryProcessor instance for testing"""
        return QueryProcessor(settings=None)

    def test_extract_keywords_simple_chinese(self, processor):
        """Test keyword extraction from simple Chinese query"""
        query = "Python编程语言的特点"
        keywords = processor.extract_keywords(query)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Should contain 'python' and '编程' or '语言' or '特点'
        assert any("python" in kw for kw in keywords)

    def test_extract_keywords_simple_english(self, processor):
        """Test keyword extraction from simple English query"""
        query = "machine learning algorithms"
        keywords = processor.extract_keywords(query)
        
        assert isinstance(keywords, list)
        assert len(keywords) >= 2
        # Should contain 'machine' and 'learning'
        assert "machine" in keywords
        assert "learning" in keywords

    def test_extract_keywords_mixed(self, processor):
        """Test keyword extraction from mixed Chinese-English query"""
        query = "如何使用Python进行数据分析"
        keywords = processor.extract_keywords(query)
        
        assert isinstance(keywords, list)
        assert len(keywords) >= 2
        # Should contain 'python' and at least one Chinese keyword
        assert any("python" in kw for kw in keywords)

    def test_stopwords_filtering(self, processor):
        """Test that stopwords are filtered out"""
        query = "这是一个测试"  # "这", "是" are stopwords
        keywords = processor.extract_keywords(query)
        
        # Should not contain stopwords
        assert "这" not in keywords
        assert "是" not in keywords
        assert "一个" not in keywords
        # Should contain content words
        assert "测试" in keywords

    def test_stopwords_filtering_english(self, processor):
        """Test that English stopwords are filtered out"""
        query = "the quick brown fox"  # "the" is a stopword
        keywords = processor.extract_keywords(query)
        
        # Should not contain stopwords
        assert "the" not in keywords
        # Should contain content words
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords

    def test_empty_query(self, processor):
        """Test handling of empty query"""
        assert processor.extract_keywords("") == []
        assert processor.extract_keywords("   ") == []
        assert processor.extract_keywords(None) == []

    def test_whitespace_only_query(self, processor):
        """Test handling of whitespace-only query"""
        keywords = processor.extract_keywords("   \n\t  ")
        assert keywords == []

    def test_punctuation_only_query(self, processor):
        """Test handling of punctuation-only query"""
        keywords = processor.extract_keywords("...，。！？")
        assert keywords == []

    def test_numbers_filtered(self, processor):
        """Test that pure numbers are filtered out"""
        query = "123 456 test"
        keywords = processor.extract_keywords(query)
        
        assert "123" not in keywords
        assert "456" not in keywords
        assert "test" in keywords

    def test_process_returns_processed_query(self, processor):
        """Test that process() returns ProcessedQuery object"""
        query = "machine learning"
        result = processor.process(query)
        
        assert isinstance(result, ProcessedQuery)
        assert result.query == query
        assert isinstance(result.keywords, list)
        assert isinstance(result.filters, dict)

    def test_process_with_filters(self, processor):
        """Test process() with provided filters"""
        query = "test query"
        filters = {"source": "docs", "date": "2024-01-01"}
        result = processor.process(query, filters=filters)
        
        assert result.filters == filters
        assert result.filters["source"] == "docs"
        assert result.filters["date"] == "2024-01-01"

    def test_process_without_filters(self, processor):
        """Test process() without filters returns empty dict"""
        query = "test query"
        result = processor.process(query)
        
        assert result.filters == {}

    def test_process_with_none_filters(self, processor):
        """Test process() with None filters returns empty dict"""
        query = "test query"
        result = processor.process(query, filters=None)
        
        assert result.filters == {}

    def test_process_preserves_original_query(self, processor):
        """Test that process() preserves original query string"""
        query = "  Original Query With Mixed Case  "
        result = processor.process(query)
        
        assert result.query == query

    def test_keywords_are_lowercase(self, processor):
        """Test that extracted keywords are lowercase"""
        query = "MACHINE LEARNING Python"
        keywords = processor.extract_keywords(query)
        
        for kw in keywords:
            assert kw == kw.lower(), f"Keyword '{kw}' is not lowercase"

    def test_keywords_deduplication(self, processor):
        """Test that duplicate keywords are removed"""
        query = "test test test"
        keywords = processor.extract_keywords(query)
        
        # Should only have 'test' once
        assert keywords.count("test") == 1

    def test_fallback_tokenization_triggered(self, processor):
        """Test that fallback tokenization works when jieba returns few results"""
        # Query with mostly stopwords that would be filtered out
        query = "the and or but"
        keywords = processor.extract_keywords(query)
        
        # Should use fallback and extract something
        # Even if all are stopwords, fallback should handle it
        assert isinstance(keywords, list)

    def test_long_query_extraction(self, processor):
        """Test keyword extraction from longer query"""
        query = "如何在Python中使用pandas库进行数据清洗和数据分析工作"
        keywords = processor.extract_keywords(query)
        
        assert isinstance(keywords, list)
        assert len(keywords) >= 3
        # Should contain important terms
        assert any("python" in kw for kw in keywords)
        assert any("pandas" in kw for kw in keywords)

    def test_special_characters_handling(self, processor):
        """Test handling of special characters in query"""
        query = "C++ programming & Java development!"
        keywords = processor.extract_keywords(query)
        
        assert isinstance(keywords, list)
        # Should extract meaningful parts
        assert any("c" in kw or "programming" in kw for kw in keywords)

    def test_case_insensitive_stopwords(self, processor):
        """Test that stopword filtering is case insensitive"""
        query = "THE Quick Brown Fox"  # "THE" uppercase
        keywords = processor.extract_keywords(query)
        
        assert "the" not in keywords
        assert "quick" in keywords

    def test_process_query_with_empty_string(self, processor):
        """Test process() with empty string query"""
        result = processor.process("")
        
        assert isinstance(result, ProcessedQuery)
        assert result.query == ""
        assert result.keywords == []
        assert result.filters == {}

    def test_filters_dict_mutation_safety(self, processor):
        """Test that modifying returned filters doesn't affect internal state"""
        query = "test"
        filters = {"key": "value"}
        result = processor.process(query, filters=filters)
        
        # Modify the returned filters
        result.filters["new_key"] = "new_value"
        
        # Original filters should not be modified
        assert "new_key" not in filters

    def test_stopwords_lazy_loading(self, processor):
        """Test that stopwords are loaded lazily"""
        # Before accessing stopwords
        assert processor._stopwords is None
        
        # Access stopwords
        stopwords = processor.stopwords
        
        # After accessing
        assert processor._stopwords is not None
        assert isinstance(stopwords, set)
        assert len(stopwords) > 0

    def test_default_stopwords_contain_common_words(self, processor):
        """Test that default stopwords contain common Chinese and English words"""
        # Access default stopwords
        from src.core.query_engine.query_processor import DEFAULT_STOPWORDS
        
        # Check Chinese stopwords
        assert "的" in DEFAULT_STOPWORDS
        assert "是" in DEFAULT_STOPWORDS
        
        # Check English stopwords
        assert "the" in DEFAULT_STOPWORDS
        assert "is" in DEFAULT_STOPWORDS


class TestProcessedQuery:
    """Test suite for ProcessedQuery dataclass"""

    def test_processed_query_creation(self):
        """Test creating ProcessedQuery instance"""
        pq = ProcessedQuery(
            query="test query",
            keywords=["test", "query"],
            filters={"source": "docs"}
        )
        
        assert pq.query == "test query"
        assert pq.keywords == ["test", "query"]
        assert pq.filters == {"source": "docs"}

    def test_processed_query_empty_values(self):
        """Test ProcessedQuery with empty values"""
        pq = ProcessedQuery(
            query="",
            keywords=[],
            filters={}
        )
        
        assert pq.query == ""
        assert pq.keywords == []
        assert pq.filters == {}

    def test_processed_query_immutability(self):
        """Test that ProcessedQuery fields can be modified (dataclass is not frozen)"""
        pq = ProcessedQuery(
            query="test",
            keywords=["test"],
            filters={}
        )
        
        # Should be able to modify fields
        pq.query = "modified"
        pq.keywords.append("new")
        pq.filters["key"] = "value"
        
        assert pq.query == "modified"
        assert "new" in pq.keywords
        assert pq.filters["key"] == "value"


class TestQueryProcessorWithMockSettings:
    """Test QueryProcessor with mock settings"""

    def test_processor_with_none_settings(self):
        """Test processor initialized with None settings"""
        processor = QueryProcessor(settings=None)
        
        assert processor.settings is None
        result = processor.process("test query")
        assert isinstance(result, ProcessedQuery)

    def test_processor_with_mock_settings(self):
        """Test processor with mock settings object"""
        mock_settings = {"some_config": "value"}
        processor = QueryProcessor(settings=mock_settings)
        
        assert processor.settings == mock_settings
        result = processor.process("test query")
        assert isinstance(result, ProcessedQuery)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
