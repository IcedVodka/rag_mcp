#!/usr/bin/env python3
"""
Query Processor - 查询预处理器

提供关键词提取和filters结构解析功能。
使用jieba分词进行中文/英文混合文本的分词处理。
"""

import logging
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import jieba
import jieba.posseg as pseg

logger = logging.getLogger(__name__)

# 默认停用词（当config/stopwords.txt不存在时使用）
DEFAULT_STOPWORDS = {
    # 中文停用词
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也",
    "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
    "有", "与", "及", "等", "或", "但", "而", "如果", "因为", "所以", "虽然", "但是", "而且",
    "可以", "这个", "那个", "这些", "那些", "这样", "那样", "这里", "那里", "哪里", "什么",
    "怎么", "为什么", "如何", "谁", "哪个", "之", "为", "以", "于", "则", "乃", "已", "被",
    "把", "让", "向", "到", "从", "将", "对", "关于", "跟", "和", "同", "给", "替", "比",
    "除了", "除", "有关", "相关", "涉及", "以及", "还是", "或者", "要么", "假如", "假设",
    # 英文停用词
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
    "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "and", "but", "if", "or", "because", "until", "while", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "it", "its", "it's", "we", "our", "you", "your",
    "they", "them", "their", "theirs", "he", "him", "his", "she", "her", "hers", "i", "me",
    "my", "mine", "us", "our", "ours",
}

# 保留的词性：名词、动词、形容词、副词、英文单词等实词
# jieba词性标注说明：
# - n: 名词, v: 动词, a: 形容词, d: 副词, i: 成语, j: 简称, x: 非语素字
# - eng: 英文单词
# - nr: 人名, ns: 地名, nt: 机构名, nw: 作品名
# - t: 时间词, s: 处所词, f: 方位词
VALID_POS_PREFIXES = ("n", "v", "a", "d", "i", "j", "x", "eng", "nr", "ns", "nt", "nw", "t")


@dataclass
class ProcessedQuery:
    """
    处理后的查询对象
    
    Attributes:
        query: 原始查询字符串
        keywords: 提取的关键词列表
        filters: 解析后的filters字典
    """
    query: str
    keywords: list[str]
    filters: dict[str, Any]


class QueryProcessor:
    """
    查询处理器 - 负责关键词提取和filters解析
    
    使用jieba分词对查询进行分词，过滤停用词，保留实词。
    支持filters结构的透传。
    
    Attributes:
        settings: 应用配置对象
        stopwords: 停用词集合
    """

    def __init__(self, settings: Optional[Any] = None) -> None:
        """
        初始化QueryProcessor
        
        Args:
            settings: 应用配置对象，可选
        """
        self.settings = settings
        self._stopwords: Optional[set[str]] = None

    @property
    def stopwords(self) -> set[str]:
        """
        获取停用词集合，懒加载
        
        优先从config/stopwords.txt加载，如不存在则使用内置默认停用词
        
        Returns:
            停用词集合
        """
        if self._stopwords is None:
            self._stopwords = self._load_stopwords()
        return self._stopwords

    def _load_stopwords(self) -> set[str]:
        """
        加载停用词表
        
        Returns:
            停用词集合
        """
        # 尝试从配置文件加载
        stopwords_path = Path("config/stopwords.txt")
        if stopwords_path.exists():
            try:
                with open(stopwords_path, "r", encoding="utf-8") as f:
                    stopwords = {line.strip() for line in f if line.strip()}
                logger.debug(f"Loaded {len(stopwords)} stopwords from {stopwords_path}")
                return stopwords
            except Exception as e:
                logger.warning(f"Failed to load stopwords from {stopwords_path}: {e}")
        
        # 使用默认停用词
        logger.debug(f"Using default stopwords ({len(DEFAULT_STOPWORDS)} words)")
        return DEFAULT_STOPWORDS.copy()

    def _is_valid_word(self, word: str, pos: str) -> bool:
        """
        判断一个词是否有效（应保留）
        
        Args:
            word: 词语
            pos: 词性标注
            
        Returns:
            是否保留该词
        """
        # 过滤空字符串
        if not word or not word.strip():
            return False
        
        word = word.strip().lower()
        
        # 过滤纯标点
        if all(c in string.punctuation for c in word):
            return False
        
        # 过滤纯数字
        if word.isdigit():
            return False
        
        # 检查停用词
        if word in self.stopwords:
            return False
        
        # 检查词性 - 保留实词
        if pos.startswith(VALID_POS_PREFIXES):
            return True
        
        # 英文单词（长度>2）
        if word.isalpha() and len(word) > 2:
            return True
        
        # 中英混合词
        if any(c.isalpha() for c in word) and len(word) >= 2:
            return True
        
        return False

    def _fallback_tokenize(self, query: str) -> list[str]:
        """
        后备分词方法 - 当jieba分词结果不足时使用
        
        按空格和标点分割，过滤停用词
        
        Args:
            query: 原始查询
            
        Returns:
            关键词列表
        """
        # 按空格、标点符号分割（包括中英文标点）
        # Unicode标点范围：\u3000-\u303F(中文标点) \uFF00-\uFFEF(全角) \u2000-\u206F(通用标点)
        # 以及标准ASCII标点
        tokens = re.split(r"[\s" + string.punctuation + r"\u3000-\u303F\uFF00-\uFFEF\u2000-\u206F]+", query)
        
        keywords = []
        for token in tokens:
            token = token.strip().lower()
            # 过滤空字符串、停用词、纯数字、过短的词
            if (token and 
                token not in self.stopwords and 
                not token.isdigit() and 
                len(token) > 1):
                keywords.append(token)
        
        return keywords

    def extract_keywords(self, query: str) -> list[str]:
        """
        从查询中提取关键词
        
        使用jieba分词，过滤停用词，保留实词。
        如果分词结果太少，则回退到简单分割方式。
        
        Args:
            query: 原始查询字符串
            
        Returns:
            关键词列表（去重，保持顺序）
        """
        if not query or not query.strip():
            return []
        
        query = query.strip()
        
        # 使用jieba进行词性标注分词
        words_pos = list(pseg.cut(query))
        logger.debug(f"Jieba segmentation result: {words_pos}")
        
        keywords = []
        seen = set()
        
        for word, pos in words_pos:
            word_lower = word.lower().strip()
            
            # 检查是否有效词
            if self._is_valid_word(word, pos):
                # 去重（不区分大小写）
                if word_lower not in seen:
                    keywords.append(word_lower)
                    seen.add(word_lower)
        
        # 如果关键词太少，使用后备方法
        if len(keywords) < 2:
            fallback_keywords = self._fallback_tokenize(query)
            # 合并后备结果，保持去重
            for kw in fallback_keywords:
                if kw not in seen:
                    keywords.append(kw)
                    seen.add(kw)
        
        logger.debug(f"Extracted keywords: {keywords}")
        return keywords

    def process(self, query: str, filters: Optional[dict[str, Any]] = None) -> ProcessedQuery:
        """
        处理查询
        
        提取关键词并处理filters结构
        
        Args:
            query: 原始查询字符串
            filters: 可选的filters字典，透传使用
            
        Returns:
            ProcessedQuery对象，包含原始query、提取的keywords和filters
        """
        # 提取关键词
        keywords = self.extract_keywords(query)
        
        # 处理filters - 返回副本以避免外部修改影响内部状态
        processed_filters = filters.copy() if filters is not None else {}
        
        return ProcessedQuery(
            query=query,
            keywords=keywords,
            filters=processed_filters
        )
