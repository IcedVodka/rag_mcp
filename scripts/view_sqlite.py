#!/usr/bin/env python3
"""
查看 SQLite 数据库内容的人类可读脚本

Usage:
    python scripts/view_sqlite.py                    # 查看所有数据库
    python scripts/view_sqlite.py bm25               # 只查看 BM25 索引
    python scripts/view_sqlite.py ingestion          # 只查看摄入历史
    python scripts/view_sqlite.py image              # 只查看图片索引
    python scripts/view_sqlite.py --path <db_file>   # 查看指定数据库
"""

import sqlite3
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 数据库路径配置
DB_PATHS = {
    "bm25": "data/db/bm25/index.db",
    "ingestion": "data/db/ingestion_history.db",
    "image": "data/db/image_index.db",
}


def format_timestamp(ts: float) -> str:
    """格式化时间戳为人类可读格式"""
    if not ts:
        return "N/A"
    try:
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(ts)


def truncate_text(text: str, max_len: int = 80) -> str:
    """截断长文本"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def get_tables(conn: sqlite3.Connection) -> List[str]:
    """获取数据库中的所有表"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cursor.fetchall()]


def get_table_schema(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    """获取表结构"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    return [
        {
            "cid": col[0],
            "name": col[1],
            "type": col[2],
            "notnull": col[3],
            "default": col[4],
            "pk": col[5],
        }
        for col in columns
    ]


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    """获取表行数"""
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    return cursor.fetchone()[0]


def fetch_rows(conn: sqlite3.Connection, table: str, limit: int = 20) -> List[tuple]:
    """获取表数据"""
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table} LIMIT {limit}")
    return cursor.fetchall()


def print_table(conn: sqlite3.Connection, table: str, db_name: str):
    """打印单个表的内容"""
    schema = get_table_schema(conn, table)
    row_count = count_rows(conn, table)
    rows = fetch_rows(conn, table, limit=20)
    
    print(f"\n{'='*80}")
    print(f"📋 表: {table}")
    print(f"{'='*80}")
    print(f"总行数: {row_count}")
    
    # 打印表结构
    print(f"\n  结构:")
    for col in schema:
        pk_marker = "🔑 " if col["pk"] else "   "
        null_marker = "NOT NULL" if col["notnull"] else "NULL"
        print(f"    {pk_marker}{col['name']:20} {col['type']:15} {null_marker}")
    
    # 打印数据
    if rows:
        print(f"\n  前 {len(rows)} 行数据:")
        col_names = [col["name"] for col in schema]
        
        # 打印表头
        header = " | ".join(f"{name:20}" for name in col_names)
        print(f"    {'-'*len(header)}")
        print(f"    {header}")
        print(f"    {'-'*len(header)}")
        
        # 打印行数据
        for row in rows:
            formatted_values = []
            for i, value in enumerate(row):
                col_name = col_names[i] if i < len(col_names) else f"col{i}"
                
                # 特殊字段处理
                if col_name in ("timestamp", "processed_at", "created_at") and isinstance(value, (int, float)):
                    formatted_values.append(f"{format_timestamp(value):20}")
                elif col_name == "status":
                    icon = "✅" if value == "success" else "❌" if value == "failed" else "⏳"
                    formatted_values.append(f"{icon} {str(value):18}")
                elif col_name in ("text", "content", "source_path", "error_message"):
                    formatted_values.append(f"{truncate_text(str(value), 20):20}")
                elif col_name in ("hash", "file_hash", "chunk_id"):
                    formatted_values.append(f"{str(value)[:18]:20}")
                else:
                    formatted_values.append(f"{truncate_text(str(value), 20):20}")
            
            print("    " + " | ".join(formatted_values))
    else:
        print("\n  (无数据)")


def print_bm25_stats(conn: sqlite3.Connection):
    """打印 BM25 统计信息"""
    print(f"\n{'='*80}")
    print("📊 BM25 索引统计")
    print(f"{'='*80}")
    
    cursor = conn.cursor()
    
    # 文档数
    cursor.execute("SELECT COUNT(DISTINCT chunk_id) FROM postings")
    doc_count = cursor.fetchone()[0]
    print(f"  索引文档数: {doc_count}")
    
    # 词项数
    cursor.execute("SELECT COUNT(*) FROM terms")
    term_count = cursor.fetchone()[0]
    print(f"  词项总数: {term_count}")
    
    # 平均文档长度
    cursor.execute("SELECT AVG(doc_length) FROM postings")
    avg_len = cursor.fetchone()[0]
    print(f"  平均文档长度: {avg_len:.1f} 词")
    
    # 高频词项 (Top 10)
    cursor.execute("""
        SELECT p.term, COUNT(*) as df, AVG(p.tf) as avg_tf
        FROM postings p
        GROUP BY p.term
        ORDER BY df DESC
        LIMIT 10
    """)
    top_terms = cursor.fetchall()
    print(f"\n  高频词项 (Top 10):")
    print(f"    {'词项':20} {'文档频率':>10} {'平均TF':>10}")
    print(f"    {'-'*45}")
    for term, df, avg_tf in top_terms:
        print(f"    {term:20} {df:>10} {avg_tf:>10.2f}")


def print_ingestion_stats(conn: sqlite3.Connection):
    """打印摄入历史统计"""
    print(f"\n{'='*80}")
    print("📊 摄入历史统计")
    print(f"{'='*80}")
    
    cursor = conn.cursor()
    
    # 获取实际表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    table_name = tables[0] if tables else "ingestion_history"
    
    # 状态分布
    cursor.execute(f"SELECT status, COUNT(*) FROM {table_name} GROUP BY status")
    status_counts = cursor.fetchall()
    print(f"  文件状态分布:")
    for status, count in status_counts:
        icon = "✅" if status == "success" else "❌" if status == "failed" else "⏳"
        print(f"    {icon} {status:10}: {count:4} 个文件")
    
    # 最近摄入的文件
    cursor.execute(f"""
        SELECT file_path, status, updated_at, 0 as chunk_count
        FROM {table_name}
        ORDER BY updated_at DESC
        LIMIT 5
    """)
    recent = cursor.fetchall()
    print(f"\n  最近摄入的文件:")
    for path, status, ts, chunks in recent:
        icon = "✅" if status == "success" else "❌"
        time_str = format_timestamp(ts)
        print(f"    {icon} {Path(path).name:30} ({chunks} chunks) @ {time_str}")


def view_database(db_path: str, db_name: str):
    """查看单个数据库"""
    if not os.path.exists(db_path):
        print(f"❌ 数据库不存在: {db_path}")
        return
    
    print(f"\n{'#'*80}")
    print(f"# 🔍 数据库: {db_name}")
    print(f"# 📁 路径: {db_path}")
    print(f"# 💾 大小: {os.path.getsize(db_path) / 1024:.1f} KB")
    print(f"{'#'*80}")
    
    try:
        conn = sqlite3.connect(db_path)
        
        tables = get_tables(conn)
        print(f"\n发现 {len(tables)} 个表: {', '.join(tables)}")
        
        # 打印每个表
        for table in tables:
            print_table(conn, table, db_name)
        
        # 特殊统计
        if db_name == "bm25":
            print_bm25_stats(conn)
        elif db_name == "ingestion":
            print_ingestion_stats(conn)
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ 数据库错误: {e}")


def main():
    # 解析参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--path" and len(sys.argv) > 2:
            # 查看指定路径的数据库
            db_path = sys.argv[2]
            db_name = Path(db_path).stem
            view_database(db_path, db_name)
        elif sys.argv[1] in DB_PATHS:
            # 查看指定类型的数据库
            db_name = sys.argv[1]
            view_database(DB_PATHS[db_name], db_name)
        else:
            print(f"未知的数据库类型: {sys.argv[1]}")
            print(f"支持的类型: {', '.join(DB_PATHS.keys())}")
            print("或用法: python scripts/view_sqlite.py --path <db_file>")
    else:
        # 查看所有数据库
        for db_name, db_path in DB_PATHS.items():
            view_database(db_path, db_name)
    
    print(f"\n{'#'*80}")
    print("# ✅ 查看完成")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()
