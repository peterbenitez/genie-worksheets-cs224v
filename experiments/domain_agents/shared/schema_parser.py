"""Parse table_schema.txt - single source of truth for schema."""

import re
from pathlib import Path
from typing import Dict, List


def parse_table_schema(schema_path: Path) -> Dict[str, Dict]:
    """
    Parse SQL DDL file and extract table/column info.

    Args:
        schema_path: Path to table_schema.txt

    Returns:
        {"table_name": {"columns": ["col1", "col2", ...]}}
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"table_schema.txt not found at {schema_path}")

    content = schema_path.read_text()
    schema = {}

    table_pattern = r'CREATE TABLE (\w+)\s*\((.*?)\);'

    for match in re.finditer(table_pattern, content, re.DOTALL | re.IGNORECASE):
        table_name = match.group(1)
        columns_block = match.group(2)

        columns = []
        for line in columns_block.strip().split('\n'):
            line = line.strip().strip(',')
            if not line or line.upper().startswith(('PRIMARY', 'FOREIGN', 'CONSTRAINT')):
                continue
            col_match = re.match(r'"?(\w+)"?\s+', line)
            if col_match:
                columns.append(col_match.group(1))

        schema[table_name] = {"columns": columns}

    return schema


def get_table_keywords(table_name: str) -> List[str]:
    """
    Generate keywords from table name for matching user queries.

    Examples:
        my_table -> ["my", "table", "tables"]
        items -> ["items", "item"]
    """
    keywords = []
    parts = table_name.lower().replace('_', ' ').split()

    for part in parts:
        keywords.append(part)
        if part.endswith('s'):
            keywords.append(part[:-1])
        else:
            keywords.append(part + 's')

    return list(set(keywords))


if __name__ == "__main__":
    test_path = Path(__file__).parent.parent / "table_schema.txt"
    schema = parse_table_schema(test_path)
    for table, info in schema.items():
        print(f"{table}: {len(info['columns'])} columns")
        print(f"  keywords: {get_table_keywords(table)}")
