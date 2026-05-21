import os
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_go as tsgo
import concurrent.futures

LANGUAGE_MAP = {
    '.py': Language(tspython.language()),
    '.js': Language(tsjavascript.language()),
    '.jsx': Language(tsjavascript.language()),
    '.go': Language(tsgo.language()),
}

MAX_LINES = 50
OVERLAP_LINES = 10

FUNCTION_NODE_TYPES = {
    '.py': ['function_definition', 'class_definition'],
    '.js': ['function_declaration', 'class_declaration', 'arrow_function'],
    '.jsx': ['function_declaration', 'class_declaration', 'arrow_function'],
    '.go': ['function_declaration', 'method_declaration'],
}


def split_large_chunk(chunk: dict, source_lines: list[str]) -> list[dict]:
    start = chunk['start_line'] - 1
    end = chunk['end_line']
    total_lines = end - start

    if total_lines <= MAX_LINES:
        return [chunk]

    sub_chunks = []
    current_start = start

    while current_start < end:
        current_end = min(current_start + MAX_LINES, end)
        content = '\n'.join(source_lines[current_start:current_end])

        sub_chunks.append({
            'file_path': chunk['file_path'],
            'start_line': current_start + 1,
            'end_line': current_end,
            'node_type': chunk['node_type'],
            'content': content,
        })

        # Reduce overlap to improve performance
        current_start = current_end

    return sub_chunks


def chunk_file(file_path: str) -> list:
    _, ext = os.path.splitext(file_path)

    if ext not in LANGUAGE_MAP:
        print(f"Unsupported file type: {ext}")
        return []

    language = LANGUAGE_MAP[ext]
    parser = Parser(language)

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        source_lines = f.readlines()

    source_code = ''.join(source_lines)
    source_bytes = bytes(source_code, 'utf-8')
    tree = parser.parse(source_bytes)

    chunks = []
    node_types = FUNCTION_NODE_TYPES.get(ext, [])

    def traverse(node):
        if node.type in node_types:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            chunk_text = source_code[node.start_byte:node.end_byte]

            chunks.append({
                'file_path': file_path,
                'start_line': start_line,
                'end_line': end_line,
                'node_type': node.type,
                'content': chunk_text,
            })

        for child in node.children:
            traverse(child)

    traverse(tree.root_node)

    final_chunks = []
    for chunk in chunks:
        final_chunks.extend(split_large_chunk(chunk, source_lines))

    return final_chunks


def chunk_files(file_paths: list[str]) -> list:
    all_chunks = []

    def process_file(file_path):
        try:
            chunks = chunk_file(file_path)
            print(f"Chunked {file_path} into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"Error chunking {file_path}: {e}")
            return []

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_file, file_paths)

    for result in results:
        all_chunks.extend(result)

    return all_chunks