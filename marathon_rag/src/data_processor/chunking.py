from typing import List, Dict, Any
# 從 models 模組導入 TextChunk，避免循環引用
from .models import TextChunk

def optimize_chunks_after_splitting(original_chunks, min_chunk_size=256, max_overlap=200):
    """
    在现有分割器基础上进行后处理优化：
    1. 合并过短的分块（小于min_chunk_size）
    2. 添加智能重叠
    """
    if not original_chunks:
        return []

    # 首先将所有分块转换为统一格式
    processed_chunks = []
    for chunk in original_chunks:
        if isinstance(chunk, str):
            processed_chunks.append({
                'text': chunk,
                'original_chunk': chunk,
                'is_textchunk': False
            })
        elif hasattr(chunk, 'content'):
            processed_chunks.append({
                'text': chunk.content,
                'original_chunk': chunk,
                'is_textchunk': True,
                'metadata': chunk.metadata,
                'path_titles': chunk.path_titles,
                'chunk_id': chunk.chunk_id
            })
        else:
            processed_chunks.append({
                'text': str(chunk),
                'original_chunk': chunk,
                'is_textchunk': False
            })

    optimized_chunks = []
    pending_merge = []  # 待合并的短分块
    current_merged_length = 0

    for i, chunk_info in enumerate(processed_chunks):
        chunk_text = chunk_info['text']
        chunk_len = len(chunk_text)

        # 如果分块太短，先暂存等待合并
        if chunk_len < min_chunk_size:
            pending_merge.append(chunk_info)
            current_merged_length += chunk_len
        else:
            # 先处理待合并的分块
            if pending_merge:
                merged_chunk = merge_pending_chunks(pending_merge, optimized_chunks, min_chunk_size)
                if merged_chunk:
                    optimized_chunks.append(merged_chunk)
                pending_merge = []
                current_merged_length = 0

            # 添加当前正常分块
            optimized_chunks.append(chunk_info['original_chunk'])

    # 处理剩余的待合并分块
    if pending_merge:
        merged_chunk = merge_pending_chunks(pending_merge, optimized_chunks, min_chunk_size)
        if merged_chunk:
            optimized_chunks.append(merged_chunk)

    # 添加分块间重叠（只在真正需要时）
    if len(optimized_chunks) > 1:
        final_chunks = add_intelligent_overlap(optimized_chunks, max_overlap)
    else:
        final_chunks = optimized_chunks

    return final_chunks

def merge_pending_chunks(pending_chunks, existing_chunks, min_chunk_size):
    """合并过短的分块"""
    if not pending_chunks:
        return None

    # 计算合并后的总长度
    total_length = sum(len(chunk['text']) for chunk in pending_chunks)

    # 如果合并后仍然很短，尝试与上一个分块合并
    if total_length < min_chunk_size and existing_chunks:
        last_chunk = existing_chunks[-1]

        # 如果是TextChunk对象
        if hasattr(last_chunk, 'content'):
            # 不能直接修改，需要创建新的TextChunk
            merged_text = "\n".join([chunk['text'] for chunk in pending_chunks])
            new_content = last_chunk.content + "\n" + merged_text

            new_chunk = TextChunk(
                chunk_id=f"merged_{last_chunk.chunk_id}",
                content=new_content,
                path_titles=last_chunk.path_titles,
                source_node_index=last_chunk.source_node_index,
                chunk_index=last_chunk.chunk_index,
                metadata={
                    **last_chunk.metadata,
                    'merged_from': [last_chunk.chunk_id] + [chunk.get('chunk_id', 'unknown') for chunk in pending_chunks if chunk.get('chunk_id')],
                    'is_merged': True,
                    'original_text_length': last_chunk.metadata.get('original_text_length', 0) + total_length
                }
            )
            # 替换最后一个分块
            existing_chunks[-1] = new_chunk
            return None
        else:
            # 普通字符串，直接合并
            merged_text = "\n".join([chunk['text'] for chunk in pending_chunks])
            existing_chunks[-1] = last_chunk + "\n" + merged_text
            return None

    # 创建新的合并分块
    return create_merged_chunk(pending_chunks)

def create_merged_chunk(pending_chunks):
    """创建合并后的分块"""
    if not pending_chunks:
        return None

    merged_text = "\n".join([chunk['text'] for chunk in pending_chunks])

    # 如果是TextChunk对象，需要保持原有结构
    if pending_chunks[0]['is_textchunk']:
        first_chunk = pending_chunks[0]['original_chunk']

        # 提取标题部分（如果有）
        content_parts = first_chunk.content.split('\n\n', 1)
        if len(content_parts) > 1:
            title_part = content_parts[0]
            new_content = title_part + "\n\n" + merged_text
        else:
            new_content = merged_text

        return TextChunk(
            chunk_id=f"merged_{first_chunk.chunk_id}",
            content=new_content,
            path_titles=first_chunk.path_titles,
            source_node_index=first_chunk.source_node_index,
            chunk_index=first_chunk.chunk_index,
            metadata={
                **first_chunk.metadata,
                'merged_from': [chunk.get('chunk_id', 'unknown') for chunk in pending_chunks if chunk.get('chunk_id')],
                'is_merged': True,
                'original_text_length': sum(len(chunk['text']) for chunk in pending_chunks)
            }
        )
    else:
        return merged_text

def add_intelligent_overlap(chunks, max_overlap, min_body_size: int = 256):
    """在分块之间添加智能重叠，同时保护正文长度"""
    if len(chunks) <= 1:
        return chunks

    def _extract_text(c):
        if hasattr(c, 'content'):
            return get_body_text(c)
        return get_chunk_text(c)

    enhanced = [chunks[0]]
    for i in range(1, len(chunks)):
        cur = chunks[i]
        prev = chunks[i - 1]

        prev_text = _extract_text(prev)
        cur_text = _extract_text(cur)

        # 正文过短时不添加 overlap，优先保证密度
        if len(cur_text) < max(min_body_size // 2, 128):
            enhanced.append(cur)
            continue

        if len(prev_text) > max_overlap and not is_content_already_included(prev_text, cur_text, max_overlap):
            overlap_text = extract_overlap_from_text(prev_text, max_overlap)
            if overlap_text and len(overlap_text) > 50 and not cur_text.startswith(overlap_text):
                combined_body = overlap_text + "\n\n" + cur_text
                if len(combined_body) >= min_body_size // 2:
                    if hasattr(cur, 'content'):
                        # TextChunk
                        enhanced_chunk = TextChunk(
                            chunk_id=(cur.chunk_id + "_overlap") if hasattr(cur, 'chunk_id') else "overlap",
                            content=add_path_title_once(tuple(getattr(cur, 'path_titles', [])), combined_body),
                            path_titles=getattr(cur, 'path_titles', []),
                            source_node_index=getattr(cur, 'source_node_index', None),
                            chunk_index=getattr(cur, 'chunk_index', None),
                            metadata={
                                **getattr(cur, 'metadata', {}),
                                'overlap_added': True,
                                'overlap_length': len(overlap_text),
                                'overlap_source': getattr(prev, 'chunk_id', 'unknown')
                            }
                        )
                        enhanced.append(enhanced_chunk)
                        continue
                    else:
                        # 普通字符串
                        enhanced.append(overlap_text + "\n\n" + cur)
                        continue

        enhanced.append(cur)

    return enhanced


def get_chunk_text(chunk):
    """获取分块的文本内容"""
    if isinstance(chunk, str):
        return chunk
    elif hasattr(chunk, 'content'):
        # 只获取正文部分（去掉标题）
        content = chunk.content
        if '\n\n' in content:
            return content.split('\n\n', 1)[1]
        return content
    else:
        return str(chunk)

def extract_overlap_text(chunk, max_overlap):
    """从分块末尾提取合适的重叠文本"""
    text = get_chunk_text(chunk)

    if len(text) <= max_overlap:
        return text

    # 从末尾开始查找合适的截断点
    punctuation_marks = ["。", "！", "？", "\n", ".", "!", "?", ";", "；", ":", "：", "，", ","]

    # 查找最近的标点符号（从max_overlap位置开始向前找）
    start_pos = max(0, len(text) - max_overlap - 100)  # 多找一些范围
    overlap_candidate = text[start_pos:]

    # 在候选文本中查找第一个标点
    for i, char in enumerate(overlap_candidate):
        if char in punctuation_marks:
            overlap_start = start_pos + i + 1  # 从标点后开始
            return text[overlap_start:]

    # 如果没有找到合适的标点，使用固定长度的重叠
    return text[-max_overlap:]


def merge_chunks_by_path(chunks: List[TextChunk], max_chunk_size: int = 4096, max_overlap: int = 400) -> List[TextChunk]:
    """
    根据路径合并相同路径的分块，并移除重复的路径标题
    """
    if not chunks:
        return []

    # 按路径分组
    chunks_by_path = {}
    for chunk in chunks:
        path_key = tuple(chunk.path_titles)  # 使用元组作为键
        if path_key not in chunks_by_path:
            chunks_by_path[path_key] = []
        chunks_by_path[path_key].append(chunk)

    merged_chunks = []

    # 对每个路径组进行合并
    for path_key, path_chunks in chunks_by_path.items():
        if len(path_chunks) == 1:
            # 单个分块，无需合并
            merged_chunks.append(path_chunks[0])
            continue

        # 按源节点索引和分块索引排序，确保顺序正确
        path_chunks.sort(key=lambda x: (x.source_node_index, x.chunk_index))

        current_merged_content = ""
        current_metadata = path_chunks[0].metadata.copy()
        chunks_to_merge = []

        for i, chunk in enumerate(path_chunks):
            # 移除分块内容中的路径标题部分（只保留正文）
            chunk_content = remove_path_title_from_content(chunk.content, chunk.path_titles)

            # 检查当前分块是否与已合并内容有重叠（避免重复）
            if current_merged_content and is_content_overlapping(current_merged_content, chunk_content):
                # 移除重叠部分
                chunk_content = remove_overlapping_content(current_merged_content, chunk_content)

            # 检查合并后是否超过最大长度
            if len(current_merged_content) + len(chunk_content) > max_chunk_size:
                # 创建合并后的分块（只在开头添加一次路径标题）
                if chunks_to_merge:
                    final_content = add_path_title_once(path_key, current_merged_content)
                    merged_chunk = create_merged_chunk_from_list(chunks_to_merge, final_content, current_metadata)
                    merged_chunks.append(merged_chunk)

                # 重置状态
                current_merged_content = chunk_content
                current_metadata = chunk.metadata.copy()
                chunks_to_merge = [chunk]
            else:
                # 添加分隔符（如果是第一个分块则不添加）
                if current_merged_content:
                    current_merged_content += "\n\n"
                current_merged_content += chunk_content
                chunks_to_merge.append(chunk)

                # 更新元数据
                current_metadata['original_text_length'] = current_metadata.get('original_text_length', 0) + chunk.metadata.get('chunk_text_length', 0)
                current_metadata['full_content_length'] = len(current_merged_content)

        # 处理最后一批分块
        if chunks_to_merge:
            final_content = add_path_title_once(path_key, current_merged_content)
            merged_chunk = create_merged_chunk_from_list(chunks_to_merge, final_content, current_metadata)
            merged_chunks.append(merged_chunk)

    # 在合并后的分块间添加最小长度兜底
    merged_chunks = ensure_min_body_length_on_textchunks(merged_chunks, 200) ### 保证最短块长度

    # 在合并后的分块间添加智能重叠（放在最小长度兜底之后）
    if len(merged_chunks) > 1:
        merged_chunks = add_intelligent_overlap_to_merged(merged_chunks, max_overlap)

    return merged_chunks

def is_content_overlapping(existing_content: str, new_content: str) -> bool:
    """检查新内容是否与现有内容有重叠"""
    if not existing_content or not new_content:
        return False

    # 检查新内容开头是否与现有内容结尾重复
    overlap_check_length = min(200, len(existing_content), len(new_content))
    if overlap_check_length == 0:
        return False

    existing_end = existing_content[-overlap_check_length:]
    new_start = new_content[:overlap_check_length]

    # 简单的字符串匹配可能误判，增加更智能的检查
    # 只有当重叠部分超过一定长度且看起来是真正的重复时才返回True
    if existing_end == new_start:
        # 检查重叠长度，太短的重复可能是巧合
        if overlap_check_length < 50:
            return False
        
        # 检查重叠部分是否包含完整的句子或段落
        # 如果重叠部分以标点符号结尾，更可能是真正的重叠
        punctuation_marks = ["。", "！", "？", ".", "!", "?", ";", "；", ":", "："]
        if overlap_check_length > 0 and existing_end[-1] not in punctuation_marks:
            # 重叠不以标点结尾，可能是巧合匹配
            return False
            
        return True

    return False

def remove_overlapping_content(existing_content: str, new_content: str) -> str:
    """移除新内容中与现有内容重叠的部分"""
    if not existing_content or not new_content:
        return new_content

    # 查找最大重叠长度
    max_overlap = min(200, len(existing_content), len(new_content))
    for overlap_length in range(max_overlap, 0, -1):
        if existing_content.endswith(new_content[:overlap_length]):
            return new_content[overlap_length:]

    return new_content


def remove_path_title_from_content(content: str, path_titles: List[str]) -> str:
    """
    从分块内容中移除路径标题部分，只保留正文
    改进版本：更智能地识别真正的路径标题，避免移除内容中的有效标题
    """
    if not path_titles or not content:
        return content

    # 构建路径标题模式（可能出现在内容开头的各种形式）
    path_patterns = []

    # 1. 完整的路径标题格式：标题1 > 标题2 > 标题3
    full_path = " > ".join(path_titles)
    path_patterns.append(full_path + "\n\n")
    path_patterns.append(full_path + "\n")

    # 2. 单独的标题（逐级检查）
    for i in range(len(path_titles)):
        partial_path = " > ".join(path_titles[i:])
        path_patterns.append(partial_path + "\n\n")
        path_patterns.append(partial_path + "\n")

    # 3. 每个单独的标题
    for title in path_titles:
        path_patterns.append(title + "\n\n")
        path_patterns.append(title + "\n")

    # 首先检查内容开头是否有明确的路径标题模式
    cleaned_content = content
    for pattern in path_patterns:
        if cleaned_content.startswith(pattern):
            # 确认这是真正的路径标题而不是内容中的标题
            # 检查移除后的内容是否仍然有意义（不是过度移除）
            temp_content = cleaned_content[len(pattern):].strip()
            if len(temp_content) > 20:  # 确保移除后还有足够的内容
                cleaned_content = temp_content
                break

    # 移除内容中间可能出现的重复路径标题（更保守的策略）
    for pattern in path_patterns:
        # 只在明显的段落分隔位置移除，避免移除内容中的标题
        pattern_with_newlines = "\n\n" + pattern
        while pattern_with_newlines in cleaned_content:
            # 检查移除后是否会影响内容完整性
            position = cleaned_content.find(pattern_with_newlines)
            if position > 0 and position + len(pattern_with_newlines) < len(cleaned_content):
                # 确保移除的是真正的重复标题，而不是内容的一部分
                before_pattern = cleaned_content[:position]
                after_pattern = cleaned_content[position + len(pattern_with_newlines):]
                
                # 如果移除后前后内容仍然连贯，则移除
                if len(before_pattern.strip()) > 10 and len(after_pattern.strip()) > 10:
                    cleaned_content = before_pattern + "\n\n" + after_pattern
                else:
                    break
            else:
                break

    return cleaned_content.strip()

def add_path_title_once(path_key: tuple, content: str) -> str:
    """
    只在合并后的内容开头添加一次路径标题
    """
    path_titles = list(path_key)
    if not path_titles:
        return content

    full_path = " > ".join(path_titles)
    return f"{full_path}\n\n{content}"

def create_merged_chunk_from_list(chunks: List[TextChunk], merged_content: str, metadata: Dict[str, Any]) -> TextChunk:
    """从多个分块创建合并后的分块"""
    if not chunks:
        return None

    first_chunk = chunks[0]
    last_chunk = chunks[-1]

    return TextChunk(
        chunk_id=f"path_merged_{first_chunk.chunk_id}_to_{last_chunk.chunk_id}",
        content=merged_content,
        path_titles=first_chunk.path_titles,
        source_node_index=first_chunk.source_node_index,
        chunk_index=first_chunk.chunk_index,
        metadata={
            **metadata,
            'merged_from': [chunk.chunk_id for chunk in chunks],
            'is_path_merged': True,
            'merged_chunk_count': len(chunks),
            'first_source_node': first_chunk.source_node_index,
            'last_source_node': last_chunk.source_node_index,
            'path_titles_preserved': True  # 标记路径标题已优化处理
        }
    )

def add_intelligent_overlap_to_merged(chunks: List[TextChunk], max_overlap: int, min_body_size: int = 256) -> List[TextChunk]:
    """为合并后的分块添加智能重叠，避免重复且不致使正文过短"""
    if len(chunks) <= 1:
        return chunks

    enhanced_chunks = [chunks[0]]

    for i in range(1, len(chunks)):
        current_chunk = chunks[i]
        previous_chunk = chunks[i - 1]

        previous_text = get_body_text(previous_chunk)
        current_text = get_body_text(current_chunk)

        # 当前正文太短时，不添加 overlap
        if len(current_text) < max(min_body_size // 2, 128):
            enhanced_chunks.append(current_chunk)
            continue

        if len(previous_text) > max_overlap and not is_content_already_included(previous_text, current_text, max_overlap):
            overlap_text = extract_overlap_from_text(previous_text, max_overlap)

            if overlap_text and len(overlap_text) > 50 and not current_text.startswith(overlap_text):
                combined_text = overlap_text + "\n\n" + current_text
                # 添加 overlap 后正文仍需合格
                if len(combined_text) >= min_body_size // 2:
                    enhanced_chunk = TextChunk(
                        chunk_id=current_chunk.chunk_id + "_overlap",
                        content=add_path_title_once(tuple(current_chunk.path_titles), combined_text),
                        path_titles=current_chunk.path_titles,
                        source_node_index=current_chunk.source_node_index,
                        chunk_index=current_chunk.chunk_index,
                        metadata={
                            **current_chunk.metadata,
                            'overlap_added': True,
                            'overlap_length': len(overlap_text),
                            'overlap_source': previous_chunk.chunk_id
                        }
                    )
                    enhanced_chunks.append(enhanced_chunk)
                    continue

        enhanced_chunks.append(current_chunk)

    return enhanced_chunks


def is_content_already_included(previous_text: str, current_text: str, max_overlap: int) -> bool:
    """
    检查当前分块是否已经包含了前一个分块的内容
    避免添加重复的重叠
    """
    if not previous_text or not current_text:
        return False

    # 检查当前分块开头是否与前一个分块结尾有重复
    previous_end = previous_text[-min(max_overlap * 2, len(previous_text)):]  # 检查前一个分块的结尾部分
    current_start = current_text[:min(max_overlap * 2, len(current_text))]    # 检查当前分块的开头部分

    # 如果当前分块开头已经包含了前一个分块结尾的内容，就不需要再添加重叠
    for overlap_length in range(50, min(len(previous_end), len(current_start)) + 1):
        if previous_end[-overlap_length:] == current_start[:overlap_length]:
            return True

    return False

def extract_overlap_from_text(text: str, max_overlap: int) -> str:
    """从文本末尾提取重叠内容，确保不会重复"""
    if len(text) <= max_overlap:
        return text

    # 从末尾开始查找合适的截断点
    punctuation_marks = ["。", "！", "？", "\n", ".", "!", "?", ";", "；", ":", "：", "，", ","]

    # 在最后max_overlap + 200字符范围内查找标点（扩大搜索范围）
    search_start = max(0, len(text) - max_overlap - 200)
    search_text = text[search_start:]

    # 优先查找段落分隔符
    if "\n\n" in search_text:
        last_double_newline = search_text.rfind("\n\n")
        if last_double_newline != -1:
            overlap_start = search_start + last_double_newline + 2
            return text[overlap_start:]

    # 其次查找句子结束标点
    for i in range(len(search_text) - 1, -1, -1):
        if search_text[i] in punctuation_marks:
            overlap_start = search_start + i + 1
            # 确保重叠文本有足够长度
            if len(text) - overlap_start >= 50:
                return text[overlap_start:]

    # 如果没有找到合适的标点，使用固定长度的重叠（但确保不会太短）
    return text[-min(max_overlap, len(text)):]


def get_body_text(chunk: TextChunk) -> str:
    """获取纯正文（移除路径标题部分）"""
    try:
        titles = getattr(chunk, 'path_titles', []) or []
        content = getattr(chunk, 'content', '') or ''
        return remove_path_title_from_content(content, titles).strip()
    except Exception:
        # 兜底
        if hasattr(chunk, 'content'):
            return chunk.content
        return str(chunk)

def ensure_min_body_length_on_textchunks(chunks: List[TextChunk], min_chunk_size: int = 256) -> List[TextChunk]:
    """
    对 TextChunk 列表做二次兜底合并：
    - 如果某个块的正文长度 < min_chunk_size，则尝试与后继块合并；
    - 若已到末尾，则与前一个块合并；
    """
    if not chunks:
        return []

    def body_len(c: TextChunk) -> int:
        return len(get_body_text(c))

    merged: List[TextChunk] = []
    i = 0
    while i < len(chunks):
        cur = chunks[i]
        if body_len(cur) >= min_chunk_size or i == len(chunks) - 1:
            # 足够长或最后一块（最后一块短的话留待与前一块合并）
            merged.append(cur)
            i += 1
            continue

        # 当前块正文过短，尝试与后续块合并
        j = i + 1
        collected = [cur]
        total_body = body_len(cur)

        while j < len(chunks) and total_body < min_chunk_size:
            collected.append(chunks[j])
            total_body += body_len(chunks[j])
            j += 1

        # 合并 collected
        if len(collected) == 1:
            # 只有自己，且不是最后一块（已在上面处理），正常追加
            merged.append(cur)
            i += 1
        else:
            # 构造合并内容（仅保留一次路径标题，正文拼接用空行）
            first = collected[0]
            merged_body = "\n\n".join(get_body_text(c) for c in collected if get_body_text(c))
            final_content = add_path_title_once(tuple(first.path_titles), merged_body)

            new_chunk = TextChunk(
                chunk_id=f"minlen_merged_{first.chunk_id}_to_{collected[-1].chunk_id}",
                content=final_content,
                path_titles=first.path_titles,
                source_node_index=first.source_node_index,
                chunk_index=first.chunk_index,
                metadata={
                    **first.metadata,
                    'is_minlen_merged': True,
                    'merged_from': [c.chunk_id for c in collected],
                    'merged_chunk_count': len(collected),
                    'original_text_length': sum(len(get_body_text(c)) for c in collected)
                }
            )
            merged.append(new_chunk)
            i = j

    # 如果最后一块依然过短，尝试与前一块合并
    if len(merged) >= 2 and len(get_body_text(merged[-1])) < min_chunk_size:
        last = merged.pop()
        prev = merged.pop()

        merged_body = "\n\n".join([get_body_text(prev), get_body_text(last)])
        final_content = add_path_title_once(tuple(prev.path_titles), merged_body)

        new_chunk = TextChunk(
            chunk_id=f"minlen_tail_merged_{prev.chunk_id}_to_{last.chunk_id}",
            content=final_content,
            path_titles=prev.path_titles,
            source_node_index=prev.source_node_index,
            chunk_index=prev.chunk_index,
            metadata={
                **prev.metadata,
                'is_minlen_tail_merged': True,
                'merged_from': [prev.chunk_id, last.chunk_id],
                'merged_chunk_count': 2,
                'original_text_length': len(get_body_text(prev)) + len(get_body_text(last))
            }
        )
        merged.append(new_chunk)

    return merged
