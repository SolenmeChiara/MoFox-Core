import asyncio
import datetime
import os
import re
import shutil
import sys
from pathlib import Path

import aiofiles
import orjson
from json_repair import repair_json

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.chat.knowledge.embedding_store import EmbeddingManager
from src.chat.knowledge.kg_manager import KGManager
from src.chat.knowledge.open_ie import OpenIE
from src.chat.knowledge.utils.hash import get_sha256
from src.common.logger import get_logger
from src.config.config import model_config
from src.llm_models.utils_model import LLMRequest

logger = get_logger("LPMM_LearningTool")
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(ROOT_PATH, "data", "lpmm_raw_data")
OPENIE_OUTPUT_DIR = os.path.join(ROOT_PATH, "data", "openie")
TEMP_DIR = os.path.join(ROOT_PATH, "temp", "lpmm_cache")

# ========== 性能配置参数 ==========
#
# 知识提取（步骤2：txt转json）并发控制
# - 控制同时进行的LLM提取请求数量
# - 推荐值: 3-10，取决于API速率限制
# - 过高可能触发429错误（速率限制）
MAX_EXTRACTION_CONCURRENCY = 12  # 提高并发以加快处理速度（OpenRouter/OpenAI速率限制较宽松）

# 数据导入（步骤3：生成embedding）性能配置
# - max_workers: 并发批次数（每批次并行处理）
# - chunk_size: 每批次包含的字符串数
# - 理论并发 = max_workers × chunk_size
# - 推荐配置:
#   * 高性能API（OpenAI）: max_workers=20-30, chunk_size=30-50
#   * 中等API: max_workers=10-15, chunk_size=20-30
#   * 本地/慢速API: max_workers=5-10, chunk_size=10-20
EMBEDDING_MAX_WORKERS = 20  # 并发批次数
EMBEDDING_CHUNK_SIZE = 30   # 每批次字符串数
# ===================================

# --- 缓存清理 ---


def clear_cache():
    """清理 lpmm_learning_tool.py 生成的缓存文件"""
    logger.info("--- 开始清理缓存 ---")
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            logger.info(f"成功删除缓存目录: {TEMP_DIR}")
        except OSError as e:
            logger.error(f"删除缓存时出错: {e}")
    else:
        logger.info("缓存目录不存在，无需清理。")
    logger.info("--- 缓存清理完成 ---")


# --- 模块一：数据预处理 ---


def process_text_file(file_path):
    # 尝试多种编码格式
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, encoding=encoding) as f:
                raw = f.read()
            logger.debug(f"成功使用 {encoding} 编码读取文件: {file_path}")
            return [p.strip() for p in raw.split("\n\n") if p.strip()]
        except (UnicodeDecodeError, UnicodeError):
            continue

    # 如果所有编码都失败，使用 utf-8 并忽略错误
    logger.warning(f"无法正确解码文件 {file_path}，使用 UTF-8 并忽略错误")
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    return [p.strip() for p in raw.split("\n\n") if p.strip()]


def preprocess_raw_data():
    logger.info("--- 步骤 1: 开始数据预处理 ---")
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    raw_files = list(Path(RAW_DATA_PATH).glob("*.txt"))
    if not raw_files:
        logger.warning(f"警告: 在 '{RAW_DATA_PATH}' 中没有找到任何 .txt 文件")
        return []

    all_paragraphs = []
    for file in raw_files:
        logger.info(f"正在处理文件: {file.name}")
        all_paragraphs.extend(process_text_file(file))

    unique_paragraphs = {get_sha256(p): p for p in all_paragraphs}
    logger.info(f"共找到 {len(all_paragraphs)} 个段落，去重后剩余 {len(unique_paragraphs)} 个。")
    logger.info("--- 数据预处理完成 ---")
    return unique_paragraphs


# --- 模块二：信息提取 ---


def _parse_and_repair_json(json_string: str) -> dict | None:
    """
    尝试解析JSON字符串，如果失败则尝试修复并重新解析。

    该函数首先会清理字符串，去除常见的Markdown代码块标记和思考过程标签，
    然后尝试直接解析。如果解析失败，它会调用 `repair_json`
    进行修复，并再次尝试解析。

    Args:
        json_string: 从LLM获取的、可能格式不正确的JSON字符串。

    Returns:
        解析后的字典。如果最终无法解析，则返回 None，并记录详细错误日志。
    """
    if not isinstance(json_string, str):
        logger.error(f"输入内容非字符串，无法解析: {type(json_string)}")
        return None

    # 1. 预处理：去除常见的多余字符
    cleaned_string = json_string.strip()

    # 🔧 修复：处理 </think> 标签（某些模型会输出思考过程）
    if "</think>" in cleaned_string:
        # 提取 </think> 之后的内容
        parts = cleaned_string.split("</think>")
        if len(parts) > 1:
            cleaned_string = parts[-1].strip()

    # 处理 Markdown 代码块标记
    if cleaned_string.startswith("```json"):
        cleaned_string = cleaned_string[7:].strip()
    elif cleaned_string.startswith("```"):
        cleaned_string = cleaned_string[3:].strip()

    if cleaned_string.endswith("```"):
        cleaned_string = cleaned_string[:-3].strip()

    # 2. 性能优化：乐观地尝试直接解析
    try:
        result = orjson.loads(cleaned_string)
        # 🔧 修复：确保返回的是字典而不是列表
        if isinstance(result, list):
            logger.warning(f"解析结果是列表而非字典，尝试提取第一个元素: {result}")
            if result and isinstance(result[0], dict):
                return result[0]
            else:
                logger.error(f"无法从列表中提取有效字典: {result}")
                return None
        return result
    except orjson.JSONDecodeError:
        logger.warning("直接解析JSON失败，将尝试修复...")

        # 3. 修复与最终解析
        repaired_json_str = ""
        try:
            repaired_json_str = repair_json(cleaned_string)
            result = orjson.loads(repaired_json_str)
            # 🔧 修复：确保返回的是字典而不是列表
            if isinstance(result, list):
                logger.warning(f"修复后的结果是列表而非字典，尝试提取第一个元素: {result}")
                if result and isinstance(result[0], dict):
                    return result[0]
                else:
                    logger.error(f"无法从列表中提取有效字典: {result}")
                    return None
            return result
        except Exception as e:
            # 4. 增强错误处理：记录详细的失败信息
            logger.error(f"修复并解析JSON后依然失败: {e}")
            logger.error(f"原始字符串 (清理后): {cleaned_string[:500]}...")  # 只记录前500字符
            logger.error(f"修复后尝试解析的字符串: {repaired_json_str[:500]}...")
            return None


def clean_xml_invalid_chars(text: str) -> str:
    """
    清理字符串中的非法 XML 字符。

    XML 1.0 规范允许的字符范围：
    - #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]

    Args:
        text: 需要清理的字符串

    Returns:
        清理后的字符串
    """
    if not isinstance(text, str):
        return text

    # 移除非法的 XML 字符（保留 tab、换行、回车以及正常可打印字符）
    # 参考：https://www.w3.org/TR/xml/#charsets
    illegal_xml_chars = re.compile(
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F\uD800-\uDFFF\uFFFE\uFFFF]'
    )
    return illegal_xml_chars.sub('', text)


def get_extraction_prompt(paragraph: str) -> str:
    return f"""
请从以下段落中提取关键信息。你需要提取两种类型的信息：
1.  **实体 (Entities)**: 识别并列出段落中所有重要的名词或名词短语。
2.  **三元组 (Triples)**: 以 [主语, 谓语, 宾语] 的格式，提取段落中描述关系或事实的核心信息。

请严格按照以下 JSON 格式返回结果，不要添加任何额外的解释或注释：
{{
    "entities": ["实体1", "实体2"],
    "triples": [["主语1", "谓语1", "宾语1"]]
}}

这是你需要处理的段落：
---
{paragraph}
---
"""


async def extract_info_async(pg_hash, paragraph, llm_api):
    """
    异步提取单个段落的信息（带缓存支持）

    Args:
        pg_hash: 段落哈希值
        paragraph: 段落文本
        llm_api: LLM请求实例

    Returns:
        tuple: (doc_item或None, failed_hash或None)
    """
    temp_file_path = os.path.join(TEMP_DIR, f"{pg_hash}.json")

    # 🔧 优化：使用异步文件检查，避免阻塞
    if os.path.exists(temp_file_path):
        try:
            async with aiofiles.open(temp_file_path, "rb") as f:
                content = await f.read()
                return orjson.loads(content), None
        except orjson.JSONDecodeError:
            # 缓存文件损坏，删除并重新生成
            try:
                os.remove(temp_file_path)
            except OSError:
                pass

    prompt = get_extraction_prompt(paragraph)
    content = None
    try:
        content, (_, _, _) = await llm_api.generate_response_async(prompt)

        # 调用封装好的函数处理JSON解析和修复
        extracted_data = _parse_and_repair_json(content)

        if extracted_data is None:
            raise ValueError("无法从LLM输出中解析有效的JSON数据")

        # 🔧 修复：清理实体和三元组中的非法 XML 字符，避免保存时 ExpatError
        entities = [clean_xml_invalid_chars(e) for e in extracted_data.get("entities", [])]
        triples = [
            [clean_xml_invalid_chars(str(item)) for item in triple]
            for triple in extracted_data.get("triples", [])
        ]

        doc_item = {
            "idx": pg_hash,
            "passage": clean_xml_invalid_chars(paragraph),
            "extracted_entities": entities,
            "extracted_triples": triples,
        }

        # 保存到缓存（异步写入）
        async with aiofiles.open(temp_file_path, "wb") as f:
            await f.write(orjson.dumps(doc_item))

        return doc_item, None
    except Exception as e:
        logger.error(f"提取信息失败：{pg_hash}, 错误：{e}")
        if content:
            logger.error(f"导致解析失败的原始输出: {content}")
        return None, pg_hash


async def extract_information(paragraphs_dict, model_set):
    """
    🔧 优化：使用真正的异步并发代替多线程

    这样可以：
    1. 避免 event loop closed 错误
    2. 更高效地利用 I/O 资源
    3. 与我们优化的 LLM 请求层无缝集成

    并发控制：
    - 使用信号量限制最大并发数为 5，防止触发 API 速率限制

    Args:
        paragraphs_dict: {hash: paragraph} 字典
        model_set: 模型配置
    """
    logger.info("--- 步骤 2: 开始信息提取 ---")
    os.makedirs(OPENIE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    failed_hashes, open_ie_docs = [], []

    # 🔧 关键修复：创建单个 LLM 请求实例，复用连接
    llm_api = LLMRequest(model_set=model_set, request_type="lpmm_extraction")

    # 🔧 并发控制：限制最大并发数，防止速率限制
    semaphore = asyncio.Semaphore(MAX_EXTRACTION_CONCURRENCY)

    async def extract_with_semaphore(pg_hash, paragraph):
        """带信号量控制的提取函数"""
        async with semaphore:
            return await extract_info_async(pg_hash, paragraph, llm_api)

    # 创建所有异步任务（带并发控制）
    tasks = [
        extract_with_semaphore(p_hash, paragraph)
        for p_hash, paragraph in paragraphs_dict.items()
    ]

    total = len(tasks)
    completed = 0

    logger.info(f"开始提取 {total} 个段落的信息（最大并发: {MAX_EXTRACTION_CONCURRENCY}）")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]正在提取信息...", total=total)

        # 🔧 优化：使用 asyncio.gather 并发执行所有任务
        # return_exceptions=True 确保单个失败不影响其他任务
        for coro in asyncio.as_completed(tasks):
            doc_item, failed_hash = await coro
            if failed_hash:
                failed_hashes.append(failed_hash)
            elif doc_item:
                open_ie_docs.append(doc_item)

            completed += 1
            progress.update(task, advance=1)

    if open_ie_docs:
        all_entities = [e for doc in open_ie_docs for e in doc["extracted_entities"]]
        num_entities = len(all_entities)
        avg_ent_chars = round(sum(len(e) for e in all_entities) / num_entities, 4) if num_entities else 0
        avg_ent_words = round(sum(len(e.split()) for e in all_entities) / num_entities, 4) if num_entities else 0
        openie_obj = OpenIE(docs=open_ie_docs, avg_ent_chars=avg_ent_chars, avg_ent_words=avg_ent_words)

        now = datetime.datetime.now()
        filename = now.strftime("%Y-%m-%d-%H-%M-%S-openie.json")
        output_path = os.path.join(OPENIE_OUTPUT_DIR, filename)
        async with aiofiles.open(output_path, "wb") as f:
            await f.write(orjson.dumps(openie_obj._to_dict()))
        logger.info(f"信息提取结果已保存到: {output_path}")
        logger.info(f"成功提取 {len(open_ie_docs)} 个段落的信息")

    if failed_hashes:
        logger.error(f"以下 {len(failed_hashes)} 个段落提取失败: {failed_hashes}")
    logger.info("--- 信息提取完成 ---")


# --- 模块三：数据导入 ---


async def import_data(openie_obj: OpenIE | None = None):
    """
    将OpenIE数据导入知识库（Embedding Store 和 KG）

    Args:
        openie_obj (Optional[OpenIE], optional): 如果提供，则直接使用这个OpenIE对象；
                                                 否则，将自动从默认文件夹加载最新的OpenIE文件。
                                                 默认为 None.
    """
    logger.info("--- 步骤 3: 开始数据导入 ---")
    # 使用配置的并发参数以加速 embedding 生成
    # max_workers: 并发批次数，chunk_size: 每批次处理的字符串数
    embed_manager = EmbeddingManager(max_workers=EMBEDDING_MAX_WORKERS, chunk_size=EMBEDDING_CHUNK_SIZE)
    kg_manager = KGManager()

    logger.info("正在加载现有的 Embedding 库...")
    try:
        embed_manager.load_from_file()
    except Exception as e:
        logger.warning(f"加载 Embedding 库失败: {e}。")

    logger.info("正在加载现有的 KG...")
    try:
        kg_manager.load_from_file()
    except Exception as e:
        logger.warning(f"加载 KG 失败: {e}。")

    try:
        if openie_obj:
            openie_data = openie_obj
            logger.info("已使用指定的 OpenIE 对象。")
        else:
            openie_data = OpenIE.load()
    except Exception as e:
        logger.error(f"加载OpenIE数据文件失败: {e}")
        return

    raw_paragraphs = openie_data.extract_raw_paragraph_dict()
    triple_list_data = openie_data.extract_triple_dict()

    # 🔧 修复：清理所有三元组中的非法 XML 字符
    cleaned_triple_list_data = {}
    for p_hash, triples in triple_list_data.items():
        cleaned_triples = [
            [clean_xml_invalid_chars(str(item)) for item in triple]
            for triple in triples
        ]
        cleaned_triple_list_data[p_hash] = cleaned_triples

    triple_list_data = cleaned_triple_list_data

    new_raw_paragraphs, new_triple_list_data = {}, {}
    stored_embeds = embed_manager.stored_pg_hashes
    stored_kgs = kg_manager.stored_paragraph_hashes

    for p_hash, raw_p in raw_paragraphs.items():
        if p_hash not in stored_embeds and p_hash not in stored_kgs:
            new_raw_paragraphs[p_hash] = raw_p
            new_triple_list_data[p_hash] = triple_list_data.get(p_hash, [])

    if not new_raw_paragraphs:
        logger.info("没有新的段落需要处理。")
    else:
        logger.info(f"去重完成，发现 {len(new_raw_paragraphs)} 个新段落。")
        logger.info("开始生成 Embedding...")
        await embed_manager.store_new_data_set(new_raw_paragraphs, new_triple_list_data)
        embed_manager.rebuild_faiss_index()
        embed_manager.save_to_file()
        logger.info("Embedding 处理完成！")

        logger.info("开始构建 KG...")
        kg_manager.build_kg(new_triple_list_data, embed_manager)
        kg_manager.save_to_file()
        logger.info("KG 构建完成！")

    logger.info("--- 数据导入完成 ---")


def import_from_specific_file():
    """从用户指定的 openie.json 文件导入数据"""
    file_path = input("请输入 openie.json 文件的完整路径: ").strip()

    if not os.path.exists(file_path):
        logger.error(f"文件路径不存在: {file_path}")
        return

    if not file_path.endswith(".json"):
        logger.error("请输入一个有效的 .json 文件路径。")
        return

    try:
        logger.info(f"正在从 {file_path} 加载 OpenIE 数据...")
        openie_obj = OpenIE.load()
        asyncio.run(import_data(openie_obj=openie_obj))
    except Exception as e:
        logger.error(f"从指定文件导入数据时发生错误: {e}")


# --- 主函数 ---


def rebuild_faiss_only():
    """仅重建 FAISS 索引，不重新导入数据"""
    logger.info("--- 重建 FAISS 索引 ---")
    # 重建索引不需要并发参数（不涉及 embedding 生成）
    embed_manager = EmbeddingManager()

    logger.info("正在加载现有的 Embedding 库...")
    try:
        embed_manager.load_from_file()
        logger.info("开始重建 FAISS 索引...")
        embed_manager.rebuild_faiss_index()
        embed_manager.save_to_file()
        logger.info("✅ FAISS 索引重建完成！")
    except Exception as e:
        logger.error(f"重建 FAISS 索引时发生错误: {e}")


def main():
    # 使用 os.path.relpath 创建相对于项目根目录的友好路径
    raw_data_relpath = os.path.relpath(RAW_DATA_PATH, os.path.join(ROOT_PATH, ".."))
    openie_output_relpath = os.path.relpath(OPENIE_OUTPUT_DIR, os.path.join(ROOT_PATH, ".."))

    print("=== LPMM 知识库学习工具 ===")
    print(f"1. [数据预处理] -> 读取 .txt 文件 (来源: ./{raw_data_relpath}/)")
    print(f"2. [信息提取] -> 提取信息并存为 .json (输出至: ./{openie_output_relpath}/)")
    print("3. [数据导入] -> 从 openie 文件夹自动导入最新知识")
    print("4. [全流程] -> 按顺序执行 1 -> 2 -> 3")
    print("5. [指定导入] -> 从特定的 openie.json 文件导入知识")
    print("6. [清理缓存] -> 删除所有已提取信息的缓存")
    print("7. [重建索引] -> 仅重建 FAISS 索引（数据已导入时使用）")
    print("0. [退出]")
    print("-" * 30)
    choice = input("请输入你的选择 (0-7): ").strip()

    if choice == "1":
        preprocess_raw_data()
    elif choice == "2":
        paragraphs = preprocess_raw_data()
        if paragraphs:
            # 🔧 修复：使用 asyncio.run 调用异步函数
            asyncio.run(extract_information(paragraphs, model_config.model_task_config.lpmm_qa))
    elif choice == "3":
        asyncio.run(import_data())
    elif choice == "4":
        paragraphs = preprocess_raw_data()
        if paragraphs:
            # 🔧 修复：使用 asyncio.run 调用异步函数
            asyncio.run(extract_information(paragraphs, model_config.model_task_config.lpmm_qa))
            asyncio.run(import_data())
    elif choice == "5":
        import_from_specific_file()
    elif choice == "6":
        clear_cache()
    elif choice == "7":
        rebuild_faiss_only()
    elif choice == "0":
        sys.exit(0)
    else:
        print("无效输入，请重新运行脚本。")


if __name__ == "__main__":
    main()
