from typing import Dict, List, Tuple, Sequence, Set
from collections import namedtuple
from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import IndexNode
from custom.transformation import CustomDocumentIdExtractor
from custom.transformation import CustomFilePathExtractor, CustomTitleExtractor, SimpleGivenKeywordExtractor, GLMKeywordExtractor
import jieba
from rank_bm25 import BM25Okapi
from llama_index.legacy.llms import OpenAILike


def read_data(path: str = "data") -> list[Document]:
    reader = SimpleDirectoryReader(
        input_dir=path,
        recursive=True,
        required_exts=[
            ".txt",
        ],
    )
    return reader.load_data()


def build_pipeline(
    llm: LLM,
    chunk_size: int,
    chunk_overlap_size: int,
    documentFileMapper: Dict[str, str],
    embed_model: BaseEmbedding,
    template: str = None,
    vector_store: BasePydanticVectorStore = None,
) -> IngestionPipeline:
    transformation = [
        SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap_size),
        CustomDocumentIdExtractor(documentIdFileMapper=documentFileMapper),
        CustomTitleExtractor(metadata_mode=MetadataMode.EMBED),
        CustomFilePathExtractor(last_path_length=4, metadata_mode=MetadataMode.EMBED),
        # SummaryExtractor(
        #     llm=llm,
        #     metadata_mode=MetadataMode.EMBED,
        #     prompt_template=template or SUMMARY_EXTRACT_TEMPLATE,
        # ),
        embed_model,
    ]

    return IngestionPipeline(transformations=transformation, vector_store=vector_store)


async def build_vector_store(
    config: dict, reindex: bool = False
) -> tuple[AsyncQdrantClient, QdrantVectorStore]:
    client = AsyncQdrantClient(
        # url=config["QDRANT_URL"],
        location=":memory:"
    )
    if reindex:
        try:
            await client.delete_collection(config["COLLECTION_NAME"] or "aiops24")
        except UnexpectedResponse as e:
            print(f"Collection not found: {e}")

    try:
        await client.create_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            vectors_config=models.VectorParams(
                size=config["VECTOR_SIZE"] or 1024, distance=models.Distance.DOT
            ),
        )
    except UnexpectedResponse:
        print("Collection already exists")
    return client, QdrantVectorStore(
        aclient=client,
        collection_name=config["COLLECTION_NAME"] or "aiops24",
        parallel=4,
        batch_size=32,
    )


def build_vector_store_index(
    documents: List[Document],
    embedding_model: BaseEmbedding,
    parent_chunk_size: int = 1024,
    parent_chunk_overlap: int = 0,
    sub_chunk_size: List[int] = [128, 256, 512],
    sub_chunk_overlap: int = 20
) -> Tuple[VectorStoreIndex, Dict[str, IndexNode]]:
    
    # 定义parent Chunk
    parentSplitter = SentenceSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap)
    base_nodes = parentSplitter.get_nodes_from_documents(documents)
    for idx, node in enumerate(base_nodes):
        node.id_ = f"node-{idx}"
    # 定义child Chunk
    sub_chunk_sizes = sub_chunk_size
    sub_node_parsers = [
        SentenceSplitter(chunk_size=c, chunk_overlap=sub_chunk_overlap) for c in sub_chunk_sizes
    ]

    all_nodes = []
    for base_node in base_nodes:
        for n in sub_node_parsers:
            sub_nodes = n.get_nodes_from_documents([base_node])
            sub_inodes = [
                IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
            ]
            all_nodes.extend(sub_inodes)

        # also add original node to node
        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)

    node_reference = {node.node_id: node for node in all_nodes}
    vector_store_index = VectorStoreIndex(
        nodes = all_nodes,
        use_async = False,
        embedding_model = embedding_model
    )

    return vector_store_index, node_reference

def contains_keyword(query_str: str, keywords: List[namedtuple]) -> list:
    may_include_keywords = [kw for kw in keywords if kw.keyword in query_str]
    non_include_keywords = list()
    # 可能存在重叠的问题, 从小到大排序后移除存在重叠的关键词
    sorted_may_include_keywords = sorted(may_include_keywords, key=lambda x: len(x.keyword), reverse=False)
    for i in range(len(sorted_may_include_keywords)):
        kw_str = sorted_may_include_keywords[i].keyword
        if not any(kw_str in s.keyword for s in sorted_may_include_keywords[i+1:]):
            non_include_keywords.append(sorted_may_include_keywords[i])
    return non_include_keywords
        

async def retrieve_doc_by_bm25_with_keyword(
    document_collections: Sequence[Document],
    llm: OpenAILike,
    keywords: List[namedtuple],
    kw2doc: Dict[str, Set[str]],
    doc_cache: Dict[str, List[str]], # 预先通过jieba分词后的文档
    query_str: str,
    top_k: int = 100
) -> List[str]:
    # 关键词过滤+bm25排序召回
    glm_keyword_extractor = GLMKeywordExtractor(llm)

    # 如果query中包含给定关键字
    kw_in_query = contains_keyword(query_str, keywords)
    if (len(kw_in_query) > 0):
        # 找出包含关键词的文档
        doc_contains_kw = set.union(*[kw2doc[kw.keyword] for kw in kw_in_query])
    # query中不包含关键词
    else:
        kw_probably = await glm_keyword_extractor.aextract_query(query_str)
        if (len(kw_probably)):
            doc_contains_kw = list()
            for kw in kw_probably:
                doc_contains_kw.extend([doc.doc_id for doc in document_collections if kw in doc.text])
    
    # 通过bm25对文档进行排序
    if (len(doc_contains_kw) > 0):
        # 已经分好词的文本
        tokenized_corpus = [doc_cache[doc_id] for doc_id in doc_contains_kw]
        # 已经分好词的query
        tokenized_query = list(jieba.cut_for_search(query_str))
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores = bm25.get_scores(tokenized_query)
        sorted_doc_with_score = sorted(zip(doc_contains_kw, doc_scores), key = lambda x: -x[1])
        return [z[0] for z in sorted_doc_with_score[:top_k]], kw_in_query
    
    return list(), list()
    