from collections import namedtuple
from typing import Dict, List, Sequence, Set
from llama_index.core.callbacks.base import CallbackManager
import qdrant_client

from llama_index.core.llms.llm import LLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.vector_stores import VectorStoreQuery
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, Document
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode

from custom.template import QA_TEMPLATE
from ingestion import retrieve_doc_by_bm25_with_keyword


class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = await self._vector_store.aquery(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = self._vector_store.query(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores
    
    # 支持带过滤器的查询
    async def aretrieve_with_filters(self, query_bundle: QueryBundle, filters: Filter) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )

        query_result = await self._vector_store.aquery(vector_store_query, qdrant_filters=filters)
        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores


class MyCustomRecursiveRetriever(BaseRetriever):

    def __init__(
            self,
            node_references: Dict[str, IndexNode],
            vector_store_index: VectorStoreIndex,
            similarity_top_k: int = 2,
    ):
        self.vector_store_index = vector_store_index
        self.similarity_top_k = similarity_top_k
        self._retriever = vector_store_index.as_retriever(similarity_top_k=self.similarity_top_k)
        self._recursive_retriever = RecursiveRetriever(
            root_id="custom-vector-recursive-starter",
            retriever_dict = {"custom-vector-recursive-starter": self._retriever},
            node_dict=node_references,
            verbose=False
        )
        super().__init__()
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return await self._recursive_retriever.aretrieve(query_bundle)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._recursive_retriever.retrieve(query_bundle)


def generate_qdrant_filters(doc_ids: List[str]) -> Filter:
    return Filter(
        must = [FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]
    )


async def generation_with_knowledge_retrieval(
    query_str: str,
    retriever: BaseRetriever,
    llm: LLM,
    qa_template: str = QA_TEMPLATE,
    reranker: BaseNodePostprocessor = None,
    debug: bool = False,
    progress=None,
) -> CompletionResponse:
    query_bundle = QueryBundle(query_str=query_str)
    node_with_scores = await retriever.aretrieve(query_bundle)
    if debug:
        print(f"retrieved:\n{node_with_scores}\n------")
    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_bundle)
        if debug:
            print(f"reranked:\n{node_with_scores}\n------")
    context_str = "\n\n".join(
        [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores]
    )
    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )
    ret = await llm.acomplete(fmt_qa_prompt)
    if progress:
        progress.update(1)
    return ret

async def generation_with_recursive_knowledge_retrieval(
    query_str: str,
    retriever: BaseRetriever,
    llm: LLM,
    qa_template: str = QA_TEMPLATE,
    reranker: BaseNodePostprocessor = None,
    debug: bool = False,
    progress=None,
) -> CompletionResponse:
    query_bundle = QueryBundle(query_str=query_str)
    node_with_scores = await retriever.aretrieve(query_bundle)
    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_bundle)
    
    context_str = "\n\n".join(
        [f"{node.text}" for node in node_with_scores]
    )
    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )
    ret = await llm.acomplete(fmt_qa_prompt)
    if progress:
        progress.update(1)
    return ret


async def generate_with_knowledge_retrieve_enhanced_by_keyword(
    document_collections: Sequence[Document],
    keywords: List[namedtuple],
    kw2doc: Dict[str, Set[str]],
    doc_cache: Dict[str, List[str]],
    query_str: str,
    retriever: BaseRetriever,
    llm: LLM,
    qa_template: str = QA_TEMPLATE,
    progress=None
) -> CompletionResponse:
    
    query_bundle = QueryBundle(query_str=query_str)
    
    # 获取相关文档id
    doc_id_collections, keyword_in_query = retrieve_doc_by_bm25_with_keyword(
        document_collections = document_collections,
        llm = llm,
        keywords = keywords,
        kw2doc = kw2doc,
        doc_cache = doc_cache,
        query_str = query_str,
        top_k = 100
    )

    # 增加了关键词过滤
    if (len(doc_id_collections)):
        filters = generate_qdrant_filters(doc_id_collections)
        node_with_scores = await retriever.aretrieve_with_filters(query_bundle, filters)
        if (len(node_with_scores) == 0):
            # 兜底
            node_with_scores = await retriever.aretrieve(query_bundle)
    else:
        node_with_scores = await retriever.aretrieve(query_bundle)
    

    context_str = "\n\n".join(
        [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores]
    )

    keyword_descriptions = "\n".join([f"关键词: {kw.keyword}, 英文全称: {kw.fullKeywordEn}, 中文全称: {kw.fullKeywordCn}" for kw in keyword_in_query]) \
        if len(keyword_in_query) else ""

    fmt_qa_prompt = PromptTemplate(qa_template).format(
        keyword_descriptions = keyword_descriptions, context_str=context_str, query_str=query_str
    )
    
    ret = await llm.acomplete(fmt_qa_prompt)
    if progress:
        progress.update(1)  
    return ret


