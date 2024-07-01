from typing import Dict, List, Tuple
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

from custom.template import SUMMARY_EXTRACT_TEMPLATE
from custom.transformation import CustomFilePathExtractor, CustomTitleExtractor


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
    embed_model: BaseEmbedding,
    template: str = None,
    vector_store: BasePydanticVectorStore = None,
) -> IngestionPipeline:
    transformation = [
        SentenceSplitter(chunk_size=1024, chunk_overlap=50),
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
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    sub_chunk_size: List[int],
    sub_chunk_overlap: int
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
        use_async = True,
        embedding_model = embedding_model
    )

    return vector_store_index, node_reference
