import asyncio

from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import resolve_embed_model
from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.ingestion import build_vector_store_index
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from pipeline.rag import MyCustomRecursiveRetriever, generate_with_knowledge_retrieve_enhanced_by_keyword
from pipeline.keyword import build_keyword_ref, build_doc_tokenization_cache, build_kw_doc_ref

async def main():
    config = dotenv_values(".env")

    # 初始化 LLM 嵌入模型 和 Reranker
    llm = OpenAI(
        api_key=config["GLM_KEY"],
        model="glm-4",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )
    embeding = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_folder="./",
        embed_batch_size=128,
    )
    # embeding = resolve_embed_model("local:BAAI/bge-small-zh-v1.5")
    Settings.embed_model = embeding

    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_store = await build_vector_store(config, reindex=False)

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "aiops24"
    )

    if collection_info.points_count == 0:
        data = read_data("data")
        # 构建文档id与file_path的映射
        documentIdFileMapper = dict()
        for dt in data:
            documentIdFileMapper[dt.metadata["file_path"]] = dt.doc_id

        # 增加文档映射
        pipeline = build_pipeline(llm, 1024, 128, documentIdFileMapper, embeding, vector_store=vector_store)
        # 暂时停止实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        await pipeline.arun(documents=data, show_progress=True, num_workers=1)
        # 恢复实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(len(data))    
    
    # data = read_data("data")
    # print(len(data))
    # vector_store_index, node_references = build_vector_store_index(documents=data, embedding_model='local')

    retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=3)

    # 构建关键词集合
    keywordCollection = build_keyword_ref("data")
    keywords = list()
    for _, v in keywordCollection.items():
        keywords.extend(list(v))
    kw2doc = build_kw_doc_ref(keywordCollection, document_collections=data)

    doc_tokenization_cache = build_doc_tokenization_cache(data)
    print("构建文档分词缓存完成")
    # retriever = MyCustomRecursiveRetriever(
    #     node_references = node_references,
    #     vector_store_index = vector_store_index,
    #     similarity_top_k = 3
    # )

    queries = read_jsonl("question.jsonl")

    # 生成答案
    print("Start generating answers...")

    results = []
    for query in tqdm(queries, total=len(queries)):
        # result = await generation_with_knowledge_retrieval(
        #     query["query"], retriever, llm
        # )
        result = await generate_with_knowledge_retrieve_enhanced_by_keyword(
            document_collections = data,
            keywords = keywords,
            kw2doc = kw2doc,
            doc_cache = doc_tokenization_cache,
            query_str = query["query"],
            retriever = retriever,
            llm = llm
        )
        # result = await generation_with_recursive_knowledge_retrieval(
        #     query["query"], retriever, llm
        # )
        results.append(result)

    # 处理结果
    save_answers(queries, results, "submit_result.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
