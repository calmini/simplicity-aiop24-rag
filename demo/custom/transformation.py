from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import BaseNode,Document,TextNode
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.async_utils import run_jobs
from llama_index.legacy.llms import OpenAILike as OpenAI
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from .template import SUMMARY_EXTRACT_TEMPLATE, KEYWORD_EXTRACT_TEMPLATE
from collections import namedtuple


class CustomFilePathExtractor(BaseExtractor):
    last_path_length: int = 4

    def __init__(self, last_path_length: int = 4, **kwargs):
        super().__init__(last_path_length=last_path_length, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomFilePathExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        metadata_list = []
        for node in nodes:
            node.metadata["file_path"] = "/".join(
                node.metadata["file_path"].split("/")[-self.last_path_length :]
            )
            metadata_list.append(node.metadata)
        return metadata_list


class CustomTitleExtractor(BaseExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomTitleExtractor"

    # 将Document的第一行作为标题
    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        try:
            document_title = nodes[0].text.split("\n")[0]
            last_file_path = nodes[0].metadata["file_path"]
        except:
            document_title = ""
            last_file_path = ""
        metadata_list = []
        for node in nodes:
            if node.metadata["file_path"] != last_file_path:
                document_title = node.text.split("\n")[0]
                last_file_path = node.metadata["file_path"]
            node.metadata["document_title"] = document_title
            metadata_list.append(node.metadata)

        return metadata_list

## 可以在这里加上一些metadata -> 支持Filter的工具
class CustomDocumentIdExtractor(BaseExtractor):
    documentIdFileMapper: Dict[str, str]

    def __init__(self, documentIdFileMapper: Dict[str, str], **kwargs):
        super().__init__(documentIdFileMapper=documentIdFileMapper, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomDocumentIdExtractor"
    
    async def aextract(self, nodes: Sequence[BaseNode]):
        metadata_list = []
        for node in nodes:
            file_path = node.metadata["file_path"]
            doc_id = self.documentIdFileMapper.get(file_path, "")
            node.metadata["doc_id"] = doc_id
            metadata_list.append(node.metadata)
        return metadata_list
    
    def extract(self, nodes: Sequence[BaseNode]):
        metadata_list = []
        for node in nodes:
            file_path = node.metadata["file_path"]
            doc_id = self.documentIdFileMapper.get(file_path, "")
            node.metadata["doc_id"] = doc_id
            metadata_list.append(node.metadata)
        return metadata_list
    

class CustomSummaryExtractor(BaseExtractor):

    llm: OpenAI = Field(description="The LLM to use for generation.")
    summaries: List[str] = Field(
        description="List of summaries to extract: 'self', 'prev', 'next'"
    )
    prompt_template: str = Field(
        default=SUMMARY_EXTRACT_TEMPLATE,
        description="Template to use when generating summaries.",
    )

    _self_summary: bool = PrivateAttr()
    _prev_summary: bool = PrivateAttr()
    _next_summary: bool = PrivateAttr()

    def __init__(self, llm: Optional[LLM], prompt_template: str, summaries: List[str],**kwargs: Any) -> List[BaseNode]:

        if not all(s in ["self", "prev", "next"] for s in summaries):
            raise ValueError("summaries must be one of ['self', 'prev', 'next']")
        
        self._self_summary = "self" in summaries
        self._prev_summary = "prev" in summaries
        self._next_summary = "next" in summaries
        
        super().__init__(
            llm = llm,
            prompt_template = prompt_template,
            summaries = summaries,
            **kwargs)
    
    @classmethod
    def class_name(cls) -> str:
        return "CustomSummaryExtractor"
    
    async def _agenerate_node_summary(self, node: BaseNode) -> str:
        """Generate a summary for a node."""
        if self.is_text_node_only and not isinstance(node, TextNode):
            return ""

        context_str = node.get_content(metadata_mode=self.metadata_mode)
        summary = await self.llm.apredict(
            PromptTemplate(template=self.prompt_template), context_str=context_str
        )

        return summary.strip()
    
    def _generate_node_summary(self, node: BaseNode) -> str:
        if self.is_text_node_only and not isinstance(node, TextNode):
            return ""

        context_str = node.get_content(metadata_mode=self.metadata_mode)
        return self.llm.predict(
            PromptTemplate(template=self.prompt_template), context_str = context_str
        ).strip()

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        if not all(isinstance(node, TextNode) for node in nodes):
            raise ValueError("Only `TextNode` is allowed for `Summary` extractor")

        node_summaries_jobs = []
        for node in nodes:
            node_summaries_jobs.append(self._agenerate_node_summary(node))

        node_summaries = await run_jobs(
            node_summaries_jobs,
            show_progress=self.show_progress,
            workers=self.num_workers,
        )

        # Extract node-level summary metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        for i, metadata in enumerate(metadata_list):
            if i > 0 and self._prev_summary and node_summaries[i - 1]:
                metadata["prev_section_summary"] = node_summaries[i - 1]
            if i < len(nodes) - 1 and self._next_summary and node_summaries[i + 1]:
                metadata["next_section_summary"] = node_summaries[i + 1]
            if self._self_summary and node_summaries[i]:
                metadata["section_summary"] = node_summaries[i]

        return metadata_list
    

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        if not all(isinstance(node, TextNode) for node in nodes):
            raise ValueError("Only `TextNode` is allowed for `Summary` extractor")

        node_summaries = []
        for node in nodes:
            node_summaries.append(self._generate_node_summary(node))

        # node_summaries = await run_jobs(
        #     node_summaries_jobs,
        #     show_progress=self.show_progress,
        #     workers=self.num_workers,
        # )
        
        # Extract node-level summary metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        for i, metadata in enumerate(metadata_list):
            if i > 0 and self._prev_summary and node_summaries[i - 1]:
                metadata["prev_section_summary"] = node_summaries[i - 1]
            if i < len(nodes) - 1 and self._next_summary and node_summaries[i + 1]:
                metadata["next_section_summary"] = node_summaries[i + 1]
            if self._self_summary and node_summaries[i]:
                metadata["section_summary"] = node_summaries[i]
        
        return metadata_list

class SimpleGivenKeywordExtractor:

    def __init__(self, 
                 keywordsMapper: Dict[str, List[namedtuple]]):
        self.given_keywords = keywordsMapper
    
    def _extract(self, text: str):
        keywordsInclude = list()
        keys = self.given_keywords.keys()
        for key in keys:
            keywordTuple = self.given_keywords[key]
            if (key in text or any(x.fullKeywordEn in text or x.fullKeywordCn in text for x in keywordTuple)):
                keywordsInclude.extend(self.given_keywords[key])
        
        return keywordsInclude

    def extract_documents(self, documents: Document) -> List[str]:
        return self._extract(documents.text)
    
    def extract_node(self, node: TextNode) -> List[str]:
        return self._extract(node.text)

class GLMKeywordExtractor:

    def __init__(self, 
                 llm: OpenAI,
                 query_template: str = KEYWORD_EXTRACT_TEMPLATE) -> None:
        self._llm = llm
        self._query_template = query_template

    async def aextract_query(self, query: str) -> List[str]:
        # 通过LLM提取问题中的关键词
        response = await self._llm.acomplete(
            PromptTemplate(self._query_template).format(query_str=query)
        )
        if response.text != "不存在名词":
            return response.text.split(",")
        
        return list()

