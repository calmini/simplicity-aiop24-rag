import re
import os
from collections import namedtuple,defaultdict
from typing import Dict, List
from llama_index.core.schema import Document
import jieba

from custom.transformation import SimpleGivenKeywordExtractor

# 定义NamedTuple结构
Acronym = namedtuple('Acronym', ['keyword', 'fullKeywordEn', 'fullKeywordCn'])

def parse_acronyms(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 跳过非缩略语行
    lines = list(filter(lambda x: x != "\n", lines))
    # 检查是否存在一对多的关系...
    # grouped_list = [tuple(lines[i:i+2]) for i in range(1, len(lines), 2)]
    acronyms = []
    i = 1
    while i < len(lines):
        if not contains_chinese(lines[i]):
            keyword = lines[i].replace("\n", "")
            i += 1
            while (i < len(lines) and contains_chinese(lines[i])):
                zipedEnCn = lines[i].replace("\n", "")
                # 进一步分割英文全称和中文含义
                try:
                    full_keyword_en, full_keyword_cn = separate_text(zipedEnCn)
                except:
                    print(f"{zipedEnCn}无法分离")
                    break
                acronyms.append(Acronym(keyword=keyword,
                                         fullKeywordEn=full_keyword_en.strip(),
                                         fullKeywordCn=full_keyword_cn.strip()))
                i += 1

    return acronyms



def contains_chinese(text):
    # 正则表达式匹配中文字符
    if re.search(r'[\u4e00-\u9fff]', text):
        return True
    else:
        return False
    
def separate_text(text):
    # 使用正则表达式匹配英文部分和中文部分
    # 假设英文部分由字母、斜杠和下划线组成，中文部分由中文字符组成
    pattern = re.compile(r'([0-9A-Za-z&-,/_ ]+)(.*?)(?=$|[^\u4e00-\u9fa5])')
    match = pattern.search(text)
    if match:
        english_part = match.group(1).strip()  # 英文部分
        chinese_part = text[len(english_part):]
        return [english_part, chinese_part]
    else:
        return None

def find_log_files(directory):
    dirLists = list()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'log.txt':
                dirLists.append(os.path.join(root, file))
    return dirLists

# 构建关键词集合
def build_keyword_ref(directory):
    keywordsMapper = defaultdict(set)
    # 首先增加文件的顶级目录
    logfilesContainsKeyword = find_log_files(directory)
    if (len(logfilesContainsKeyword)):
        for file in logfilesContainsKeyword:
            acronyms_entities = parse_acronyms(file)
            if (len(acronyms_entities)):
                for acronyms_entity in acronyms_entities:
                    keywordsMapper[acronyms_entity.keyword].add(acronyms_entity)
    return keywordsMapper

# 构建关键词和文档的映射关系
def build_kw_doc_ref(
    keywords_ref: Dict[str, List[namedtuple]],
    document_collections: List[Document]) -> defaultdict:

    keyword2docId = defaultdict(set)
    keyword_extractor = SimpleGivenKeywordExtractor(keywordsMapper=keywords_ref)

    for doc in document_collections:
        if (doc.metadata['file_name'] == "log.txt"):
            continue # 过滤

        keywords_in_dt = keyword_extractor.extract_documents(doc)
        for kw in keywords_in_dt:
            keyword2docId[kw.keyword].add(doc.doc_id)
        
    return keyword2docId

def build_doc_tokenization_cache(document_collections: List[Document]) -> Dict[str, List[str]]:
    doc_cache = dict()
    for doc in document_collections:
        doc_cache[doc.doc_id] = list(jieba.cut_for_search(doc.text.replace("\n", "")))
    
    return doc_cache