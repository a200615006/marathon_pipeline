```markdown
# RAG-LlamaIndex 项目

基于 LlamaIndex 构建的 RAG（检索增强生成）系统，支持文档检索和智能问答，含2025创新大赛专属RAG模块。

## 功能特性

- 📚 支持多种文档格式（PDF、TXT、MD、DOCX等）
- 🔍 混合检索（向量检索 + BM25）
- 🎯 Qwen3 Reranker 重排序
- 💾 Milvus 向量数据库存储
- 🔄 节点分割与父子关系处理
- 🤖 SiliconFlow LLM 集成
- 🚀 2025创新大赛专属流程（PDF转MD→向量库构建→HTTP服务启动）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# PDF处理依赖（ libreoffice 用于格式转换）
sudo apt update && sudo apt install -y libreoffice
```

### 2. 配置参数（含性能优化）
编辑 `config/config.yaml` 及 `src/app.py`，关键配置分三类：

#### 2.1 数据库相关参数（config.yaml）
| 参数                | 含义说明                                                                 |
|---------------------|--------------------------------------------------------------------------|
| chunk_size          | 文本分块大小（基于整句切分，实际大小在该数值上下浮动）                   |
| chunk_overlap       | 文本分块重叠长度                                                         |
| max_tokens          | 传递给Qwen3-Embedding-0.6B，过大文件会切分到该长度                       |
| if_ivfpq            | 是否用ivf_flat索引（启用时nlist按节点数自动算，nprobe固定10）             |
| paths.documents_dir  | 文档存放路径（默认：`./data`）                                          |
| paths.milvus_dir     | Milvus数据库路径（默认：`./milvus_test`）                               |
| paths.pdf_model_path | Docling模型路径（默认：`/root/autodl-tmp/docling_model`）                |

#### 2.2 检索相关参数
- **混合检索TopK**：在 `src/app.py` 的 `lifespan` 内，通过 `bm25_retriever`/`vector_retriever`/`hybrid_retriever` 设置
- **重排序配置**（config.yaml）：
  | 参数              | 含义说明                                                                 |
  |-------------------|--------------------------------------------------------------------------|
  | if_vllmReranker   | 是否用vllm加速Reranker                                                   |
  | batch_size        | Reranker输入批量大小                                                     |
  | models.embedding  | Qwen3-Embedding路径（默认：`/root/autodl-tmp/Qwen3-Embedding-0.6B`）     |
  | models.reranker   | Qwen3-Reranker路径（默认：`/root/autodl-tmp/Qwen3-Reranker-4B`）         |

#### 2.3 生成相关参数（src/app.py）
| 参数                | 含义说明                                                                 |
|---------------------|--------------------------------------------------------------------------|
| llm                 | 生成模型（默认：`Qwen/Qwen3-32B`）                                       |
| max_length          | 生成文本最大长度（默认：`8192`）                                         |
| api.siliconflow_key | SiliconFlow API密钥（需替换为个人密钥）                                  |

llm 提示词在marathon_rag/src/query_engine.py中配置 注：refine 提示词未使用

### 3. 模型下载
需提前下载3个核心模型，执行命令：
```bash
# 1. Docling（PDF解析）
modelscope download --model docling/docling-base --local_dir /root/autodl-tmp/docling_model

# 2. Qwen3-Embedding（向量生成）
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir /root/autodl-tmp/Qwen3-Embedding-0.6B

# 3. Qwen3-Reranker（结果重排序）
modelscope download --model Qwen/Qwen3-Reranker-4B --local_dir /root/autodl-tmp/Qwen3-Reranker-4B
```
> 提示：若修改模型路径，需同步更新 `config.yaml` 中对应配置项

### 4. 使用方式

#### 方式A：原项目通用流程
1. 构建数据库：
```bash
python -m src.build_database
```
2. 启动查询系统：
```bash
python -m src.main
```

#### 方式B：2025创新大赛RAG模块专属流程
1. **PDF转MD格式**：将 `paths.documents_dir` 中的PDF转为MD，执行：
```bash
python src/pdf2md.py
```
2. **构建向量数据库**：基于MD文档生成Milvus索引，执行：
```bash
python src/build_database.py
```
3. **启动HTTP服务**：启动FastAPI本机服务供检索调用，执行：
```bash
python src/app.py
```

## 项目结构

```
src/
├── build_database.py    # 数据库构建（通用+创新大赛共用）
├── build_and_query.py   # 一次性构建+查询（原项目）
├── main.py              # 原项目RAG查询入口
├── app.py               # 创新大赛：FastAPI服务启动（含检索配置）
├── pdf2md.py            # 创新大赛：PDF转MD工具
├── config.py            # 配置管理（读取config.yaml）
├── data_loader.py       # 文档加载（多格式支持）
├── index_builder.py     # 索引构建（含ivf_flat索引逻辑）
├── retiever.py          # 检索器（混合检索+Qwen3重排序）
├── query_engine.py      # 查询引擎（LLM调用+结果整合）
├── llm.py               # SiliconFlow LLM接口
└── utils.py             # 工具函数（日志、路径处理等）
config/
└── config.yaml          # 核心配置文件（数据库/检索/模型路径）
```

## 核心功能

### 数据库构建
- 自动检测文档目录，支持批量预处理
- 并行加载文档，PDF需先经 `pdf2md.py` 转换
- 构建Milvus向量索引，支持ivf_flat优化查询速度
- 保存节点信息至 `saved_nodes` 目录，供检索复用

### RAG查询（创新大赛模块增强）
- 混合检索（向量+BM25关键词）提升召回率
- Qwen3-Reranker重排序优化结果相关性
- FastAPI服务化部署，支持外部系统调用
- 长文档自动节点分割，避免上下文截断

## 注意事项

1. **创新大赛模块前置条件**：必须先执行 `pdf2md.py` 转换PDF，否则数据库构建会失败
2. **API密钥有效性**：`siliconflow_key` 需在SiliconFlow平台申请，过期会导致LLM调用失败
3. **存储空间要求**：Milvus数据库需预留至少2倍文档大小的磁盘空间
4. **依赖版本兼容**：libreoffice版本建议≥7.0，避免PDF转MD格式错乱

## 故障排除

- **PDF转MD失败**：检查Docling模型路径是否正确，或 libreoffice 是否安装
- **数据库构建报错**：确认 `documents_dir` 存在文档，且权限为可读
- **HTTP服务启动失败**：检查端口是否被占用，或 `app.py` 中模型配置是否正确
- **LLM无响应**：验证 `siliconflow_key` 有效性，及网络是否能访问SiliconFlow接口

---

快速启动创新大赛模块：`python src/pdf2md.py && python src/build_database.py && python src/app.py`
```
