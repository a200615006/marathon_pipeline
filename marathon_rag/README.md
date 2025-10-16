# RAG-LlamaIndex 项目

基于 LlamaIndex 构建的 RAG（检索增强生成）系统，支持文档检索和智能问答。

## 功能特性

- 📚 支持多种文档格式（PDF、TXT、MD、DOCX等）
- 🔍 混合检索（向量检索 + BM25）
- 🎯 Qwen3 Reranker 重排序
- 💾 Milvus 向量数据库存储
- 🔄 节点分割与父子关系处理
- 🤖 SiliconFlow LLM 集成

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

sudo apt get
sudo apt install libreoffice

```

### 2. 配置参数

编辑 `config/config.yaml` 文件，设置：
- API密钥（SiliconFlow）
- 模型路径
- 文档目录
- 数据库路径

### 3. 使用方式



#### 方式:分步操作
1. 构建数据库：
```bash
python -m src.build_database
```

2. 启动查询系统：
```bash
python -m src.main
```

#### 方式三：直接查询（需先构建数据库）
```bash
python -m src.main
```

## 项目结构

```
src/
├── build_database.py    # 数据库构建入口
├── build_and_query.py   # 一次性构建+查询入口
├── main.py             # RAG查询入口
├── config.py           # 配置管理
├── data_loader.py      # 文档加载
├── index_builder.py    # 索引构建
├── retiever.py         # 检索器实现
├── query_engine.py     # 查询引擎
├── llm.py              # LLM接口
└── utils.py            # 工具函数
```

## 核心功能

### 数据库构建
- 自动检测文档目录
- 并行加载和预处理文档
- 构建 Milvus 向量索引
- 保存节点信息供检索使用

### RAG查询
- 混合检索策略（向量+关键词）
- 智能节点分割处理长文档
- 重排序优化结果质量
- 交互式问答界面

## 注意事项

1. **首次运行**：需要先构建数据库才能进行查询
2. **文档格式**：支持常见文本格式，PDF需要额外依赖
3. **API配置**：确保 config.yaml 中的 API 密钥正确
4. **存储空间**：数据库构建需要足够的磁盘空间

## 故障排除

- 如果数据库构建失败，检查文档目录和权限
- 如果查询出错，确认数据库是否已正确构建
- API 相关问题检查网络连接和密钥配置

---

开始使用：`python -m src.build_and_query`


下载 docling 模型
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir ./autodl-tmp
modelscope download --model Qwen/Qwen3-Reranker-4B --local_dir ./autodl-tmp