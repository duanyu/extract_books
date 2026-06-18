# Extract Books

从 LLM 参数记忆中抽取完整文本的实验项目。

## 原理

通过给 LLM 提供种子文本（书名+作者，或开头段落），利用模型在训练时记忆的文本内容，迭代式地续写出完整原文。核心流程：

1. **Phase 1（可选）**：概率探测 —— 取前 50% 预测后 50%，评估相似度决定是否继续
2. **Phase 2**：滑动窗口迭代抽取 —— 每次取最后 N 个 token 作为上下文，逐步续写直到全文末尾
3. **格式化输出**：按中英文句末标点分句，双换行分隔

## 使用

```bash
pip install openai python-dotenv modelscope
```

在 `.env` 中配置 API 密钥：

```
API_KEY=your_api_key_here
```

修改 `extract_memory.py` 中 `__main__` 部分的配置后运行：

```bash
python extract_memory.py
```

## 项目结构

```
.
├── book/                  # 原始书籍文本（ground truth）
├── output/                # 抽取结果输出
├── extract_memory.py      # 核心抽取脚本
└── .env                   # API 密钥配置（不纳入版本控制）
```
