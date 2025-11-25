# 开源数据集推荐

## 预训练数据集

### 1. 中文维基百科
- **来源**: Kaggle
- **链接**: https://www.kaggle.com/datasets/notoookay/chinese-wikipedia-2023/data
- **规模**: 2.62 GB JSONL格式
- **描述**: 2024年12月最新的中文维基百科数据，已按200字符分块处理
- **适用**: 通用中文知识预训练

### 2. Leipzig中文语料库
- **来源**: Leipzig Corpora Collection
- **链接**: https://corpora.uni-leipzig.de/en?corpusId=cmn-cn_web_2019
- **规模**: 61.34M tokens, 2.55M sentences
- **描述**: 2019年中文网络文本语料库
- **适用**: 通用中文语言建模

### 3. Chinese Tiny LLM (CT-LLM) 数据集
- **来源**: MAP-CC (Massive Appropriate Pretraining Chinese Corpus)
- **链接**: https://chinese-tiny-llm.github.io/
- **规模**: 800亿中文token + 300亿英文token + 100亿代码token
- **描述**: 专为中文优化的超大规模预训练语料库
- **适用**: 大规模中文语言模型预训练

## 对话数据集

### 4. LCCC (Large-scale Cleaned Chinese Conversation)
- **来源**: 清华大学
- **链接**: https://github.com/thu-coai/CDial-GPT
- **规模**: 
  - LCCC-base: 6.7M utterances, 68.6M characters
  - LCCC-large: 32.9M utterances, 380M characters
- **描述**: 大规模清洗后的中文对话数据集
- **适用**: 对话系统训练

### 5. KdConv (Knowledge-driven Conversation)
- **来源**: 清华大学
- **链接**: https://github.com/thu-coai/KdConv
- **规模**: 4.5K对话，86K utterances，平均19轮
- **描述**: 基于知识图谱的中文多领域对话数据集
- **适用**: 知识增强对话系统

### 6. CPED (Chinese Personalized and Emotional Dialogue)
- **来源**: 华南理工大学
- **链接**: https://github.com/scutcyr/CPED
- **规模**: 12K对话，133K utterances
- **描述**: 包含情感、个性特征的中文对话数据集
- **适用**: 个性化对话系统

## 指令微调数据集

### 7. BELLE (Chinese Instruction Dataset)
- **来源**: 链家科技
- **链接**: https://github.com/LianjiaTech/BELLE
- **规模**: 
  - train_1M_CN: 100万条
  - train_3.5M_CN: 350万条
  - train_10M_CN: 1000万条
- **描述**: 基于Self-Instruct方法构建的中文指令数据集
- **适用**: 中文指令微调

### 8. COIG (Chinese Open Instruction Generalist)
- **来源**: 北京智源人工智能研究院
- **链接**: https://huggingface.co/datasets/BAAI/COIG
- **规模**: 
  - Translated Instructions: 67,798
  - Exam Instructions: 63,532
  - Human Value Alignment: 34,471
  - Counterfactual Correction: 13,653
  - Leetcode Instructions: 11,737
- **描述**: 多领域中文指令数据集
- **适用**: 中文指令微调

### 9. pCLUE
- **来源**: CLUE benchmark
- **链接**: https://github.com/CLUEbenchmark/pCLUE
- **规模**: 1.2M样本
- **描述**: 基于NLP任务数据集和Prompt模板生成
- **适用**: 中文NLP任务指令微调

### 10. Firefly
- **来源**: 个人项目
- **链接**: https://github.com/yangjianxin1/Firefly
- **规模**: 1.65M训练样本
- **描述**: 整合23个中文数据集，包含传统文学生成任务
- **适用**: 中文生成任务

## 电商领域数据集

### 11. 电商对话语料库
- **来源**: 淘宝公开数据
- **链接**: https://www.heywhale.com/mw/dataset/5e50cb410e2b66002c203007
- **规模**: 100万训练对，1万验证对，1万测试对
- **描述**: 淘宝客服对话语料库
- **适用**: 电商客服对话系统

### 12. 天池开放数字商业知识图谱
- **来源**: 阿里云天池
- **链接**: https://tianchi.aliyun.com/dataset/122271?lang=en-us
- **规模**: 
  - OpenBG500: 1.24M训练样本
  - OpenBG500-L: 47.41M训练样本
  - OpenBG(Full): 260.3M训练样本
- **描述**: 电商领域知识图谱数据集
- **适用**: 电商知识增强

### 13. 京东电商数据
- **来源**: 张永辉教授
- **链接**: http://yongfeng.me/dataset/
- **规模**: 6000万评论，200万用户，10万商品
- **描述**: 京东用户评论和购买行为数据
- **适用**: 商品评论分析

## 代码和数据获取

### HuggingFace Datasets
```python
from datasets import load_dataset

# 中文对话数据
dialogue_dataset = load_dataset("lccc", "base")

# BELLE指令数据
belle_dataset = load_dataset("BelleGroup/train_0.5M_CN")

# COIG指令数据
coig_dataset = load_dataset("BAAI/COIG")
```

### 直接下载
```bash
# 使用wget或curl下载数据集
wget https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/resolve/main/train_0.5M_CN.jsonl

# 或使用git lfs
git lfs install
git clone https://huggingface.co/datasets/BelleGroup/train_0.5M_CN
```

## 数据预处理建议

### 1. 数据清洗
- 过滤敏感词汇和不当内容
- 去除特殊字符和表情符号
- 修正语法错误

### 2. 数据平衡
- 确保各类型数据比例合理
- 避免特定领域数据过拟合
- 保持数据多样性

### 3. 质量控制
- 建立数据质量评估标准
- 定期抽样检查数据质量
- 建立用户反馈机制

## 使用建议

### 预训练阶段
1. **基础语料**: 中文维基百科 + Leipzig语料库
2. **扩展语料**: CT-LLM数据集 + 大规模网络爬取数据
3. **领域语料**: 电商对话数据 + 商品评论数据

### 指令微调阶段
1. **通用指令**: BELLE + COIG + pCLUE
2. **对话指令**: LCCC + KdConv + CPED
3. **电商指令**: 自定义电商场景指令数据

### 强化学习阶段
1. **人类偏好数据**: 自建电商场景偏好数据集
2. **质量评估数据**: 多维度质量评估样本
3. **安全性数据**: 内容安全和对齐数据

## 注意事项

1. **版权问题**: 使用数据集前请仔细阅读各数据集的许可协议
2. **数据质量**: 不同数据集质量差异较大，建议进行质量评估
3. **数据平衡**: 注意各类数据的比例，避免过拟合
4. **隐私保护**: 处理用户数据时注意隐私保护
5. **持续更新**: 定期关注数据集的最新版本和更新

## 数据集统计

| 数据集 | 类型 | 规模 | 主要用途 |
|--------|------|------|----------|
| 中文维基百科 | 通用文本 | 2.62GB | 预训练 |
| CT-LLM MAP-CC | 通用文本 | 800B tokens | 预训练 |
| LCCC | 对话 | 6.7M utterances | 对话训练 |
| BELLE | 指令 | 1.5M样本 | 指令微调 |
| COIG | 指令 | 190K样本 | 指令微调 |
| 电商对话 | 电商对话 | 1M对话对 | 电商训练 |
| 天池知识图谱 | 电商知识 | 260M样本 | 知识增强 |

通过合理组合这些数据集，可以构建出高质量的中文电商MoE语言模型。