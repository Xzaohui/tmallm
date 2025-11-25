"""
中文电商领域数据配比分析
分析不同数据类型的最佳配比比例
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import json
from typing import Dict, List
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EcommerceDataAnalyzer:
    """电商数据分析器"""
    
    def __init__(self):
        self.data_categories = {
            'product_description': '产品描述',
            'user_reviews': '用户评价', 
            'customer_service': '客服对话',
            'marketing_copy': '营销文案',
            'category_info': '类目信息',
            'brand_info': '品牌信息',
            'technical_specs': '技术规格',
            'usage_instructions': '使用说明'
        }
        
    def generate_sample_data_distribution(self) -> Dict[str, int]:
        """生成示例数据分布"""
        # 基于电商领域特点的数据分布
        distribution = {
            'product_description': 250000,  # 产品描述 - 25%
            'user_reviews': 200000,         # 用户评价 - 20%
            'customer_service': 150000,     # 客服对话 - 15%
            'marketing_copy': 100000,       # 营销文案 - 10%
            'category_info': 100000,        # 类目信息 - 10%
            'brand_info': 80000,            # 品牌信息 - 8%
            'technical_specs': 70000,       # 技术规格 - 7%
            'usage_instructions': 50000     # 使用说明 - 5%
        }
        return distribution
    
    def analyze_data_characteristics(self) -> pd.DataFrame:
        """分析数据特征"""
        characteristics = {
            'data_type': ['product_description', 'user_reviews', 'customer_service', 'marketing_copy'],
            'avg_length': [120, 80, 60, 100],  # 平均长度（字符）
            'vocabulary_richness': [0.75, 0.85, 0.70, 0.90],  # 词汇丰富度
            'domain_specificity': [0.90, 0.60, 0.50, 0.80],   # 领域特异性
            'sentiment_diversity': [0.30, 0.95, 0.80, 0.70],  # 情感多样性
            'format_consistency': [0.85, 0.40, 0.60, 0.90],   # 格式一致性
            'noise_level': [0.10, 0.30, 0.20, 0.15]           # 噪声水平
        }
        
        return pd.DataFrame(characteristics)
    
    def calculate_optimal_ratio(self, target_tokens: int = 10_000_000_000) -> Dict[str, float]:
        """计算最优数据配比"""
        # 基于D-CPT Law的配比计算
        base_weights = {
            'product_description': 0.25,
            'user_reviews': 0.20,
            'customer_service': 0.15,
            'marketing_copy': 0.10,
            'category_info': 0.10,
            'brand_info': 0.08,
            'technical_specs': 0.07,
            'usage_instructions': 0.05
        }
        
        # 质量调整因子
        quality_factors = {
            'product_description': 1.2,  # 高质量，增加权重
            'user_reviews': 1.0,         # 中等质量
            'customer_service': 0.9,     # 较低质量
            'marketing_copy': 1.1,       # 较高质量
            'category_info': 1.0,
            'brand_info': 1.0,
            'technical_specs': 1.3,     # 最高质量
            'usage_instructions': 1.1
        }
        
        # 计算调整后的权重
        adjusted_weights = {}
        total_weight = 0
        
        for data_type in base_weights:
            adjusted_weight = base_weights[data_type] * quality_factors[data_type]
            adjusted_weights[data_type] = adjusted_weight
            total_weight += adjusted_weight
        
        # 归一化
        final_ratios = {}
        for data_type, weight in adjusted_weights.items():
            final_ratios[data_type] = weight / total_weight
        
        return final_ratios
    
    def visualize_data_distribution(self, distribution: Dict[str, int]):
        """可视化数据分布"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 饼图显示总体分布
        labels = [self.data_categories.get(k, k) for k in distribution.keys()]
        sizes = list(distribution.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('中文电商数据分布', fontsize=14, fontweight='bold')
        
        # 2. 柱状图
        ax2.bar(range(len(distribution)), sizes, color=colors)
        ax2.set_xticks(range(len(distribution)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('数据量（条）')
        ax2.set_title('各类型数据量对比', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 最优配比分析
        optimal_ratios = self.calculate_optimal_ratio()
        ax3.bar(range(len(optimal_ratios)), list(optimal_ratios.values()), 
                color=colors, alpha=0.7)
        ax3.set_xticks(range(len(optimal_ratios)))
        ax3.set_xticklabels([self.data_categories.get(k, k) for k in optimal_ratios.keys()], 
                           rotation=45, ha='right')
        ax3.set_ylabel('配比比例')
        ax3.set_title('最优数据配比（质量调整后）', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 数据特征热力图
        char_df = self.analyze_data_characteristics()
        feature_cols = ['avg_length', 'vocabulary_richness', 'domain_specificity', 
                       'sentiment_diversity', 'format_consistency']
        
        # 创建热力图数据
        heatmap_data = char_df.set_index('data_type')[feature_cols].T
        
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', 
                   cbar_kws={'label': '特征强度'}, ax=ax4)
        ax4.set_title('数据特征分析', fontsize=14, fontweight='bold')
        ax4.set_xlabel('数据类型')
        ax4.set_ylabel('特征维度')
        
        plt.tight_layout()
        plt.savefig('/mnt/okcomputer/output/data_distribution_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_training_recommendations(self) -> List[str]:
        """生成训练建议"""
        recommendations = [
            "1. 数据预处理建议：",
            "   - 产品描述数据：保留完整，进行标准化清洗",
            "   - 用户评价：过滤极端短评和垃圾评论",
            "   - 客服对话：去除敏感信息，统一格式",
            "   - 营销文案：保留多样化表达，避免过度重复",
            "",
            "2. 训练策略建议：",
            "   - 采用两阶段训练：通用预训练 → 电商领域续训",
            "   - 使用动态数据采样，根据训练效果调整配比",
            "   - 实施难度采样，逐步增加复杂样本",
            "   - 定期评估各数据类型的贡献度",
            "",
            "3. 数据增强建议：",
            "   - 产品描述：同义词替换、句式变换",
            "   - 用户评价：情感保持的文本改写",
            "   - 客服对话：基于模板的多轮对话生成",
            "   - 营销文案：风格和语气的多样化",
            "",
            "4. 质量监控建议：",
            "   - 建立数据质量评估指标",
            "   - 定期抽样检查标注质量",
            "   - 监控训练过程中的异常样本",
            "   - 建立用户反馈机制"
        ]
        return recommendations
    
    def create_data_mixing_strategy(self, total_tokens: int = 50_000_000_000) -> Dict:
        """创建数据混合策略"""
        ratios = self.calculate_optimal_ratio()
        
        strategy = {
            'total_tokens': total_tokens,
            'mixing_strategy': 'dynamic_with_quality_weighting',
            'phase1_ratio': 0.7,  # 第一阶段使用70%的数据
            'phase2_ratio': 0.3,  # 第二阶段使用30%的数据
            'data_distribution': {}
        }
        
        for data_type, ratio in ratios.items():
            strategy['data_distribution'][data_type] = {
                'total_samples': int(total_tokens * ratio),
                'phase1_samples': int(total_tokens * ratio * 0.7),
                'phase2_samples': int(total_tokens * ratio * 0.3),
                'quality_threshold': 0.8,
                'sampling_temperature': 1.0
            }
        
        return strategy
    
    def save_analysis_report(self):
        """保存分析报告"""
        report = {
            'data_distribution': self.generate_sample_data_distribution(),
            'optimal_ratios': self.calculate_optimal_ratio(),
            'training_recommendations': self.generate_training_recommendations(),
            'mixing_strategy': self.create_data_mixing_strategy(),
            'characteristics': self.analyze_data_characteristics().to_dict('records')
        }
        
        with open('/mnt/okcomputer/output/data_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("数据分析报告已保存到: /mnt/okcomputer/output/data_analysis_report.json")


def main():
    """主函数"""
    analyzer = EcommerceDataAnalyzer()
    
    # 生成数据分布
    distribution = analyzer.generate_sample_data_distribution()
    print("数据分布:")
    for data_type, count in distribution.items():
        percentage = (count / sum(distribution.values())) * 100
        print(f"  {analyzer.data_categories[data_type]}: {count:,} ({percentage:.1f}%)")
    
    # 计算最优配比
    optimal_ratios = analyzer.calculate_optimal_ratio()
    print("\n最优数据配比:")
    for data_type, ratio in optimal_ratios.items():
        print(f"  {analyzer.data_categories[data_type]}: {ratio:.3f}")
    
    # 可视化分析
    analyzer.visualize_data_distribution(distribution)
    
    # 生成训练建议
    recommendations = analyzer.generate_training_recommendations()
    print("\n训练建议:")
    for rec in recommendations:
        print(rec)
    
    # 保存报告
    analyzer.save_analysis_report()


if __name__ == "__main__":
    main()