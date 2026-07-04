"""
消息面分析模块 — FLOW AI Trader 三维框架第一层
功能：输入新闻文本 → 关键词匹配 + TF-IDF语义匹配 → 输出11板块评分

核心设计：
- 双路融合：关键词匹配(0.4权重) + TF-IDF语义(0.6权重)
- 关键词来自 event_sector_mapping.json（17组预设映射）
- TF-IDF 基于 company_info.csv 中503只股票的 BusinessSummary
- 输出：每个板块的消息面评分 -2~+2
"""

import json
import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ============================================================
# 负面/利好方向词表
# ============================================================
NEGATIVE_WORDS = [
    'risk', 'concern', 'fear', 'decline', 'drop', 'fall', 'plunge', 'crash',
    'sell-off', 'selloff', 'bear', 'recession', 'crisis', 'sanction', 'tariff',
    'trade war', 'ban', 'restriction', 'regulation', 'crackdown', 'probe',
    'investigation', 'lawsuit', 'fine', 'penalty', 'layoff', 'cut', 'warn',
    'downgrade', 'deficit', 'bubble', 'overvalued', 'uncertainty',
    '下跌', '暴跌', '崩盘', '危机', '衰退', '制裁', '管制', '调查', '罚款',
    '处罚', '裁员', '下调', '警告', '利空', '恐慌', '泡沫', '不确定',
    '贸易战', '加税', '禁令', '限令', '收紧',
]

POSITIVE_WORDS = [
    'growth', 'surge', 'rally', 'bull', 'record', 'breakthrough', 'innovation',
    'approval', 'launch', 'partnership', 'deal', 'acquisition', 'expansion',
    'beat', 'upgrade', 'optimistic', 'recovery', 'stimulus', 'easing',
    '上涨', '增长', '突破', '创新', '批准', '合作', '协议', '利好',
    '超预期', '上调', '复苏', '刺激', '降息', '宽松',
]

# ============================================================
# 板块定义（来自 event_sector_mapping.json + 11 SPDR）
# ============================================================
ALL_SECTORS = [
    'Technology', 'Healthcare', 'Financials', 'Energy', 'Industrials',
    'Consumer Discretionary', 'Consumer Staples', 'Utilities', 'Materials',
    'Real Estate', 'Communication Services',
]

SECTOR_TO_ETF = {
    'Technology': 'XLK', 'Healthcare': 'XLV', 'Financials': 'XLF',
    'Energy': 'XLE', 'Industrials': 'XLI', 'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP', 'Utilities': 'XLU', 'Materials': 'XLB',
    'Real Estate': 'XLRE', 'Communication Services': 'XLC',
}

# ============================================================
# 数据路径
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MAPPING_FILE = os.path.join(DATA_DIR, 'event_sector_mapping.json')
COMPANY_INFO_FILE = os.path.join(DATA_DIR, 'company_info.csv')
TFIDF_MODEL_FILE = os.path.join(DATA_DIR, 'tfidf_model.pkl')


class NewsAnalyzer:
    """消息面分析器 — 双路融合"""

    def __init__(self):
        self.mapping = None
        self.vectorizer = None
        self.sector_vectors = None      # {sector: avg_tfidf_vector}
        self.company_sectors = None      # {symbol: sector}
        self.keyword_cache = None        # 预编译的正则模式

        self._load_mapping()
        self._load_or_train_tfidf()

    # ---- 关键词匹配 ----

    def _load_mapping(self):
        """加载事件关键词映射"""
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)

        # 预编译正则：每个事件组的关键词
        self.keyword_cache = []
        for item in self.mapping['event_keywords']:
            if item.get('sector') is None:
                continue  # 跳过无板块映射的（如通用财报）
            patterns = []
            for kw in item['keywords']:
                # 转义特殊字符，支持中英文
                escaped = re.escape(kw)
                patterns.append(escaped)
            if patterns:
                compiled = re.compile('|'.join(patterns), re.IGNORECASE)
                self.keyword_cache.append({
                    'regex': compiled,
                    'sector': item['sector'],
                    'etf': item['etf'],
                    'direction': item['direction'],
                    'weight': item['weight'],
                    'note': item['note'],
                    'keywords': item['keywords'],
                })

    def _detect_direction(self, text, item_direction):
        """根据新闻文本中的正/负面词判断实际方向"""
        text_lower = text.lower()

        neg_count = sum(1 for w in NEGATIVE_WORDS if w.lower() in text_lower)
        pos_count = sum(1 for w in POSITIVE_WORDS if w.lower() in text_lower)

        if item_direction == 'mixed':
            if neg_count > pos_count:
                return -1, 'negative'
            elif pos_count > neg_count:
                return 1, 'positive'
            else:
                return 0, 'neutral'

        if item_direction == 'negative':
            return -1, 'negative'
        if item_direction == 'positive':
            return 1, 'positive'
        if item_direction == 'neutral':
            return 0, 'neutral'

        # 如果原方向不确定，看文本中的倾向
        if neg_count > pos_count:
            return -1, 'negative'
        elif pos_count > neg_count:
            return 1, 'positive'
        else:
            return 0, 'neutral'

    def keyword_match(self, news_text):
        """
        关键词匹配：在新闻中搜索17组关键词，输出板块评分

        Returns:
            {
                sector_score: {sector: float},   # 关键词命中加权分(未归一化)
                hits: [{sector, etf, direction, matched_keywords, note, weight}],
                raw_scores: {sector: float},
            }
        """
        sector_hits = defaultdict(lambda: {'weighted': 0, 'count': 0, 'keywords': [], 'directions': [], 'notes': []})
        all_hits = []

        for item in self.keyword_cache:
            regex = item['regex']
            matches = regex.findall(news_text)
            if not matches:
                continue

            # 去重匹配到的关键词
            matched_kws = list(set(match for match in matches if match))
            if not matched_kws:
                continue

            sector = item['sector']
            weight = item['weight']
            direction_val, direction_label = self._detect_direction(news_text, item['direction'])

            impact = weight * direction_val

            sector_hits[sector]['weighted'] += impact
            sector_hits[sector]['count'] += 1
            sector_hits[sector]['keywords'].extend(matched_kws)
            sector_hits[sector]['directions'].append(direction_label)
            sector_hits[sector]['notes'].append(item['note'])

            all_hits.append({
                'sector': sector,
                'etf': item['etf'],
                'direction': direction_label,
                'impact': impact,
                'matched_keywords': matched_kws,
                'weight': weight,
                'note': item['note'],
            })

        # 转为评分（限制在 ±2 范围内）
        raw_scores = {}
        sector_score = {}
        for sector in ALL_SECTORS:
            info = sector_hits.get(sector)
            if info and info['count'] > 0:
                # 加权分归一化：除以命中组数，限制范围
                raw = info['weighted'] / max(info['count'], 1)
                raw_scores[sector] = round(raw, 2)
                sector_score[sector] = round(max(-2.0, min(2.0, raw)), 1)
            else:
                raw_scores[sector] = 0.0
                sector_score[sector] = 0.0

        return {
            'sector_score': sector_score,
            'raw_scores': raw_scores,
            'hits': all_hits,
            'hit_count': len(all_hits),
        }

    # ---- TF-IDF 语义匹配 ----

    def _load_or_train_tfidf(self):
        """加载已有TF-IDF模型，或训练新模型"""
        if os.path.exists(TFIDF_MODEL_FILE):
            print("[news_analysis] 加载已有 TF-IDF 模型...")
            with open(TFIDF_MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.sector_vectors = data['sector_vectors']
            self.company_sectors = data['company_sectors']
            return

        print("[news_analysis] 训练 TF-IDF 模型（首次运行约需30秒）...")
        self._train_tfidf()

    def _train_tfidf(self):
        """在 company_info.csv 上训练 TF-IDF，构建板块向量"""
        df = pd.read_csv(COMPANY_INFO_FILE)
        df = df[df['BusinessSummary'].notna() & (df['BusinessSummary'].str.len() > 50)]
        df['BusinessSummary'] = df['BusinessSummary'].fillna('')

        # TF-IDF 向量化
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2,
        )
        tfidf_matrix = self.vectorizer.fit_transform(df['BusinessSummary'])

        # 记录公司→板块映射
        self.company_sectors = dict(zip(df['Symbol'], df['Sector']))

        # 按板块聚合：每个板块取该板块内所有公司的TF-IDF平均向量
        self.sector_vectors = {}
        for sector in ALL_SECTORS:
            sector_mask = (df['Sector'] == sector).values  # numpy array
            if sector_mask.sum() > 0:
                sector_vec = tfidf_matrix[sector_mask].mean(axis=0)
                self.sector_vectors[sector] = np.asarray(sector_vec).flatten()
            else:
                self.sector_vectors[sector] = np.zeros(self.vectorizer.max_features)

        # 保存模型
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(TFIDF_MODEL_FILE, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'sector_vectors': self.sector_vectors,
                'company_sectors': self.company_sectors,
            }, f)

        print(f"[news_analysis] TF-IDF模型已保存 ({self.vectorizer.max_features}维)")

    def _get_top_companies(self, news_vec, sector, n=5):
        """获取某个板块内与新闻最相关的公司"""
        df = pd.read_csv(COMPANY_INFO_FILE)
        df = df[df['BusinessSummary'].notna() & (df['BusinessSummary'].str.len() > 50)]
        df = df[df['Sector'] == sector]

        if df.empty:
            return []

        sector_matrix = self.vectorizer.transform(df['BusinessSummary'])
        sims = cosine_similarity(news_vec, sector_matrix).flatten()

        top_idx = np.argsort(sims)[-n:][::-1]
        results = []
        for idx in top_idx:
            results.append({
                'symbol': df.iloc[idx]['Symbol'],
                'company': df.iloc[idx]['Company'],
                'similarity': round(float(sims[idx]), 4),
            })
        return results

    def tfidf_match(self, news_text):
        """
        TF-IDF语义匹配：将新闻文本向量化，与每个板块的平均向量计算余弦相似度

        Returns:
            {
                sector_similarity: {sector: similarity},
                sector_score: {sector: float},    # 基于排名的评分 -2~+2
                top_companies: {sector: [{symbol, company, similarity}]},
            }
        """
        if self.vectorizer is None:
            self._load_or_train_tfidf()

        # 新闻向量化
        news_vec = self.vectorizer.transform([news_text])

        # 与各板块计算余弦相似度
        raw_sims = {}
        for sector in ALL_SECTORS:
            if sector in self.sector_vectors:
                sim = cosine_similarity(news_vec, self.sector_vectors[sector].reshape(1, -1))[0][0]
                raw_sims[sector] = float(sim)
            else:
                raw_sims[sector] = 0.0

        sector_similarity = {s: round(v, 4) for s, v in raw_sims.items()}

        # 基于排名的评分（比绝对值更鲁棒）
        # 11个板块按相似度排名：Top3=+1, Mid4=0, Bot4=-1 → 再×2 映射到 ±2
        ranked = sorted(raw_sims.items(), key=lambda x: x[1], reverse=True)
        rank_score = {}
        for i, (sector, sim) in enumerate(ranked):
            if i < 3:
                rank_score[sector] = 2.0  # 排名前3 = +2
            elif i < 7:
                rank_score[sector] = 0.0  # 排名中游 = 0
            else:
                rank_score[sector] = -2.0  # 排名后4 = -2

        sector_score = {s: round(rank_score.get(s, 0.0), 1) for s in ALL_SECTORS}

        # 找每个板块内最相关的TOP-3公司
        top_companies = {}
        for sector, sim in ranked[:5]:
            if sim > 0.1:
                top_companies[sector] = self._get_top_companies(news_vec, sector, n=3)

        return {
            'sector_similarity': sector_similarity,
            'sector_score': sector_score,
            'top_companies': top_companies,
        }

    # ---- 双路融合 ----

    def analyze_news(self, news_text):
        """
        消息面分析主入口：关键词 + TF-IDF 双路融合

        Args:
            news_text: 新闻/事件文本

        Returns:
            {
                sector_scores: {sector: score},     # -2~+2 综合评分
                direction: 'bullish'|'bearish'|'neutral',
                summary: str,                        # 一句话总结
                keyword_result: {...},               # 关键词匹配详情
                tfidf_result: {...},                 # TF-IDF语义详情
                top_sectors: [(sector, score), ...], # 排序后的板块评分
            }
        """
        # 路径1：关键词匹配
        kw_result = self.keyword_match(news_text)

        # 路径2：TF-IDF语义
        tf_result = self.tfidf_match(news_text)

        # 融合：关键词命中时信任关键词，TF-IDF补充
        has_kw = kw_result['hit_count'] > 0
        sector_scores = {}
        for sector in ALL_SECTORS:
            kw_score = kw_result['sector_score'].get(sector, 0.0)
            tf_score = tf_result['sector_score'].get(sector, 0.0)

            if has_kw and abs(kw_score) >= 1.0:
                # 关键词强信号(±1+)，TF-IDF只做微调(±0.4)
                final = round(kw_score + tf_score * 0.2, 1)
            elif has_kw and abs(kw_score) >= 0.5:
                # 关键词弱信号，关键词6:TF-IDF4
                final = round(kw_score * 0.6 + tf_score * 0.4, 1)
            elif has_kw:
                # 关键词命中但方向不明确
                final = round(kw_score * 0.5 + tf_score * 0.5, 1)
            else:
                # 无关键词：纯TF-IDF，打折防噪声
                final = round(tf_score * 0.5, 1)

            sector_scores[sector] = max(-2.0, min(2.0, final))

        # 排序
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)

        # 整体方向判定：基于最强信号的方向和强度
        max_score = max(v for _, v in sorted_sectors)
        min_score = min(v for _, v in sorted_sectors)
        if max_score > abs(min_score) and max_score >= 0.5:
            direction = 'bullish'
        elif abs(min_score) > max_score and abs(min_score) >= 0.5:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # 生成一句话总结
        strong_bull = [(s, v) for s, v in sorted_sectors if v >= 1.0]
        strong_bear = [(s, v) for s, v in sorted_sectors if v <= -1.0]

        parts = []
        if strong_bull:
            names = ', '.join(s for s, _ in strong_bull[:3])
            parts.append(f'利好板块: {names}')
        if strong_bear:
            names = ', '.join(s for s, _ in strong_bear[:3])
            parts.append(f'利空板块: {names}')

        if not parts:
            parts.append('无明显板块倾向')

        summary = '；'.join(parts)

        return {
            'sector_scores': sector_scores,
            'direction': direction,
            'summary': summary,
            'keyword_result': {
                'hits': kw_result['hits'],
                'hit_count': kw_result['hit_count'],
                'sector_score': kw_result['sector_score'],
            },
            'tfidf_result': {
                'sector_similarity': tf_result['sector_similarity'],
                'sector_score': tf_result['sector_score'],
                'top_companies': tf_result['top_companies'],
            },
            'top_sectors': sorted_sectors,
        }


# ============================================================
# 便捷函数
# ============================================================

# 全局单例
_analyzer = None


def get_analyzer():
    """获取全局NewsAnalyzer实例（懒加载）"""
    global _analyzer
    if _analyzer is None:
        _analyzer = NewsAnalyzer()
    return _analyzer


def analyze_news(news_text):
    """便捷函数：分析新闻文本"""
    return get_analyzer().analyze_news(news_text)


# ============================================================
# CLI 测试
# ============================================================
if __name__ == '__main__':
    import sys

    analyzer = NewsAnalyzer()

    # 测试新闻
    test_news = [
        "NVIDIA发布新一代AI芯片H200，性能提升90%，全球云厂商纷纷追加算力采购订单。",
        "美联储宣布降息50个基点，为四年来首次降息。房贷利率应声下跌，房地产市场活跃度提升。",
        "美国宣布对中国加征60%关税，贸易战再度升级。钢铁和铝制品受影响严重。",
        "中东局势紧张，原油价格暴涨8%，能源股集体走高。",
    ]

    if len(sys.argv) > 1:
        test_news = [' '.join(sys.argv[1:])]

    for text in test_news:
        print("=" * 70)
        print(f"📰 新闻: {text[:80]}...")
        print("-" * 70)
        result = analyzer.analyze_news(text)
        print(f"整体方向: {result['direction']}")
        print(f"总结: {result['summary']}")
        print(f"\n板块评分:")
        for sector, score in result['top_sectors']:
            bar = '+' * max(0, int(score * 5)) + '-' * max(0, int(-score * 5))
            bar = bar or '='
            etf = SECTOR_TO_ETF.get(sector, '?')
            print(f"  {sector:25s} ({etf:4s}): {score:4.1f} |{bar}")

        if result['keyword_result']['hits']:
            print(f"\n关键词命中 ({result['keyword_result']['hit_count']}组):")
            for hit in result['keyword_result']['hits']:
                print(f"  {hit['sector']}: {hit['matched_keywords'][:5]} → {hit['direction']} (影响分 {hit['impact']})")

        print()
