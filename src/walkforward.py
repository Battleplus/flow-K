"""
Walk-Forward 滚动验证引擎

核心思想：
  训练窗口 (N天) → 选策略 → 测试窗口 (M天) → 纯回测 → 滚动

  数据: |===TRAIN===||==TEST==|→
                     |===TRAIN===||==TEST==|→
                                  |===TRAIN===||==TEST==|

  - 训练集：计算因子 + 生成信号 + 回测所有策略 → 选Top K
  - 测试集：用选出的策略纯回测，不参与任何策略选择
  - 拼接所有测试段 → 得到样本外真实表现

比传统回测更接近真实交易：
  1. 策略选择只看过去（训练集），不看未来
  2. 因子计算用滚动窗口，天然无未来函数
  3. 测试集是完全样本外的，策略表现更可信
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from src.strategies import (
    STRATEGIES, run_backtest, BacktestResult,
    backtest_all, rank_strategies, HOLD_PROFILES,
)


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    """单个 fold 的结果"""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_days: int
    test_days: int
    selected_strategies: List[str]
    # 样本内（训练集）表现
    in_sample_return: float
    in_sample_sharpe: float
    # 样本外（测试集）表现
    out_sample_return: float
    out_sample_sharpe: float
    out_sample_max_dd: float
    out_sample_trades: int
    out_sample_win_rate: float
    out_sample_equity: pd.Series


@dataclass
class WalkForwardResult:
    """Walk-Forward 完整结果"""
    # 拼接后的样本外表现
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_hold_days: float
    profit_factor: float
    # 对比
    in_sample_total_return: float  # 训练集平均收益（用于对比过拟合）
    buy_hold_return: float        # 买入持有基准
    overfit_ratio: float          # 样本内/样本外比率，>3 说明过拟合严重
    # 详情
    n_folds: int
    folds: List[FoldResult]
    combined_equity: pd.Series
    fold_selections: List[dict]   # 每折选了哪些策略


# ═══════════════════════════════════════════════════════════════
# 窗口大小自适应
# ═══════════════════════════════════════════════════════════════

# 不同持仓周期的推荐窗口大小（交易日）
WF_WINDOWS = {
    "short":  {"train": 120, "test": 30,  "min_data": 180},   # 6mo训练, 1.5mo测试
    "medium": {"train": 252, "test": 63,  "min_data": 400},   # 1y训练, 3mo测试
    "long":   {"train": 360, "test": 90,  "min_data": 540},   # 1.5y训练, 4.5mo测试
}


def get_window_sizes(hold_profile: str, data_len: int) -> Tuple[int, int]:
    """根据持仓周期和数据长度自适应窗口大小"""
    cfg = WF_WINDOWS.get(hold_profile, WF_WINDOWS["medium"])
    train = cfg["train"]
    test = cfg["test"]

    # 数据不够时缩减窗口
    if data_len < cfg["min_data"]:
        ratio = data_len / cfg["min_data"]
        train = max(int(train * ratio), 60)
        test = max(int(test * ratio), 20)

    # 确保 train + test <= data_len 且至少有 2 个 fold
    while train + test * 2 > data_len and test > 10:
        test = max(test - 5, 10)
    while train + test > data_len and train > 60:
        train = max(train - 10, 60)

    return train, test


# ═══════════════════════════════════════════════════════════════
# Walk-Forward 核心引擎
# ═══════════════════════════════════════════════════════════════

def walk_forward(
    df: pd.DataFrame,
    hold_profile: str = "medium",
    worst_case: bool = True,
    initial_capital: float = 100000,
    top_k: int = 5,
    train_window: Optional[int] = None,
    test_window: Optional[int] = None,
) -> WalkForwardResult:
    """
    Walk-Forward 滚动验证

    参数:
      df: 已计算因子的 DataFrame (含 OHLCV + 所有因子列)
      hold_profile: 持仓周期 short/medium/long
      worst_case: 最差执行模式
      initial_capital: 初始资金
      top_k: 每折选多少策略
      train_window: 训练窗口天数 (None=自动)
      test_window: 测试窗口天数 (None=自动)

    返回:
      WalkForwardResult
    """
    n = len(df)
    hp = HOLD_PROFILES.get(hold_profile, HOLD_PROFILES["medium"])

    # 自适应窗口
    if train_window is None or test_window is None:
        tw, ew = get_window_sizes(hold_profile, n)
        train_window = train_window or tw
        test_window = test_window or ew

    # 确保窗口合理
    if train_window + test_window > n:
        train_window = int(n * 0.6)
        test_window = n - train_window

    rank_metric = "total_return" if hold_profile == "long" else "sharpe_ratio"

    # 回测参数
    bt_params = dict(
        stop_loss_pct=hp["sl"],
        take_profit_pct=hp["tp"],
        hold_days_min=hp["hold_min"],
        hold_days_max=hp["hold_max"],
        worst_case=worst_case,
        limit_pct=0.0,
    )

    # 生成所有折
    folds = []
    fold_id = 0
    current_capital = initial_capital

    # 预计算所有策略的信号序列（一次性，因为因子已经算好）
    all_signals = {}
    for sid in STRATEGIES:
        try:
            all_signals[sid] = STRATEGIES[sid].generate(df)
        except Exception:
            all_signals[sid] = pd.Series(0, index=df.index)

    # 滚动
    start = 0
    all_test_equity = []
    all_test_dates = []
    all_in_sample_returns = []
    fold_selections = []

    while start + train_window + test_window <= n:
        train_start_idx = start
        train_end_idx = start + train_window
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_window, n)

        # === 训练阶段：在训练窗口回测所有策略，选 Top K ===
        df_train = df.iloc[train_start_idx:train_end_idx].copy()
        train_results = {}
        for sid in STRATEGIES:
            sig_train = all_signals[sid].iloc[train_start_idx:train_end_idx]
            try:
                r = run_backtest(
                    df_train, sig_train,
                    initial_capital=current_capital,
                    **bt_params,
                )
                if r.total_trades >= 1:
                    train_results[sid] = r
            except Exception:
                pass

        if not train_results:
            start += test_window
            fold_id += 1
            continue

        # 排名选 Top K
        ranked = rank_strategies(train_results, metric=rank_metric)
        selected = [sid for sid, _, _ in ranked[:top_k]]

        # 记录样本内表现
        in_sample_return = np.mean([train_results[s].total_return for s in selected])
        in_sample_sharpe = np.mean([train_results[s].sharpe_ratio for s in selected])
        all_in_sample_returns.append(in_sample_return)

        # === 测试阶段：用选出的策略在测试窗口纯回测 ===
        df_test = df.iloc[test_start_idx:test_end_idx].copy()
        n_sel = len(selected)
        if n_sel == 0:
            start += test_window
            fold_id += 1
            continue

        # 并行分仓：每个策略 1/n_sel 资金
        fold_equity = pd.Series(0.0, index=df_test.index)
        fold_trades = 0
        fold_winning = 0
        fold_hold_days = []
        fold_returns = []

        for sid in selected:
            sig_test = all_signals[sid].iloc[test_start_idx:test_end_idx]
            alloc = current_capital / n_sel
            try:
                r = run_backtest(
                    df_test, sig_test,
                    initial_capital=alloc,
                    **bt_params,
                )
                fold_equity += r.equity_curve.values
                fold_trades += r.total_trades
                if r.total_trades > 0:
                    fold_winning += r.total_trades * r.win_rate
                    fold_hold_days.append(r.avg_hold_days * r.total_trades)
                    fold_returns.extend([t["return_pct"] for t in r._trades] if hasattr(r, '_trades') else [])
            except Exception:
                fold_equity += alloc  # 如果失败，至少保住本金

        # 计算该折的样本外指标
        fold_return = (fold_equity.iloc[-1] - current_capital) / current_capital
        fold_daily_ret = fold_equity.pct_change().dropna()
        fold_sharpe = (fold_daily_ret.mean() / fold_daily_ret.std() * np.sqrt(252)) if fold_daily_ret.std() > 0 else 0
        fold_peak = fold_equity.expanding().max()
        fold_dd = (fold_equity - fold_peak) / fold_peak
        fold_max_dd = fold_dd.min()
        fold_win_rate = fold_winning / max(fold_trades, 1)

        fold_result = FoldResult(
            fold_id=fold_id,
            train_start=str(df.iloc[train_start_idx]["Date"]),
            train_end=str(df.iloc[train_end_idx - 1]["Date"]),
            test_start=str(df.iloc[test_start_idx]["Date"]),
            test_end=str(df.iloc[test_end_idx - 1]["Date"]),
            train_days=train_window,
            test_days=test_end_idx - test_start_idx,
            selected_strategies=selected,
            in_sample_return=in_sample_return,
            in_sample_sharpe=in_sample_sharpe,
            out_sample_return=fold_return,
            out_sample_sharpe=fold_sharpe,
            out_sample_max_dd=fold_max_dd,
            out_sample_trades=fold_trades,
            out_sample_win_rate=fold_win_rate,
            out_sample_equity=fold_equity,
        )
        folds.append(fold_result)

        # 记录选择历史
        fold_selections.append({
            "fold": fold_id,
            "test_period": f"{str(df.iloc[test_start_idx]['Date'])[:10]} ~ {str(df.iloc[test_end_idx-1]['Date'])[:10]}",
            "selected": [STRATEGIES[s].name for s in selected],
            "in_sample_ret": round(in_sample_return * 100, 2),
            "out_sample_ret": round(fold_return * 100, 2),
        })

        # 更新资金（复利）
        current_capital = fold_equity.iloc[-1]

        # 拼接权益曲线
        all_test_equity.extend(fold_equity.values)
        all_test_dates.extend(fold_equity.index)

        # 滚动
        start += test_window
        fold_id += 1

    # === 汇总所有测试段 ===
    if not folds:
        return WalkForwardResult(
            total_return=0, annual_return=0, sharpe_ratio=0,
            max_drawdown=0, win_rate=0, total_trades=0,
            avg_hold_days=0, profit_factor=0,
            in_sample_total_return=0, buy_hold_return=0,
            overfit_ratio=0, n_folds=0, folds=[],
            combined_equity=pd.Series(),
            fold_selections=[],
        )

    combined_equity = pd.Series(all_test_equity, index=all_test_dates)

    # 拼接后的整体表现
    total_return = (combined_equity.iloc[-1] - initial_capital) / initial_capital
    n_test_days = len(combined_equity)
    n_years = n_test_days / 252
    annual_return = ((1 + total_return) ** (1 / max(n_years, 0.01))) - 1

    daily_ret = combined_equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    peak = combined_equity.expanding().max()
    dd = (combined_equity - peak) / peak
    max_dd = dd.min()

    total_trades = sum(f.out_sample_trades for f in folds)
    win_rate = sum(f.out_sample_win_rate * f.out_sample_trades for f in folds) / max(total_trades, 1)

    # 买入持有基准（同期）
    test_start_idx_global = train_window
    test_end_idx_global = min(train_window + test_window * len(folds), n)
    bh_start = df["Close"].iloc[test_start_idx_global]
    bh_end = df["Close"].iloc[min(test_end_idx_global - 1, n - 1)]
    bh_return = (bh_end - bh_start) / bh_start

    # 过拟合比率：样本内收益 / 样本外收益
    avg_in_sample = np.mean(all_in_sample_returns)
    overfit_ratio = abs(avg_in_sample / total_return) if total_return != 0 else 0

    return WalkForwardResult(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_trades=total_trades,
        avg_hold_days=np.mean([f.out_sample_trades for f in folds]),
        profit_factor=1.0,
        in_sample_total_return=avg_in_sample,
        buy_hold_return=bh_return,
        overfit_ratio=overfit_ratio,
        n_folds=len(folds),
        folds=folds,
        combined_equity=combined_equity,
        fold_selections=fold_selections,
    )


# ═══════════════════════════════════════════════════════════════
# 结果转字典 (用于 API 返回)
# ═══════════════════════════════════════════════════════════════

def result_to_dict(r: WalkForwardResult) -> dict:
    """Walk-Forward 结果转 JSON 可序列化字典"""
    return {
        "total_return": round(r.total_return * 100, 2),
        "annual_return": round(r.annual_return * 100, 2),
        "sharpe_ratio": round(r.sharpe_ratio, 2),
        "max_drawdown": round(r.max_drawdown * 100, 2),
        "win_rate": round(r.win_rate * 100, 1),
        "total_trades": r.total_trades,
        "n_folds": r.n_folds,
        "in_sample_return": round(r.in_sample_total_return * 100, 2),
        "buy_hold_return": round(r.buy_hold_return * 100, 2),
        "overfit_ratio": round(r.overfit_ratio, 2),
        "folds": [
            {
                "fold": f.fold_id,
                "train_period": f"{f.train_start[:10]} ~ {f.train_end[:10]}",
                "test_period": f"{f.test_start[:10]} ~ {f.test_end[:10]}",
                "selected": [STRATEGIES[s].name for s in f.selected_strategies],
                "in_sample_ret": round(f.in_sample_return * 100, 2),
                "in_sample_sharpe": round(f.in_sample_sharpe, 2),
                "out_sample_ret": round(f.out_sample_return * 100, 2),
                "out_sample_sharpe": round(f.out_sample_sharpe, 2),
                "out_sample_max_dd": round(f.out_sample_max_dd * 100, 2),
                "out_sample_trades": f.out_sample_trades,
            }
            for f in r.folds
        ],
        "equity_curve": [
            {"date": str(idx)[:10], "value": round(val, 2)}
            for idx, val in zip(r.combined_equity.index, r.combined_equity.values)
        ] if len(r.combined_equity) > 0 else [],
    }


# ═══════════════════════════════════════════════════════════════
# 对比：传统回测 vs Walk-Forward
# ═══════════════════════════════════════════════════════════════

def compare_in_sample_vs_oos(
    df: pd.DataFrame,
    hold_profile: str = "medium",
    worst_case: bool = True,
    initial_capital: float = 100000,
    top_k: int = 5,
) -> dict:
    """
    对比传统（样本内）回测 vs Walk-Forward（样本外）回测
    返回两者的关键指标对比
    """
    from src.strategies import backtest_all, rank_strategies

    hp = HOLD_PROFILES.get(hold_profile, HOLD_PROFILES["medium"])
    rank_metric = "total_return" if hold_profile == "long" else "sharpe_ratio"

    # 传统回测（全量数据，样本内）
    results = backtest_all(
        df, min_trades=1, worst_case=worst_case,
        hold_profile=hold_profile,
    )
    ranked = rank_strategies(results, metric=rank_metric)
    top_sids = [sid for sid, _, _ in ranked[:top_k]]

    # 并行分仓组合（样本内）
    bt_params = dict(
        stop_loss_pct=hp["sl"], take_profit_pct=hp["tp"],
        hold_days_min=hp["hold_min"], hold_days_max=hp["hold_max"],
        worst_case=worst_case, limit_pct=0.0,
    )
    in_sample_equity = pd.Series(0.0, index=df.index)
    for sid in top_sids:
        sig = STRATEGIES[sid].generate(df)
        r = run_backtest(df, sig, initial_capital=initial_capital / top_k, **bt_params)
        in_sample_equity += r.equity_curve.values

    in_sample_return = (in_sample_equity.iloc[-1] - initial_capital) / initial_capital
    in_sample_daily = in_sample_equity.pct_change().dropna()
    in_sample_sharpe = (in_sample_daily.mean() / in_sample_daily.std() * np.sqrt(252)) if in_sample_daily.std() > 0 else 0
    in_sample_peak = in_sample_equity.expanding().max()
    in_sample_dd = (in_sample_equity - in_sample_peak) / in_sample_peak
    in_sample_max_dd = in_sample_dd.min()

    # Walk-Forward（样本外）
    wf = walk_forward(
        df, hold_profile=hold_profile,
        worst_case=worst_case,
        initial_capital=initial_capital,
        top_k=top_k,
    )

    # 买入持有
    bh_return = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]

    return {
        "in_sample": {
            "total_return": round(in_sample_return * 100, 2),
            "sharpe_ratio": round(in_sample_sharpe, 2),
            "max_drawdown": round(in_sample_max_dd * 100, 2),
            "selected_strategies": [STRATEGIES[s].name for s in top_sids],
            "label": "传统回测 (全量样本内)",
        },
        "out_of_sample": {
            "total_return": round(wf.total_return * 100, 2),
            "sharpe_ratio": round(wf.sharpe_ratio, 2),
            "max_drawdown": round(wf.max_drawdown * 100, 2),
            "n_folds": wf.n_folds,
            "overfit_ratio": round(wf.overfit_ratio, 2),
            "label": f"Walk-Forward ({wf.n_folds}折样本外)",
        },
        "buy_hold": {
            "total_return": round(bh_return * 100, 2),
        },
        "overfit_gap": round((in_sample_return - wf.total_return) * 100, 2),
        "overfit_ratio": round(wf.overfit_ratio, 2),
        "fold_details": wf.fold_selections,
    }
