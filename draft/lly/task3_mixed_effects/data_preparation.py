"""
数据准备模块 - 构建 Panel Data 用于分层混合效应模型。

将 Task 1 的 fan_shares.csv 与原始数据合并，
创建长格式面板数据，包含所有协变量。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass


# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
TASK1_DIR = BASE_DIR / "task1_bayesian_mcmc" / "outputs"


def load_raw_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    加载原始数据文件。

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        contestants, scores_long, fan_shares 三个数据框
    """
    contestants = pl.read_csv(DATA_DIR / "contestants.csv")
    scores_long = pl.read_csv(DATA_DIR / "scores_long.csv")
    fan_shares = pl.read_csv(TASK1_DIR / "fan_shares.csv")
    return contestants, scores_long, fan_shares


def standardize_industry(industry: str) -> str:
    """
    标准化行业类别，合并小类别。

    Parameters
    ----------
    industry : str
        原始行业名称

    Returns
    -------
    str
        标准化后的行业类别
    """
    # 映射表：合并相似或小样本类别
    mapping = {
        "Actor/Actress": "Actor",
        "Singer/Rapper": "Singer",
        "Musician": "Singer",
        "TV Personality": "TV",
        "Social Media Personality": "TV",
        "Social media personality": "TV",
        "News Anchor": "TV",
        "Radio Personality": "TV",
        "Sports Broadcaster": "TV",
        "Journalist": "TV",
        "Athlete": "Athlete",
        "Racing Driver": "Athlete",
        "Model": "Model",
        "Beauty Pagent": "Model",
        "Comedian": "Entertainment",
        "Magician": "Entertainment",
        "Motivational Speaker": "Entertainment",
        "Fitness Instructor": "Other",
        "Entrepreneur": "Other",
        "Politician": "Other",
        "Military": "Other",
        "Astronaut": "Other",
        "Conservationist": "Other",
        "Fashion Designer": "Other",
        "Producer": "Entertainment",
        "Con artist": "Other",
    }
    return mapping.get(industry, "Other")


def compute_weekly_judge_score(scores_long: pl.DataFrame) -> pl.DataFrame:
    """
    计算每周平均评委得分。

    Parameters
    ----------
    scores_long : pl.DataFrame
        长格式评分数据

    Returns
    -------
    pl.DataFrame
        每周平均评委得分
    """
    return (
        scores_long
        .filter(pl.col("score") > 0)  # 排除淘汰后的0分
        .group_by(["contestant_id", "season", "week"])
        .agg(
            pl.col("score").mean().alias("judge_score_raw"),
            pl.col("score").count().alias("n_judges"),
        )
    )


def build_panel_data(
    contestants: pl.DataFrame,
    scores_long: pl.DataFrame,
    fan_shares: pl.DataFrame,
) -> pl.DataFrame:
    """
    构建用于混合效应模型的面板数据。

    Parameters
    ----------
    contestants : pl.DataFrame
        选手基本信息
    scores_long : pl.DataFrame
        长格式评分数据
    fan_shares : pl.DataFrame
        Task 1 估计的粉丝份额

    Returns
    -------
    pl.DataFrame
        合并后的面板数据
    """
    # 1. 计算每周平均评委得分
    weekly_scores = compute_weekly_judge_score(scores_long)

    # 2. 标准化行业类别
    contestants = contestants.with_columns(
        pl
        .col("celebrity_industry")
        .map_elements(standardize_industry, return_dtype=pl.Utf8)
        .alias("industry_std")
    )

    # 3. 合并数据
    panel = (
        fan_shares
        .join(
            weekly_scores,
            on=["contestant_id", "season", "week"],
            how="inner",
        )
        .join(
            contestants.select([
                "contestant_id",
                "celebrity_name",
                "ballroom_partner",
                "celebrity_age_during_season",
                "celebrity_industry",
                "industry_std",
                "placement",
                "is_withdrew",
            ]),
            on="contestant_id",
            how="left",
        )
        .filter(~pl.col("is_withdrew"))  # 排除退赛选手
    )

    # 4. 特征工程
    # 计算全局年龄均值用于中心化
    age_mean = panel["celebrity_age_during_season"].mean()

    panel = panel.with_columns(
        # 年龄中心化
        (pl.col("celebrity_age_during_season") - age_mean).alias("age_centered"),
        # 对数粉丝份额（防止 log(0)）
        (pl.col("estimated_fan_share") + 1e-6).log().alias("log_fan_share"),
        # 评委平均分（每周多位评委评分的均值，范围约 1-10）
        pl.col("judge_score_raw").alias("judge_score"),
    )

    # 5. 添加性别信息（基于名字推断，这里简化处理）
    # 实际应用中应该有更准确的性别数据
    # 这里我们创建一个随机变量作为占位符，后续可以手动修正
    panel = panel.with_columns(pl.lit(0).alias("gender"))  # 0 = unknown

    return panel


def prepare_model_data(panel: pl.DataFrame) -> dict:
    """
    准备用于 statsmodels 的模型数据。

    Parameters
    ----------
    panel : pl.DataFrame
        面板数据

    Returns
    -------
    dict
        包含模型所需的所有变量
    """
    # 转换为 pandas（statsmodels 需要）
    df = panel.to_pandas()

    # 创建行业哑变量
    df = df.copy()
    industries = sorted(df["industry_std"].unique())
    base_industry = "Actor"  # 基准类别

    for ind in industries:
        if ind != base_industry:
            col_name = f"industry_{ind.replace(' ', '_')}"
            df[col_name] = (df["industry_std"] == ind).astype(int)

    # 移除原始 industry_std 列以避免共线性
    df = df.drop(columns=["industry_std"])

    # 确保分组变量是分类变量
    df["pro_id"] = df["ballroom_partner"].astype("category")
    df["season_id"] = df["season"].astype("category")
    df["contestant_id_cat"] = df["contestant_id"].astype("category")

    return df


def get_summary_statistics(panel: pl.DataFrame) -> dict:
    """
    计算数据摘要统计量。

    Parameters
    ----------
    panel : pl.DataFrame
        面板数据

    Returns
    -------
    dict
        摘要统计量
    """
    return {
        "n_observations": len(panel),
        "n_contestants": panel["contestant_id"].n_unique(),
        "n_seasons": panel["season"].n_unique(),
        "n_pros": panel["ballroom_partner"].n_unique(),
        "n_industries": panel["industry_std"].n_unique(),
        "age_range": (
            panel["celebrity_age_during_season"].min(),
            panel["celebrity_age_during_season"].max(),
        ),
        "weeks_range": (panel["week"].min(), panel["week"].max()),
        "industry_distribution": panel
        .group_by("industry_std")
        .count()
        .sort("count", descending=True)
        .to_dicts(),
    }


def main() -> None:
    """主函数：加载数据并构建面板数据。"""
    print("加载原始数据...")
    contestants, scores_long, fan_shares = load_raw_data()

    print("构建面板数据...")
    panel = build_panel_data(contestants, scores_long, fan_shares)

    print("数据摘要统计:")
    stats = get_summary_statistics(panel)
    for key in sorted(stats.keys()):
        value = stats[key]
        print(f"  {key}: {value}")

    # 保存处理后的数据
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    panel.sort(["season", "week", "contestant_id"]).write_csv(
        output_dir / "panel_data.csv"
    )
    print(f"\n面板数据已保存至: {output_dir / 'panel_data.csv'}")

    return panel


if __name__ == "__main__":
    main()
