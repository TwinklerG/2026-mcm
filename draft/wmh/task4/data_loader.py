"""
Task 4 数据加载模块。

加载 Task 1 的粉丝投票估计和原始比赛数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
TASK1_OUTPUT = SCRIPT_DIR.parent / "task1_bayesian_mcmc" / "outputs"


@dataclass(frozen=True)
class ContestantWeekData:
    """选手单周数据。"""

    season: int
    week: int
    contestant_id: str
    judge_score: float
    fan_share: float
    is_eliminated: bool
    placement: int | None  # 仅决赛周有


@dataclass(frozen=True)
class SeasonData:
    """单赛季数据。"""

    season: int
    weeks: list[int]
    contestants: dict[str, list[ContestantWeekData]]  # contestant_id -> week data
    max_week: int
    winner_id: str | None


def load_raw_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """加载原始数据。

    Returns
    -------
    contestants : pl.DataFrame
        选手信息
    week_summary : pl.DataFrame
        每周总分
    fan_shares : pl.DataFrame
        Task 1 估计的粉丝份额
    """
    contestants = pl.read_csv(DATA_DIR / "contestants.csv")
    week_summary = pl.read_csv(DATA_DIR / "week_summary.csv")
    fan_shares = pl.read_csv(TASK1_OUTPUT / "fan_shares.csv")

    return contestants, week_summary, fan_shares


def build_season_data(
    contestants: pl.DataFrame,
    week_summary: pl.DataFrame,
    fan_shares: pl.DataFrame,
) -> dict[int, SeasonData]:
    """构建按赛季组织的数据。

    Parameters
    ----------
    contestants : pl.DataFrame
        选手信息
    week_summary : pl.DataFrame
        每周总分
    fan_shares : pl.DataFrame
        估计的粉丝份额

    Returns
    -------
    dict[int, SeasonData]
        赛季 -> 数据
    """
    # 合并数据
    merged = (
        week_summary.join(
            fan_shares.select(
                ["season", "week", "contestant_id", "estimated_fan_share"]
            ),
            on=["season", "week", "contestant_id"],
            how="left",
        )
        .join(
            contestants.select(["contestant_id", "placement"]),
            on="contestant_id",
            how="left",
        )
        .filter(pl.col("total_score") > 0)  # 仅保留活跃选手
        .sort(["season", "week", "contestant_id"])
    )

    # 构建赛季数据
    seasons: dict[int, SeasonData] = {}
    for season in sorted(merged["season"].unique().to_list()):
        season_df = merged.filter(pl.col("season") == season)
        weeks = sorted(season_df["week"].unique().to_list())
        max_week = max(weeks)

        # 确定淘汰状态
        contestants_data: dict[str, list[ContestantWeekData]] = {}

        for week in weeks:
            week_df = season_df.filter(pl.col("week") == week)
            next_week_ids: set[str] = set()

            if week < max_week:
                next_week_df = season_df.filter(pl.col("week") == week + 1)
                next_week_ids = set(next_week_df["contestant_id"].to_list())

            for row in week_df.iter_rows(named=True):
                cid = row["contestant_id"]
                is_finale = week == max_week
                is_eliminated = (not is_finale) and (cid not in next_week_ids)

                cwd = ContestantWeekData(
                    season=season,
                    week=week,
                    contestant_id=cid,
                    judge_score=row["total_score"],
                    fan_share=row["estimated_fan_share"] or 0.0,
                    is_eliminated=is_eliminated,
                    placement=row["placement"] if is_finale else None,
                )

                if cid not in contestants_data:
                    contestants_data[cid] = []
                contestants_data[cid].append(cwd)

        # 找出冠军
        winner_id = None
        winner_df = contestants.filter(
            (pl.col("season") == season) & (pl.col("placement") == 1)
        )
        if winner_df.height > 0:
            winner_id = winner_df["contestant_id"].to_list()[0]

        seasons[season] = SeasonData(
            season=season,
            weeks=weeks,
            contestants=contestants_data,
            max_week=max_week,
            winner_id=winner_id,
        )

    return seasons


def get_week_data(
    season_data: SeasonData,
    week: int,
) -> list[ContestantWeekData]:
    """获取指定周次的所有选手数据。

    Parameters
    ----------
    season_data : SeasonData
        赛季数据
    week : int
        周次

    Returns
    -------
    list[ContestantWeekData]
        该周所有选手数据
    """
    result = []
    for cid, weeks_data in season_data.contestants.items():
        for wd in weeks_data:
            if wd.week == week:
                result.append(wd)
                break
    return result


def get_controversy_cases() -> list[dict]:
    """获取已知的争议案例。

    Returns
    -------
    list[dict]
        争议案例列表
    """
    return [
        {
            "name": "Jerry Rice",
            "season": 2,
            "contestant_id": "Jerry Rice_S2",
            "issue": "技术分 5 周垫底仍进决赛（第 2 名）",
            "expected_result": "不应进入决赛",
            "actual_placement": 2,
        },
        {
            "name": "Billy Ray Cyrus",
            "season": 4,
            "contestant_id": "Billy Ray Cyrus_S4",
            "issue": "技术分 6 周垫底获第 5",
            "expected_result": "应更早淘汰",
            "actual_placement": 5,
        },
        {
            "name": "Bristol Palin",
            "season": 11,
            "contestant_id": "Bristol Palin_S11",
            "issue": "技术分 12 次最低仍获第 3",
            "expected_result": "应更早淘汰",
            "actual_placement": 3,
        },
        {
            "name": "Bobby Bones",
            "season": 27,
            "contestant_id": "Bobby Bones_S27",
            "issue": "持续低分夺冠",
            "expected_result": "不应夺冠",
            "actual_placement": 1,
        },
    ]


def compute_cumulative_judge_rank(
    season_data: SeasonData,
    up_to_week: int,
) -> dict[str, float]:
    """计算累积技术分排名。

    Parameters
    ----------
    season_data : SeasonData
        赛季数据
    up_to_week : int
        截止周次

    Returns
    -------
    dict[str, float]
        选手 ID -> 累积技术分
    """
    cumulative: dict[str, float] = {}
    for cid, weeks_data in season_data.contestants.items():
        total = 0.0
        count = 0
        for wd in weeks_data:
            if wd.week <= up_to_week:
                total += wd.judge_score
                count += 1
        if count > 0:
            cumulative[cid] = total / count  # 使用平均分
    return cumulative


def get_actual_elimination_order(season_data: SeasonData) -> list[str]:
    """获取实际淘汰顺序。

    Parameters
    ----------
    season_data : SeasonData
        赛季数据

    Returns
    -------
    list[str]
        按淘汰顺序排列的选手 ID（先淘汰的在前）
    """
    eliminations: list[tuple[int, str]] = []

    for cid, weeks_data in season_data.contestants.items():
        for wd in weeks_data:
            if wd.is_eliminated:
                eliminations.append((wd.week, cid))
                break
        else:
            # 未被淘汰 = 决赛选手
            # 使用 placement 作为淘汰顺序（placement 越大越早淘汰）
            final_week_data = [w for w in weeks_data if w.week == season_data.max_week]
            if final_week_data and final_week_data[0].placement:
                # 决赛选手按名次逆序排列（第 3 名比第 1 名先"淘汰"）
                eliminations.append(
                    (season_data.max_week + final_week_data[0].placement, cid)
                )

    eliminations.sort(key=lambda x: x[0])
    return [cid for _, cid in eliminations]
