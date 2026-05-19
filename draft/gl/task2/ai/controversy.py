# %%
import polars as pl
from consts import DATA_RES

controversy_cases = [
    ("Jerry Rice", 2, "亚军但5周最低评委分"),
    ("Billy Ray Cyrus", 4, "第5名但6周最低评委分"),
    ("Bristol Palin", 11, "季军但12次最低评委分"),
    ("Bobby Bones", 27, "冠军但持续低评委分"),
]

df_criteria = pl.read_csv(DATA_RES / "criteria.csv")

df_controversy = df_criteria.filter(
    pl.col("contestant_id").is_in(
        [
            f"{contestant_name}_S{season}"
            for contestant_name, season, _ in controversy_cases
        ]
    )
)

df_controversy.write_csv(DATA_RES / "controversy_cases.csv")
