import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_matches_index_from_year_data(data_dir: Path):
    data = []
    for year in range(2001, 2026):
        readme_path = data_dir / f"{year}_male_json/README.txt"
        # data = pd.read_csv(readme_path, delimiter=" - ")
        if readme_path.exists():
            with readme_path.open() as file:
                content = file.readlines()
                data.extend(
                    line.strip().split(" - ")
                    for line in content
                    if line.startswith(f"{year}-")
                )
        else:
            print(f"README.txt not found for year {year}")
    df = pd.DataFrame(
        data, columns=["date", "level", "format", "gender", "id", "teams"]
    )
    df = df.loc[lambda x: x["level"] == "international"]
    df[["team1", "team2"]] = df["teams"].str.split(" vs ", expand=True).to_numpy()
    df["date"] = pd.to_datetime(df["date"])
    # print(df.head())
    return df.drop(columns=["level", "gender", "teams"])


def get_matches_index(data_dir: Path):
    readme_path = data_dir / "README.txt"
    with readme_path.open() as file:
        lines = file.readlines()
        data = (
            line.strip().split(" - ")
            for line in lines
            if bool(re.match(r"^\d{4}-\d{2}-\d{2}", line))
        )
    df = pd.DataFrame(
        data, columns=["date", "level", "format", "gender", "id", "teams"]
    )
    df = df.loc[lambda x: x["level"] == "international"]
    df[["team1", "team2"]] = df["teams"].str.split(" vs ", expand=True).to_numpy()
    df["date"] = pd.to_datetime(df["date"])
    df["id"] = df["id"].astype(int)
    # print(df.head())
    return (
        df.drop(columns=["level", "format", "gender", "teams"])
        .sort_values(by="date")
        .reset_index(drop=True)
        .sort_index(axis="columns")
    )


def construct_dataframes(data_dir: Path, save_dir: Path = None):
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    index = get_matches_index(data_dir=data_dir)
    player_registry = pd.Series([])
    {}.get("id")
    print(index.head())
    data = []
    for match_id in tqdm(index["id"].to_numpy()):
        overs_data = []
        filepath = data_dir / f"{match_id}.json"
        # print(f"{filepath} is file: {filepath.is_file()}")
        with open(filepath) as file:
            json_data = json.load(file)
        for innings, innings_data in enumerate(json_data["innings"]):
            for over in innings_data["overs"]:
                df = pd.json_normalize(over["deliveries"]).reset_index(names="ball")
                df["innings"] = innings
                df["over"] = over["over"]
                overs_data.append(df)
        df = pd.concat(overs_data).reset_index(drop=True)
        if "wickets" in df.columns:
            df = df.join(
                df.loc[df["wickets"].notna(), "wickets"]
                .apply(
                    lambda x: {
                        "dismissal_type": x[0].get("kind"),
                        "player_out": x[0].get("player_out"),
                        "fielder": x[0].get("fielders")[0].get("name")
                        if x[0].get("fielders")
                        else np.nan,
                    }
                )
                .apply(pd.Series)
            ).drop(columns="wickets")

        registry = pd.Series(json_data["info"]["registry"]["people"])
        player_registry = pd.concat(
            [player_registry, pd.Series(registry.index, index=registry)]
        )
        for col in df.columns.intersection(
            ["batter", "bowler", "non_striker", "player_out", "fielder"]
        ):
            df[col] = registry.reindex(df[col]).to_numpy()
            # df[f"{col}_id"] = registry.reindex(df[col]).to_numpy()

        df["match_id"] = match_id
        data.append(df)
        # break
    df = pd.concat(data, ignore_index=True).sort_index(axis="columns")
    cols = df.columns[df.columns.str.contains("review|replacement")]
    if save_dir is not None:
        index.to_feather(save_dir / "index.feather")
        player_registry[~player_registry.index.duplicated(keep="last")].to_frame(
            "player"
        ).to_feather(save_dir / "player_registry.feather")
        df.drop(columns=cols).to_feather(save_dir / "data.feather")


def analyze_batsmen(df: pd.DataFrame, runs_col: str):
    return (
        pd.concat(
            [
                df.pivot_table(
                    index="batter",
                    # index=["batter", "bowler"],
                    values=runs_col,
                    aggfunc={runs_col: ["sum", "count", "mean"]},
                ),
                df.value_counts("player_out").rename("out"),
            ],
            axis="columns",
        )
        .dropna()
        .sort_values("count", ascending=False)
    )


def analyze_data(data_dir: Path):
    df = pd.read_feather(data_dir / "data.feather")
    index = pd.read_feather(data_dir / "index.feather")
    registry = pd.read_feather(data_dir / "player_registry.feather")["player"]
    # for col in ["batter", "bowler", "non_striker", "player_out", "fielder"]:
    #     df[col] = registry.reindex(df[col]).to_numpy()
    mean_std = (
        df.pivot_table(
            index="match_id",
            values="runs.total",
            aggfunc={"runs.total": ["mean", "std"]},
        )
        .reindex(df["match_id"])
        .set_axis(df.index)
    )
    runs = (df["runs.total"] - mean_std["mean"]) / mean_std["std"]
    df["runs.norm"] = runs.set_axis(df.index)

    # batsmen = batsmen.reindex(
    #     batsmen[["count", "mean"]]
    #     .rank(pct=True)
    #     .prod(axis=1)
    #     .sort_values(ascending=False)
    #     .index
    # )
    # .head(100)
    # .sort_values("mean", ascending=False)
    batsmen = analyze_batsmen(df, "runs.norm")
    batsmen["avg"] = batsmen["sum"] / batsmen["out"]
    batsmen.index = batsmen.index.map(registry)
    batsmen[lambda df: df["out"].rank(pct=True) > 0.95].sort_values(
        "avg", ascending=False
    ).head(15)
    k: pd.Series = (index["date"].iloc[0] - index["date"]).dt.days.set_axis(index["id"])
    discount_factor = 0.999
    weights = discount_factor**k
    # weights = pd.Series(1, index=index["id"])
    df["weights"] = weights.reindex(df["match_id"]).set_axis(df.index)
    df["runs.weighted"] = df["runs.norm"] * df["weights"]
    batsmen = df.pivot_table(
        index="batter",
        # index=["batter", "bowler"],
        # values="runs.weighted",
        aggfunc={
            "runs.weighted": "sum",
            "weights": "sum",
        },
    )
    batsmen["strike_rate"] = batsmen["runs.weighted"] / batsmen["weights"]
    batsmen.index = batsmen.index.map(registry)
    rank = batsmen.rank(pct=True)
    batsmen[(rank["weights"] > 0.90) & (rank["strike_rate"] > 0.90)].sort_values(
        # "runs.weighted",
        # "weights",
        "strike_rate",
        ascending=False,
    ).head(30)

    return


def main():
    paths = [
        Path("/", "home", "debnath", "Files", "cricsheet.org", "t20s_male_json"),
        Path("/", "storage2", "Files", "cricsheet.org", "t20s_male_json"),
    ]
    for path in paths:
        if path.exists():
            data_dir = path
            break
    else:
        print("No valid data directory found.")
        return

    # construct_dataframes(
    #     data_dir=data_dir,
    #     save_dir=data_dir.parent / "data",
    # )
    analyze_data(data_dir=data_dir.parent / "data")


def temp():
    scores = pd.read_csv(
        "/home/debnath/Documents/IIITH/sem_6/SMAI/Project/SMAI Project Evaluation and Discussion Slots - Evaluation.csv"
    )
    print(scores)


if __name__ == "__main__":
    # temp()
    main()
