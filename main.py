import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)


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
    # print(index.head())
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


def normalize_runs(df: pd.DataFrame, runs_col: str):
    mean_std = (
        df.pivot_table(
            index="match_id",
            values=runs_col,
            aggfunc={runs_col: ["mean", "std"]},
        )
        .reindex(df["match_id"])
        .set_axis(df.index)
    )
    norm_runs = (df[runs_col] - mean_std["mean"]) / mean_std["std"]
    return norm_runs


def calculate_weights(df: pd.DataFrame, index: pd.DataFrame):
    discount_factor = 0.999
    dates = index["date"].drop_duplicates().rename(None)
    k = (dates.iloc[0] - dates).dt.days.to_numpy().reshape(-1, 1)

    mask = np.triu(np.ones((len(dates), len(dates)), dtype=bool))

    weights = pd.DataFrame(
        discount_factor**k * mask,
        index=dates,
        columns=dates,
    )

    return weights.reindex(index["date"]).set_axis(index["id"])


def analyze_batsmen(df: pd.DataFrame, runs_col: str, weights: pd.DataFrame = None):
    if weights is None:
        weights = pd.Series(1, index=df.match_id.drop_duplicates())

    runs = df.pivot_table(
        columns="match_id",
        index="batter",
        values=runs_col,
        aggfunc={runs_col: ["sum", "count"]},
        fill_value=0,
    )
    print(runs.head())
    wickets = df.pivot_table(
        columns="match_id",
        index="player_out",
        aggfunc={"player_out": "count"},
        fill_value=0,
    )["player_out"]
    wickets[weights.index.difference(wickets.columns)] = 0

    runs_weighted = runs["sum"] @ weights
    balls_weighted = runs["count"] @ weights
    wickets_weighted = wickets @ weights
    average = runs_weighted / wickets_weighted
    strike_rate = runs_weighted / balls_weighted

    return average, strike_rate


def analyze_data(data_dir: Path):
    df = pd.read_feather(data_dir / "data.feather")
    index = pd.read_feather(data_dir / "index.feather")
    registry = pd.read_feather(data_dir / "player_registry.feather")["player"]

    df["runs.norm"] = normalize_runs(df=df, runs_col="runs.total")
    weights = calculate_weights(df=df, index=index)
    avg, sr = analyze_batsmen(df, runs_col="runs.norm", weights=weights)
    avg.index = avg.index.map(registry)
    sr.index = sr.index.map(registry)

    print(avg.loc["V Kohli"]["2022"])
    print(sr.loc["V Kohli"]["2022"])
    print(list(df.columns))

# Analyze bowlers
    # Convert weights DataFrame to a Series by summing across dates for overall metrics
    overall_weights = weights.sum(axis=1)
    bowler_stats = analyze_bowlers(df, weights=overall_weights)
    bowler_stats.index = bowler_stats.index.map(registry)
    # print(bowler_stats.loc["Jasprit Bumrah"])  # Example: print stats for a specific bowler
    
    return

# def analyze_bowlers(df: pd.DataFrame, weights: pd.Series = None):
#     print("Analyzing bowlers")
#     if weights is None:
#         weights = pd.Series(1, index=df.match_id.drop_duplicates())

#     runs_bowler = df.pivot_table(
#         columns="match_id",
#         index="bowler",
#         values="runs.total",
#         aggfunc={"runs.total": ["sum", "count"]},
#         fill_value=0,
#     )
#     print(runs_bowler.columns)
#     wickets_bolwer = df[df["player_out"].notna()].pivot_table(
#         columns="match_id",
#         index="bowler",
#         aggfunc={"player_out": "count"},
#         fill_value=0,
#     )
#     # Ensure all match columns exist
#     # for match in weights.index.difference(wickets.columns):
#         # wickets[match] = 0

#     runs_weighted = runs_bowler["sum"] @ weights
#     balls_weighted = runs_bowler["count"] @ weights
#     wickets_weighted = wickets_bolwer @ weights

#     average = runs_weighted / wickets_weighted.replace(0, np.nan)
#     strike_rate = balls_weighted / wickets_weighted.replace(0, np.nan)
#     economy = runs_weighted / (balls_weighted / 6)

#     print("Bowler weighted metrics calculated.")
#     return wickets_weighted, average, strike_rate, economy


def analyze_bowlers(df: pd.DataFrame, weights: pd.Series = None):
    if weights is None:
        weights = pd.Series(1, index=df['match_id'].unique())

    # Identify legal deliveries and wickets credited to the bowler
    df['is_legal'] = (df['extras.wides'].isna()) & (df['extras.noballs'].isna())
    df['is_wicket'] = df['dismissal_type'].notna() & (df['dismissal_type'] != 'run out')

    runs_conceded = df.pivot_table(
        columns="match_id",
        index="bowler",
        values="runs.total",
        aggfunc="sum",
        fill_value=0,
    )
    legal_balls = df.pivot_table(
        columns="match_id",
        index="bowler",
        values="is_legal",
        aggfunc="sum",
        fill_value=0,
    )
    wickets_taken = df.pivot_table(
        columns="match_id",
        index="bowler",
        values="is_wicket",
        aggfunc="sum",
        fill_value=0,
    )

    # Ensure all match_ids from weights are included
    all_match_ids = weights.index
    runs_conceded = runs_conceded.reindex(columns=all_match_ids, fill_value=0)
    legal_balls = legal_balls.reindex(columns=all_match_ids, fill_value=0)
    wickets_taken = wickets_taken.reindex(columns=all_match_ids, fill_value=0)

    # Compute weighted sums
    runs_conceded_weighted = runs_conceded @ weights
    legal_balls_weighted = legal_balls @ weights
    wickets_weighted = wickets_taken @ weights

    # Calculate bowling metrics
    bowling_average = runs_conceded_weighted / wickets_weighted
    bowling_average[wickets_weighted == 0] = np.nan

    economy_rate = runs_conceded_weighted / (legal_balls_weighted / 6)
    economy_rate[legal_balls_weighted == 0] = np.nan

    strike_rate = legal_balls_weighted / wickets_weighted
    strike_rate[wickets_weighted == 0] = np.nan

    total_wickets = wickets_weighted

    result = pd.DataFrame({
        'bowling_average': bowling_average,
        'economy_rate': economy_rate,
        'strike_rate': strike_rate,
        'total_wickets': total_wickets,
    })

    return result
def main():
    paths = [
        # Path("/", "home", "om", "Files", "cricsheet.org", "t20s_male_json"),
        # Path("/", "storage2", "Files", "cricsheet.org", "t20s_male_json"),
        Path("/","home","om","Downloads","t20s_male_json"),
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


if __name__ == "__main__":
    main()
