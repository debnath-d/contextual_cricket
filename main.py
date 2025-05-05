import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBRegressor

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


def get_matches_index(data_dir: Path, save_dir: Path = None):
    if save_dir is not None:
        filepath = save_dir / "index.feather"
        if filepath.is_file():
            return pd.read_feather(filepath)
    readme_path = data_dir / "README.txt"
    with readme_path.open() as file:
        lines = file.readlines()
        data = (
            line.strip().split(" - ")
            for line in lines
            if re.match(r"^\d{4}-\d{2}-\d{2}", line)
        )
    df = pd.DataFrame(
        data, columns=["date", "level", "format", "gender", "id", "teams"]
    )
    df = df.loc[lambda x: x["level"] == "international"]
    df[["team1", "team2"]] = df["teams"].str.split(" vs ", expand=True).to_numpy()
    df["date"] = pd.to_datetime(df["date"])
    df["id"] = df["id"].astype(int)
    df = (
        df.drop(columns=["level", "format", "gender", "teams"])
        .sort_values(by="date")
        .reset_index(drop=True)
        .sort_index(axis="columns")
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_feather(filepath)
    return df


def construct_dataframes(data_dir: Path, save_dir: Path = None):
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    index = get_matches_index(data_dir=data_dir, save_dir=save_dir)
    player_registry = pd.Series([])
    {}.get("id")
    print(index.head())
    data = []
    outcome_data = []
    for match_id in tqdm(index["id"].to_numpy()):
        overs_data = []
        filepath = data_dir / f"{match_id}.json"
        # print(f"{filepath} is file: {filepath.is_file()}")
        with open(filepath) as file:
            json_data = json.load(file)

        try:
            overs = (
                json_data["innings"][1]["target"]["overs"]
                if len(json_data["innings"]) > 1
                else None
            )
        except Exception as e:
            print(f"Error in match {match_id}: {e}")
            overs = np.nan
        outcome_data.append(
            json_data["info"]["outcome"] | {"id": match_id, "overs": overs}
        )
        for innings, innings_data in enumerate(json_data["innings"]):
            for over in innings_data["overs"]:
                df = pd.json_normalize(over["deliveries"]).reset_index(names="ball")
                df["innings"] = innings
                df["over"] = over["over"]
                cols = df.columns.intersection(["extras.wides", "extras.noballs"])
                df["ball_legit"] = df[cols].isna().all(axis="columns").cumsum()
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
    index = index.merge(pd.json_normalize(outcome_data, sep="_"), on="id")
    df = pd.concat(data, ignore_index=True).sort_index(axis="columns")
    cols = df.columns[df.columns.str.contains("review|replacement")]
    if save_dir is not None:
        index.to_feather(save_dir / "index.feather")
        player_registry[~player_registry.index.duplicated(keep="last")].to_frame(
            "player"
        ).to_feather(save_dir / "player_registry.feather")
        df.drop(columns=cols).to_feather(save_dir / "data.feather")


def update_mean_std(mean: pd.Series, std: pd.Series, N: pd.Series, K: pd.Series):
    """
    Compute new mean and standard deviation after adding K zeroes to a dataset.

    mean : Original mean of the dataset
    std : Original standard deviation of the dataset
    N : Number of original data points
    K : Number of zeroes to add

    Returns:
    new_mean : New mean after adding zeroes
    new_std : New standard deviation after adding zeroes
    """
    # Compute K/N ratio
    r = K.divide(N, fill_value=0)

    # New mean: mu' = mu / (1 + K/N)
    new_mean = mean / (1 + r)

    # New standard deviation: sqrt(sigma^2 / (1 + K/N) + (K/N) * mu'^2)
    new_std = np.sqrt((std**2) / (1 + r) + r * (new_mean**2))

    return new_mean, new_std


def normalize_runs(df: pd.DataFrame, runs_col: str, overs: pd.Series):
    mean_std = df.pivot_table(
        index="match_id",
        values=runs_col,
        aggfunc={runs_col: ["mean", "std", "count"]},
    )
    # mean_std = mean_std.reindex(df["match_id"]).set_axis(df.index)
    x = df.pivot_table(
        index="match_id",
        columns="innings",
        aggfunc={
            "runs.total": "sum",
            "player_out": "count",
            "over": "last",
            "ball_legit": "last",
        },
    ).swaplevel(axis="columns")
    first_innings, second_innings = x[0], x[1]
    not_all_out = first_innings["player_out"] < 10
    # less_than_20 = first_innings["over"] < 19
    # total_overs = 1 + (
    #     first_innings.loc[less_than_20 & not_all_out, "over"]
    #     .reindex_like(x)
    #     .fillna(19)
    #     .astype(int)
    # )

    chase_lose = x[1, "runs.total"] < x[0, "runs.total"]
    balls_remaining: pd.Series = (
        6 * (overs[~not_all_out] - first_innings.loc[~not_all_out, "over"])
        - first_innings.loc[~not_all_out, "ball_legit"]
    )
    balls_remaining = balls_remaining.add(
        6 * (overs[chase_lose] - second_innings.loc[chase_lose, "over"])
        - second_innings.loc[chase_lose, "ball_legit"],
        fill_value=0,
    )
    mean, std = update_mean_std(
        mean=mean_std["mean"],
        std=mean_std["std"],
        N=mean_std["count"],
        K=balls_remaining,
    )
    norm_runs = (df[runs_col] - df["match_id"].map(mean)) / df["match_id"].map(std)
    return norm_runs


def calculate_weights(index: pd.DataFrame, discount_factor: float):
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
        weights = pd.Series(1, index=df["match_id"].unique())

    runs = df.pivot_table(
        columns="match_id",
        index="batter",
        values=runs_col,
        aggfunc={runs_col: ["sum", "count"]},
        fill_value=0,
    )
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

    return {
        "avg": average,
        "sr": strike_rate,
    }


def analyze_bowlers(df: pd.DataFrame, runs_col: str, weights: pd.Series = None):
    if weights is None:
        weights = pd.Series(1, index=df["match_id"].unique())

    agg = df.pivot_table(
        columns="match_id",
        index="bowler",
        values=[runs_col, "is_legal", "batsman_dismissed"],
        aggfunc="sum",
        fill_value=0,
    )

    # Compute weighted sums
    runs_weighted = agg[runs_col] @ weights
    balls_weighted = agg["is_legal"] @ weights
    wickets_weighted = agg["batsman_dismissed"] @ weights

    # Calculate bowling metrics
    bowling_average = runs_weighted / wickets_weighted
    # bowling_average[wickets_weighted == 0] = np.nan
    economy = runs_weighted / balls_weighted
    # economy[balls_weighted == 0] = np.nan
    strike_rate = balls_weighted / wickets_weighted
    # strike_rate[wickets_weighted == 0] = np.nan

    # TODO experience = total balls played

    return {
        "avg": bowling_average,
        "eco": economy,
        "sr": strike_rate,
    }


def analyze_batsmen_bowlers(df: pd.DataFrame, runs_col: str, weights: pd.Series = None):
    if weights is None:
        weights = pd.Series(1, index=df["match_id"].unique())
    agg = df.pivot_table(
        # columns="match_id",
        index=["batter", "bowler", "match_id"],
        # aggfunc={
        #     runs_col: ["sum", "count"],
        #     "batsman_dismissed": "sum",
        # },
        values=[runs_col, "is_legal", "batsman_dismissed"],
        aggfunc="sum",
    )
    runs = agg[runs_col]
    return {runs_col: runs}

    agg = agg.unstack(fill_value=0)

    runs_weighted = agg[runs_col] @ weights
    balls_weighted = agg["is_legal"] @ weights
    wickets_weighted = agg["batsman_dismissed"] @ weights
    batting_strike_rate = runs_weighted / balls_weighted  # bowler's economy rate
    average = runs_weighted / wickets_weighted
    bowling_strike_rate = balls_weighted / wickets_weighted

    return {
        runs_col: runs,
        "bat_sr": batting_strike_rate,
        "avg": average,
        "bowl_sr": bowling_strike_rate,
    }


def filter_data(df: pd.DataFrame, index: pd.DataFrame):
    """
    This does two things:
    - Filter out matches with no result or where the D/L method was used.
    - Filter out super overs.
    """
    no_result = index["result"] == "no result"
    d_l_method = index["method"].notna()
    condition = no_result | d_l_method
    remove_ids = index.loc[condition, "id"].to_numpy()
    index = index[~condition].drop(columns="method").reset_index(drop=True)
    df = df[~df["match_id"].isin(remove_ids)]
    df = df[df["innings"] < 2].reset_index(drop=True)

    index = index.iloc[: len(index) // 5]  # TODO remove this line
    df = df[df["match_id"].isin(index["id"])]  # TODO remove this line

    return df, index


def create_dataset(stats: dict, index: pd.DataFrame, target_col: str):
    df: pd.DataFrame = stats["matchup"][target_col].reset_index()
    df["date"] = df["match_id"].map(index["date"].set_axis(index["id"]))
    dates = index["date"].drop_duplicates().sort_values().reset_index(drop=True)
    dates.index = dates
    df["prev_date"] = df["date"].map(dates.shift(1))
    for stat_type, stat in stats.items():
        for key, val in stat.items():
            if key == target_col:
                continue
            cols = ["batter", "bowler"] if stat_type == "matchup" else [stat_type]
            df[f"{stat_type}_{key}"] = pd.Index(df[cols + ["prev_date"]]).map(
                val.stack()
            )
    df = (
        df.drop(columns=["date", "prev_date"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    df[["batter", "bowler"]] = df[["batter", "bowler"]].astype("category")
    return df.drop(columns=target_col), df[target_col]


def analyze_data(data_dir: Path):
    df = pd.read_feather(data_dir / "data.feather")
    index = pd.read_feather(data_dir / "index.feather")
    registry = pd.read_feather(data_dir / "player_registry.feather")["player"]

    df, index = filter_data(df=df, index=index)

    if True:
        df["runs.norm"] = normalize_runs(
            df=df,
            runs_col="runs.total",
            overs=index["overs"].set_axis(index["id"]).fillna(20).sort_index(),
        )
        runs_col = "runs.norm"
    else:
        runs_col = "runs.total"
    # Identify legal deliveries and wickets credited to the bowler
    df["is_legal"] = df["extras.wides"].isna() & df["extras.noballs"].isna()
    df["batsman_dismissed"] = df["dismissal_type"].isin(
        [
            "caught",
            "bowled",
            "lbw",
            "stumped",
            "caught and bowled",
            "hit wicket",
        ]
    )
    weights = calculate_weights(index=index, discount_factor=0.999)
    stats = {}

    stats["bowler"] = analyze_bowlers(df=df, runs_col=runs_col, weights=weights)
    stats["batter"] = analyze_batsmen(df, runs_col=runs_col, weights=weights)
    stats["matchup"] = analyze_batsmen_bowlers(
        df=df, runs_col=runs_col, weights=weights
    )
    X, y = create_dataset(stats=stats, index=index, target_col=runs_col)
    train_id, test_id = train_test_split(index["id"], test_size=0.2, random_state=42)
    X_train = X[X["match_id"].isin(train_id)]
    X_test = X[X["match_id"].isin(test_id)]
    y_train = y[X["match_id"].isin(train_id)]
    y_test = y[X["match_id"].isin(test_id)]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model_reg = XGBRegressor(
        enable_categorical=True,
        # subsample=0.7,
        # evaluation_metric=mean_squared_error,
    )
    model_reg.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model_reg.predict(X_train)
    mse = mean_squared_error(y_train, y_pred_train)
    print(f"Regression MSE (train): {mse:.4f}")

    y_pred_test = model_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    print(f"Regression MSE (test): {mse:.4f}")

    # avg, sr = stats["bat"]["avg"], stats["bat"]["sr"]
    # avg.index = avg.index.map(registry)
    # sr.index = sr.index.map(registry)

    # print(avg.loc["V Kohli"]["2022"])
    # print(sr.loc["V Kohli"]["2022"])

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


if __name__ == "__main__":
    main()
