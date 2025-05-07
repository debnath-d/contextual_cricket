import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBRegressor, plot_importance

np.set_printoptions(precision=3, suppress=True)


class CricketModel(nn.Module):
    def __init__(self, input_size):
        super(CricketModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x


def prepare_data(X: pd.DataFrame, y: pd.Series):
    # Convert to numpy arrays
    X_np = X.to_numpy()
    y_np = y.to_numpy().reshape(-1, 1)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)

    # Convert to torch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    return X_tensor, y_tensor, scaler


def plot_losses(train_losses: list, test_losses: list, plot_dir: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / "loss_torch_model.png")
    plt.close()


def train_torch_model(
    X_train,
    X_test,
    y_train,
    y_test,
    epochs=100,
    batch_size=32,
    plot_dir: Path = None,
    model_save_dir: Path = None,
):
    # Prepare data
    X_train, y_train, scaler = prepare_data(X_train, y_train)
    X_test, y_test, _ = prepare_data(X_test, y_test)
    joblib.dump(scaler, model_save_dir / "scaler.pkl")

    # Initialize model
    input_size = X_train.shape[1]
    model = CricketModel(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    train_losses = []
    test_losses = []
    best_loss = float("inf")
    if model_save_dir is not None:
        best_model_path = model_save_dir / "best_torch_model.pt"

    progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")

    # Training loop
    for epoch in progress_bar:
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = (total_loss / len(train_loader)) ** 0.5  # RMSE
        train_losses.append(avg_train_loss)

        # Test
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item() ** 0.5  # RMSE
            test_losses.append(test_loss)

        # Save model and scaler if test_loss improves
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), best_model_path)

        # Update progress bar with current losses
        progress_bar.set_postfix(
            {
                "Test Loss": f"{test_loss:.4f}",
                "Train Loss": f"{avg_train_loss:.4f}",
            }
        )

        # Plot and save loss curve
        if plot_dir is not None:
            plot_losses(train_losses, test_losses, plot_dir=plot_dir)

    print(f"Best Test Loss: {best_loss:.4f}")
    return train_losses, test_losses


def get_matches_index(data_dir: Path, save_dir: Path = None):
    """
    Construct the index of matches from the README file.
    Dataset: https://cricsheet.org/downloads/t20s_male_json.zip
    """
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

    match_stats = df.pivot_table(
        index="match_id",
        columns="innings",
        aggfunc={
            "runs.total": "sum",
            "player_out": "count",
            "over": "last",
            "ball_legit": "last",
        },
    ).swaplevel(axis="columns")
    first_innings, second_innings = match_stats[0], match_stats[1]
    not_all_out = first_innings["player_out"] < 10

    chase_lose = match_stats[1, "runs.total"] < match_stats[0, "runs.total"]
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
        "experience": balls_weighted,
        "avg": average,
        "sr": strike_rate,
    }


def analyze_bowlers(df: pd.DataFrame, runs_col: str, weights: pd.Series = None):
    if weights is None:
        weights = pd.Series(1, index=df["match_id"].unique())

    agg = df.pivot_table(
        columns="match_id",
        index="bowler",
        values=[runs_col, "is_legal", "batsman_out"],
        aggfunc="sum",
        fill_value=0,
    )

    # Compute weighted sums
    runs_weighted = agg[runs_col] @ weights
    balls_weighted = agg["is_legal"] @ weights
    wickets_weighted = agg["batsman_out"] @ weights

    # Calculate bowling metrics
    bowling_average = runs_weighted / wickets_weighted
    economy = runs_weighted / balls_weighted
    strike_rate = balls_weighted / wickets_weighted

    return {
        "experience": balls_weighted,
        "avg": bowling_average,
        "eco": economy,
        "sr": strike_rate,
    }


def analyze_batsmen_bowlers(df: pd.DataFrame, runs_col: str, weights: pd.Series = None):
    if weights is None:
        weights = pd.Series(1, index=df["match_id"].unique())
    agg = df.pivot_table(
        index=["batter", "bowler", "match_id"],
        values=[runs_col, "is_legal", "batsman_out"],
        aggfunc="sum",
    )
    results = {key: val for key, val in agg.items()}

    match_sr = (agg[runs_col] / agg["is_legal"]).rename("match_sr")

    agg = agg.unstack(fill_value=0)

    runs_weighted = agg[runs_col] @ weights
    balls_weighted = agg["is_legal"] @ weights
    wickets_weighted = agg["batsman_out"] @ weights
    batting_strike_rate = runs_weighted / balls_weighted  # bowler's economy rate
    average = runs_weighted / wickets_weighted
    bowling_strike_rate = balls_weighted / wickets_weighted

    return results | {
        "experience": balls_weighted,
        "match_sr": match_sr,
        "bat_sr": batting_strike_rate,
        "avg": average,
        "bowl_sr": bowling_strike_rate,
    }


def filter_data(df: pd.DataFrame, index: pd.DataFrame):
    """
    This does three things:
    - Filter out matches that are not between full members.
    - Filter out matches with no result or where the D/L method was used.
    - Filter out super overs.
    """
    full_members = [
        "India",
        "New Zealand",
        "Australia",
        "England",
        "South Africa",
        "Pakistan",
        "Bangladesh",
        "Sri Lanka",
        "West Indies",
        "Zimbabwe",
        "Ireland",
    ]
    index = index[index["team1"].isin(full_members) & index["team2"].isin(full_members)]
    df = df[df["match_id"].isin(index["id"])]

    no_result = index["result"] == "no result"
    d_l_method = index["method"].notna()
    condition = no_result | d_l_method
    remove_ids = index.loc[condition, "id"].to_numpy()
    index = index[~condition].drop(columns="method").reset_index(drop=True)
    df = df[~df["match_id"].isin(remove_ids)]
    df = df[df["innings"] < 2].reset_index(drop=True)

    return df, index


def create_dataset(
    stats: dict[str, dict[str, pd.DataFrame | pd.Series]],
    index: pd.DataFrame,
    target_col: str,
):
    """
    Create a dataset for training the model.
    stats: dict of stats for each player
    index: index of matches
    target_col: target column to predict

    Returns:
    X: features
    y: target
    """
    df: pd.DataFrame = stats["matchup"][target_col].reset_index()
    df["date"] = df["match_id"].map(index["date"].set_axis(index["id"]))

    del stats["matchup"]

    dates = index["date"].drop_duplicates().sort_values().reset_index(drop=True)
    dates.index = dates
    df["prev_date"] = df["date"].map(dates.shift(1))

    # create feature columns
    for stat_type, stat in stats.items():
        for key, val in stat.items():
            df[f"{stat_type}_{key}"] = pd.Index(df[[stat_type, "prev_date"]]).map(
                val.stack()
            )
    df = (
        df.drop(columns=["date", "prev_date", "batter", "bowler"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    # df[["batter", "bowler"]] = df[["batter", "bowler"]].astype("category")
    return df.drop(columns=target_col), df[target_col]


def create_plot(
    data: pd.DataFrame,
    plot_title: str,
    y_label: str,
    legend_title: str,
    save_path: Path = None,
):
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, dashes=False)

    plt.title(plot_title, fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_stats(
    stats: dict[str, dict[str, pd.DataFrame | pd.Series]],
    registry: pd.Series,
    plot_dir: Path = None,
):
    for stat_type, stat in stats.items():
        for key, val in stat.items():
            val[np.isinf(val)] = np.nan

    # batsman
    avg_pct = stats["batter"]["avg"].iloc[:, -1].rank(pct=True)
    sr_pct = stats["batter"]["sr"].iloc[:, -1].rank(pct=True)
    exp_pct = stats["batter"]["experience"].iloc[:, -1].rank(pct=True)

    cond = (avg_pct > 0.7) & (sr_pct > 0.65) & (exp_pct > 0.98)

    top = cond[cond].index
    avg = stats["batter"]["avg"].loc[top]
    sr = stats["batter"]["sr"].loc[top]

    avg.index = avg.index.map(registry)
    sr.index = sr.index.map(registry)

    create_plot(
        data=avg.T,
        plot_title="Average Over Time (Higher is better) for Top Batsmen",
        y_label="Average",
        legend_title="Batsmen",
        save_path=(plot_dir / "top_batsmen_average.png") if plot_dir else None,
    )
    create_plot(
        data=sr.T,
        plot_title="Srike Rate Over Time (Higher is better) for Top Batsmen",
        y_label="Strike Rate",
        legend_title="Batsmen",
        save_path=(plot_dir / "top_batsmen_strike_rate.png") if plot_dir else None,
    )
    # bowlers
    eco_pct = stats["bowler"]["eco"].iloc[:, -1].rank(pct=True, ascending=False)
    avg_pct = stats["bowler"]["avg"].iloc[:, -1].rank(pct=True, ascending=False)
    sr_pct = stats["bowler"]["sr"].iloc[:, -1].rank(pct=True, ascending=False)
    exp_pct = stats["bowler"]["experience"].iloc[:, -1].rank(pct=True)

    cond = (avg_pct > 0.7) & (sr_pct > 0.65) & (eco_pct > 0.7) & (exp_pct > 0.8)

    top = cond[cond].index
    registry[top]
    avg = stats["bowler"]["avg"].loc[top]
    sr = stats["bowler"]["sr"].loc[top]
    eco = stats["bowler"]["eco"].loc[top]

    avg.index = avg.index.map(registry)
    sr.index = sr.index.map(registry)
    eco.index = eco.index.map(registry)

    create_plot(
        data=avg.T,
        plot_title="Average Over Time (Lower is better) for Top Bowlers",
        y_label="Average",
        legend_title="Bowlers",
        save_path=(plot_dir / "top_bowlers_average.png") if plot_dir else None,
    )
    create_plot(
        data=sr.T,
        plot_title="Srike Rate Over Time (Lower is better) for Top Bowlers",
        y_label="Strike Rate",
        legend_title="Bowlers",
        save_path=(plot_dir / "top_bowlers_strike_rate.png") if plot_dir else None,
    )
    create_plot(
        data=eco.T,
        plot_title="Economy Over Time (Lower is better) for Top Bowlers",
        y_label="Economy",
        legend_title="Bowlers",
        save_path=(plot_dir / "top_bowlers_economy.png") if plot_dir else None,
    )

    # matchups

    avg_pct = stats["matchup"]["avg"].iloc[:, -1].rank(pct=True)
    bat_sr_pct = stats["matchup"]["bat_sr"].iloc[:, -1].rank(pct=True)
    bowl_sr_pct = stats["matchup"]["bowl_sr"].iloc[:, -1].rank(pct=True)
    exp_pct = stats["matchup"]["experience"].iloc[:, -1].rank(pct=True)

    cond = (
        (exp_pct > 0.98)
        & ((avg_pct > 0.85) | (avg_pct < 0.15))
        & ((bat_sr_pct > 0.85) | (bat_sr_pct < 0.15))
        & ((bowl_sr_pct > 0.85) | (bowl_sr_pct < 0.15))
    )

    top = cond[cond].index

    avg = stats["matchup"]["avg"].loc[top]
    bat_sr = stats["matchup"]["bat_sr"].loc[top]
    bowl_sr = stats["matchup"]["bowl_sr"].loc[top]

    avg.index = avg.index.map(lambda x: f"{registry[x[0]]} vs {registry[x[1]]}")
    bat_sr.index = bat_sr.index.map(lambda x: f"{registry[x[0]]} vs {registry[x[1]]}")
    bowl_sr.index = bowl_sr.index.map(lambda x: f"{registry[x[0]]} vs {registry[x[1]]}")

    create_plot(
        data=avg.T,
        plot_title="Average Over Time for Top Matchups",
        y_label="Average",
        legend_title="Batsmen vs Bowler",
        save_path=(plot_dir / "top_matchup_average.png") if plot_dir else None,
    )
    create_plot(
        data=bat_sr.T,
        plot_title="Batsmen Strike Rate (Bowler Economy) Over Time for Top Matchups",
        y_label="Batsmen Strike Rate (Bowler Economy)",
        legend_title="Batsmen vs Bowler",
        save_path=(plot_dir / "top_matchup_batting_strike_rate.png"),
    )
    create_plot(
        data=bowl_sr.T,
        plot_title="Bowling Strike Rate Over Time for Top Matchups",
        y_label="Bowling Strike Rate",
        legend_title="Batsmen vs Bowler",
        save_path=(plot_dir / "top_matchup_bowling_strike_rate.png"),
    )


def analyze_data(data_dir: Path, plot_dir: Path = None):
    df = pd.read_feather(data_dir / "data.feather")
    index = pd.read_feather(data_dir / "index.feather")
    registry = pd.read_feather(data_dir / "player_registry.feather")["player"]

    df, index = filter_data(df=df, index=index)

    df["runs.norm"] = normalize_runs(
        df=df,
        runs_col="runs.total",
        overs=index["overs"].set_axis(index["id"]).fillna(20).sort_index(),
    )
    runs_col = "runs.norm"
    # Identify legal deliveries and wickets credited to the bowler
    df["is_legal"] = df["extras.wides"].isna() & df["extras.noballs"].isna()
    df["batsman_out"] = df["dismissal_type"].isin(
        ["caught", "bowled", "lbw", "stumped", "caught and bowled", "hit wicket"]
    )
    weights = calculate_weights(index=index, discount_factor=0.99962)
    stats = {}

    stats["batter"] = analyze_batsmen(df, runs_col=runs_col, weights=weights)
    stats["bowler"] = analyze_bowlers(df=df, runs_col=runs_col, weights=weights)
    stats["matchup"] = analyze_batsmen_bowlers(
        df=df, runs_col=runs_col, weights=weights
    )

    plot_stats(stats=stats, registry=registry, plot_dir=plot_dir)

    X, y = create_dataset(stats=stats, index=index, target_col="match_sr")
    train_id, test_id = train_test_split(index["id"], test_size=0.2, random_state=42)
    X_train = X[X["match_id"].isin(train_id)].drop(columns="match_id")
    X_test = X[X["match_id"].isin(test_id)].drop(columns="match_id")
    y_train = y[X["match_id"].isin(train_id)]
    y_test = y[X["match_id"].isin(test_id)]

    return X_train, X_test, y_train, y_test


def train_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    plot_dir: Path = None,
    model_save_dir: Path = None,
):
    model_reg = XGBRegressor(
        enable_categorical=True,
        early_stopping_rounds=20,
    )
    model_reg.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
    )
    results = model_reg.evals_result()

    plt.figure(figsize=(10, 6))
    plt.plot(results["validation_0"]["rmse"], label="Training loss")
    plt.plot(results["validation_1"]["rmse"], label="Validation loss")
    plt.title("Loss (RMSE) vs Number of Trees")
    plt.xlabel("Number of trees")
    plt.ylabel("Loss (RMSE)")
    plt.legend()
    if plot_dir is not None:
        plt.savefig(plot_dir / "loss.png")
    else:
        plt.show()

    # Evaluate
    y_pred_train = model_reg.predict(X_train)
    mse = root_mean_squared_error(y_train, y_pred_train)
    print(f"Regression RMSE (train): {mse:.4f}")  # 0.5308

    y_pred_test = model_reg.predict(X_test)
    mse = root_mean_squared_error(y_test, y_pred_test)
    print(f"Regression RMSE (test): {mse:.4f}")  # 0.5383

    _, ax = plt.subplots(figsize=(10, 6))
    plot_importance(model_reg, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=60)
    if plot_dir is not None:
        plt.savefig(plot_dir / "feature_importance.png")
    else:
        plt.show()

    # Save XGBoost model
    if model_save_dir is not None:
        model_reg.save_model(str(model_save_dir / "xgboost_model.json"))


def main():
    paths = [
        Path("/", "home", "debnath", "Files", "cricsheet.org", "t20s_male_json"),
        Path("/", "storage2", "Files", "cricsheet.org", "t20s_male_json"),
    ]
    plot_dir = Path("plots")
    model_save_dir = Path("models")
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
    if model_save_dir is not None:
        model_save_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            data_dir = path
            break
    else:
        print("No valid data directory found.")
        return

    construct_dataframes(
        data_dir=data_dir,
        save_dir=data_dir.parent / "data",
    )
    X_train, X_test, y_train, y_test = analyze_data(
        data_dir=data_dir.parent / "data", plot_dir=plot_dir
    )
    train_xgboost(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        plot_dir=plot_dir,
        model_save_dir=model_save_dir,
    )

    train_torch_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        plot_dir=plot_dir,
        model_save_dir=model_save_dir,
    )


if __name__ == "__main__":
    main()
