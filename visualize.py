import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(data_dir: Path):
    """Load the feather files into DataFrames"""
    index = pd.read_feather(data_dir / "index.feather")
    player_registry = pd.read_feather(data_dir / "player_registry.feather")
    match_data = pd.read_feather(data_dir / "data.feather")
    return index, player_registry, match_data

def visualize_match_index(index: pd.DataFrame):
    """Visualize match index data"""
    plt.figure(figsize=(12, 6))
    
    # Plot matches over time
    plt.subplot(1, 2, 1)
    index['date'].value_counts().sort_index().plot()
    plt.title('Number of Matches Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Matches')
    
    # Plot team participation
    plt.subplot(1, 2, 2)
    team_counts = pd.concat([index['team1'], index['team2']]).value_counts()
    team_counts.plot(kind='bar')
    plt.title('Team Participation')
    plt.xlabel('Team')
    plt.ylabel('Number of Matches')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def visualize_match_data(match_data: pd.DataFrame):
    """Visualize match data"""
    plt.figure(figsize=(15, 10))
    
    # Plot runs distribution
    plt.subplot(2, 2, 1)
    sns.histplot(match_data['runs.total'], bins=30)
    plt.title('Distribution of Runs per Ball')
    plt.xlabel('Runs')
    
    # Plot dismissal types
    plt.subplot(2, 2, 2)
    if 'dismissal_type' in match_data.columns:
        match_data['dismissal_type'].value_counts().plot(kind='bar')
        plt.title('Types of Dismissals')
        plt.xlabel('Dismissal Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    
    # Plot runs by over
    plt.subplot(2, 2, 3)
    runs_by_over = match_data.groupby('over')['runs.total'].mean()
    runs_by_over.plot()
    plt.title('Average Runs per Over')
    plt.xlabel('Over Number')
    plt.ylabel('Average Runs')
    
    # Plot top run scorers
    plt.subplot(2, 2, 4)
    top_batters = match_data.groupby('batter')['runs.total'].sum().sort_values(ascending=False).head(10)
    top_batters.plot(kind='bar')
    plt.title('Top 10 Run Scorers')
    plt.xlabel('Batter')
    plt.ylabel('Total Runs')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def visualize_batsmen_analysis(match_data: pd.DataFrame, index: pd.DataFrame, player_registry: pd.DataFrame):
    """Visualize batsmen analysis results"""
    # Calculate basic statistics
    batsmen_stats = match_data.groupby('batter').agg({
        'runs.total': ['sum', 'count'],
        'player_out': 'count'
    }).reset_index()
    
    batsmen_stats.columns = ['batter', 'total_runs', 'balls_faced', 'times_out']
    batsmen_stats['average'] = batsmen_stats['total_runs'] / batsmen_stats['times_out']
    batsmen_stats['strike_rate'] = batsmen_stats['total_runs'] / batsmen_stats['balls_faced'] * 100
    
    # Convert player IDs to names
    batsmen_stats['batter_name'] = batsmen_stats['batter'].map(player_registry['player'])
    
    # Filter for batsmen with significant data
    significant_batsmen = batsmen_stats[batsmen_stats['balls_faced'] > 100]
    
    # Plot 1: Top 10 Run Scorers
    plt.figure(figsize=(12, 6))
    top_runs = significant_batsmen.nlargest(10, 'total_runs')
    plt.bar(top_runs['batter_name'], top_runs['total_runs'])
    plt.title('Top 10 Run Scorers')
    plt.xlabel('Batsman')
    plt.ylabel('Total Runs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Top 10 Best Averages
    plt.figure(figsize=(12, 6))
    top_average = significant_batsmen.nlargest(10, 'average')
    plt.bar(top_average['batter_name'], top_average['average'])
    plt.title('Top 10 Best Batting Averages')
    plt.xlabel('Batsman')
    plt.ylabel('Average')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Top 10 Best Strike Rates
    plt.figure(figsize=(12, 6))
    top_strike_rate = significant_batsmen.nlargest(10, 'strike_rate')
    plt.bar(top_strike_rate['batter_name'], top_strike_rate['strike_rate'])
    plt.title('Top 10 Best Strike Rates')
    plt.xlabel('Batsman')
    plt.ylabel('Strike Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot 4: Most Consistent Batsmen (lowest std deviation)
    plt.figure(figsize=(12, 6))
    consistency = match_data.groupby(['match_id', 'batter'])['runs.total'].sum().reset_index()
    consistency = consistency.groupby('batter')['runs.total'].std().reset_index()
    consistency = consistency[consistency['batter'].isin(significant_batsmen['batter'])]
    consistency['batter_name'] = consistency['batter'].map(player_registry['player'])
    top_consistent = consistency.nsmallest(10, 'runs.total')
    plt.bar(top_consistent['batter_name'], top_consistent['runs.total'])
    plt.title('Top 10 Most Consistent Batsmen')
    plt.xlabel('Batsman')
    plt.ylabel('Standard Deviation of Runs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_bowler_analysis(match_data: pd.DataFrame, index: pd.DataFrame, player_registry: pd.DataFrame):
    """Visualize bowler analysis results"""
    # Calculate basic statistics
    bowler_stats = match_data.groupby('bowler').agg({
        'runs.total': ['sum', 'count'],
        'player_out': 'count'
    }).reset_index()
    
    bowler_stats.columns = ['bowler', 'runs_conceded', 'balls_bowled', 'wickets_taken']
    bowler_stats['average'] = bowler_stats['runs_conceded'] / bowler_stats['wickets_taken']
    bowler_stats['strike_rate'] = bowler_stats['balls_bowled'] / bowler_stats['wickets_taken']
    bowler_stats['economy'] = bowler_stats['runs_conceded'] / (bowler_stats['balls_bowled'] / 6)
    
    # Convert player IDs to names
    bowler_stats['bowler_name'] = bowler_stats['bowler'].map(player_registry['player'])
    
    # Filter for bowlers with significant data
    significant_bowlers = bowler_stats[bowler_stats['balls_bowled'] > 100]
    
    # Plot 1: Top 10 Wicket Takers
    plt.figure(figsize=(12, 6))
    top_wickets = significant_bowlers.nlargest(10, 'wickets_taken')
    plt.bar(top_wickets['bowler_name'], top_wickets['wickets_taken'])
    plt.title('Top 10 Wicket Takers')
    plt.xlabel('Bowler')
    plt.ylabel('Total Wickets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Top 10 Most Economical Bowlers
    plt.figure(figsize=(12, 6))
    top_economy = significant_bowlers.nsmallest(10, 'economy')
    plt.bar(top_economy['bowler_name'], top_economy['economy'])
    plt.title('Top 10 Most Economical Bowlers')
    plt.xlabel('Bowler')
    plt.ylabel('Economy Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Top 10 Best Strike Rates
    plt.figure(figsize=(12, 6))
    top_strike_rate = significant_bowlers.nsmallest(10, 'strike_rate')
    plt.bar(top_strike_rate['bowler_name'], top_strike_rate['strike_rate'])
    plt.title('Top 10 Best Strike Rates')
    plt.xlabel('Bowler')
    plt.ylabel('Strike Rate (balls per wicket)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot 4: Top 10 Best Averages
    plt.figure(figsize=(12, 6))
    top_average = significant_bowlers.nsmallest(10, 'average')
    plt.bar(top_average['bowler_name'], top_average['average'])
    plt.title('Top 10 Best Bowling Averages')
    plt.xlabel('Bowler')
    plt.ylabel('Average (runs per wicket)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    paths = [
        Path("/","home","om","Downloads")
    ]
    data_dir = Path("/home/om/Downloads/data")  # Update this path to your data directory
    index, player_registry, match_data = load_data(data_dir)
    
    print("\nMatch Index Summary:")
    print(index.describe())
    print("\nMatch Data Summary:")
    print(match_data.describe())
    
    visualize_match_index(index)
    visualize_match_data(match_data)
    visualize_batsmen_analysis(match_data, index, player_registry)
    visualize_bowler_analysis(match_data, index, player_registry)

if __name__ == "__main__":
    main() 