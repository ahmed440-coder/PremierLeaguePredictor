#############################################
#####-- IMPORTING LIBRARIES AND FILES --#####
#############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


###################################
#####-- LOADING THE DATASET --#####
###################################

df = pd.read_csv("pl2526.csv")

###############################
#####-- CLEAN AND PARSE --#####
###############################

##removing rows with empty results
df = df.dropna(subset=['Result'])
#splitting the result into the home goals and away goals
df[['HomeGoals','AwayGoals']] = df['Result'].str.split(' - ', expand=True).astype(int);

#OutCome column Creation
#H -> Home Wins, D -> draw, A -> Away Wins
def get_res(row):
  if row['HomeGoals'] > row['AwayGoals']:
    return 'H'
  elif row['HomeGoals'] < row['AwayGoals']:
    return 'A'
  else:
    return 'D'
df['outcome'] = df.apply(get_res, axis=1)

##############################
######-- ENGENEERING -- ######
##############################

teams = df['Home Team'].unique()
##initalizing stats
team_stats = {team :{ 'Form':[], 'Points':0, 'HomeGF':0, 'HomeGA':0,'AwayGF':0, 'AwayGA':0, 'HomeMatches':0, 'AwayMatches':0} for team in teams}
#features for each match
features = []

for idx, row in df.iterrows():
    
    home = row['Home Team']
    away = row['Away Team']

    if home not in team_stats:
        team_stats[home] = {'Form': [], 'Points':0, 'HomeGF':0, 'HomeGA':0, 'AwayGF':0, 'AwayGA':0, 'HomeMatches':0, 'AwayMatches':0}
    if away not in team_stats:
        team_stats[away] = {'Form': [], 'Points':0, 'HomeGF':0, 'HomeGA':0, 'AwayGF':0, 'AwayGA':0, 'HomeMatches':0, 'AwayMatches':0}
    #last 10 games points
    home_form_10 = team_stats[home]['Form'][-10:]
    away_form_10 = team_stats[away]['Form'][-10:]
    home_points_10 = sum([x[0] for x in home_form_10])
    away_points_10 = sum([x[0] for x in away_form_10])
    #GOALS SCORED IN THE LAST 10 GAMES
    home_GF_last10 = sum([x[1] for x in home_form_10])
    home_GA_last10 = sum([x[2] for x in home_form_10])
    away_GF_last10 = sum([x[1] for x in away_form_10])
    away_GA_last10 = sum([x[2] for x in away_form_10])
    #last 3 games points
    home_form_3 = team_stats[home]['Form'][-3:]
    away_form_3 = team_stats[away]['Form'][-3:]
    home_points_3 = sum([x[0] for x in home_form_3])
    away_points_3 = sum([x[0] for x in away_form_3])
    #LAST 5 GAMES POINTS
    home_form_5 = team_stats[home]['Form'][-5:]
    away_form_5 = team_stats[away]['Form'][-5:]
    home_points_5 = sum([x[0] for x in home_form_5])
    away_points_5 = sum([x[0] for x in away_form_5])  
    #goal difference last 10
    home_GD = sum([x[1]-x[2] for x in home_form_10])
    away_GD = sum([x[1]-x[2] for x in away_form_10])
    #DIFFERENCE
    points_diff_10 = home_points_10 - away_points_10
    points_diff_3 = home_points_3 - away_points_3
    gd_diff_10 = home_GD - away_GD
    #average goals scored per game
    home_avg_GF = team_stats[home]['HomeGF'] / team_stats[home]['HomeMatches'] if team_stats[home]['HomeMatches'] > 0 else 0
    home_avg_GA = team_stats[home]['HomeGA'] / team_stats[home]['HomeMatches'] if team_stats[home]['HomeMatches'] > 0 else 0
    away_avg_GF = team_stats[away]['AwayGF'] / team_stats[away]['AwayMatches'] if team_stats[away]['AwayMatches'] > 0 else 0
    away_avg_GA = team_stats[away]['AwayGA'] / team_stats[away]['AwayMatches'] if team_stats[away]['AwayMatches'] > 0 else 0
      
    #adding home advantage because teams play better at their staduims 
    home_advantage = 0.8
    #win rate
    home_win_rate = sum(1 for x in team_stats[home]['Form'] if x[0] == 3) / max (1,team_stats[home]['HomeMatches'])
    away_win_rate = sum(1 for x in team_stats[away]['Form'] if x[0] == 3) / max (1,team_stats[away]['AwayMatches'])
    #appending features
    features.append([home_points_10, away_points_10, home_points_3, away_points_3, home_GD, away_GD, home_advantage,
                    home_avg_GF, home_avg_GA, away_avg_GF, away_avg_GA, home_GF_last10, home_GA_last10, away_GF_last10, away_GA_last10,
                     home_points_5, away_points_5, points_diff_10, points_diff_3, gd_diff_10, home_win_rate, away_win_rate])
    #calculating points
    if row['outcome'] == 'H':
        home_points_match = 3
        away_points_match = 0
    elif row ['outcome'] == 'A':
        home_points_match = 0
        away_points_match = 3
    else:
        home_points_match = 1
        away_points_match = 1
    #adding home & away goals
    team_stats[home]['Form'].append((home_points_match, row['HomeGoals'], row['AwayGoals']))
    team_stats[away]['Form'].append((away_points_match, row['AwayGoals'], row['HomeGoals']))
    #updating teams after matches
    team_stats[home]['HomeGF'] += row['HomeGoals']
    team_stats[home]['HomeGA'] += row['AwayGoals']
    team_stats[home]['HomeMatches']+=1
    team_stats[away]['AwayGF'] += row['AwayGoals']
    team_stats[away]['AwayGA'] += row['HomeGoals']
    team_stats[away]['AwayMatches']+=1
#converting the feaature list to the dataframe
X = pd.DataFrame(features, columns=['HomePointsLast10', 'AwayPointsLast10','HomePointsLast3', 'AwayPointsLast3','HomePointsLast5','AwayPointsLast5',
                 'HomeGDLast10', 'AwayGDLast10', 'home_advantage',
                'HomeAvgGF','HomeAvgGA','AwayAvgGF','AwayAvgGA','HomeGFLast10', 'HomeGALast10',
                'AwayGFLast10','AwayGALast10','PointsDiff10',
                'PointsDiff3','GDDiff10','HomeWinRate','AwayWinRate'
                                    ])
y = df['outcome']
##############################
#####-- MODEL TRAINING --#####
##############################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

model = RandomForestClassifier(
    n_estimators = 500,
    max_depth=None,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

##############################
#####-- EVALUATE MODEL --#####
##############################

y_pred = model.predict(X_test)
print(f"ACCUARCY: {accuracy_score(y_test, y_pred):.2f}")
print("CONFUSION MATRIX : \n", confusion_matrix(y_test, y_pred))
##############################
#######  -- VISUALS -- ####### 	
##############################
def team_form(team):
    form = [x[0] for x in team_stats[team]['Form'][-10:]]
    games = list(range(1, len(form)+1))

    plt.figure(figsize=(8,4))
    plt.plot(games,form,marker='o', linestyle='-', color='blue')
    plt.title(f"{team} - Last {len(form)} games Form (Points)")
    plt.xlabel("Game")
    plt.ylabel("Points")
    plt.ylim(0,3)
    plt.grid(True)
    plt.show()
def compare_teams(team1,team2):
    team1_GF = team_stats[team1]['HomeGF']/team_stats[team1]['HomeMatches'] if team_stats[team1]['HomeMatches']>0 else 0
    team1_GA = team_stats[team1]['HomeGA']/team_stats[team1]['HomeMatches'] if team_stats[team1]['HomeMatches']>0 else 0
    team2_GF = team_stats[team2]['AwayGF']/team_stats[team2]['AwayMatches'] if team_stats[team2]['AwayMatches']>0 else 0
    team2_GA = team_stats[team2]['AwayGA']/team_stats[team2]['AwayMatches'] if team_stats[team2]['AwayMatches']>0 else 0
    labels = ['GF', 'GA']
    width = 0.4
    plt.figure(figsize=(6,4))
    plt.bar([x-width/2 for x in range(len(labels))], [team1_GF, team1_GA], width=width, label=team1, color='blue')
    plt.bar([x+width/2 for x in range(len(labels))], [team2_GF, team2_GA], width=width, label=team2, color='red')
    plt.xticks(range(len(labels)),labels)
    plt.ylabel('Goals')
    plt.title(f"{team1} VS {team2} - Average Goals Scored Comparaison")
    plt.legend()
    plt.show()
##############################
#####  -- USER INPUT -- ######
##############################

def predict_match(home_team, away_team):
    #form in the last 10 games
    home_form_10 = team_stats[home_team]['Form'][-10:]
    away_form_10 = team_stats[away_team]['Form'][-10:]
    #points in the last 10 games
    home_points_10 = sum([x[0] for x in home_form_10])
    away_points_10 = sum([x[0] for x in away_form_10])
    #goals for and goals against in the last 10 games (home team)
    home_GF_last10 = sum([x[1] for x in home_form_10])
    home_GA_last10 = sum([x[2] for x in home_form_10])
    #goals for and goals against in the last 10 games (away team)
    away_GF_last10 = sum([x[1] for x in away_form_10])
    away_GA_last10 = sum([x[2] for x in away_form_10])

    #form in the last 3 games
    home_form_3 = team_stats[home_team]['Form'][-3:]
    away_form_3 = team_stats[away_team]['Form'][-3:]
    #points in the last 3 games
    home_points_3 = sum([x[0] for x in home_form_3])
    away_points_3 = sum([x[0] for x in away_form_3])
    #form in the last 5 games
    home_form_5 = team_stats[home_team]['Form'][-5:]
    away_form_5 = team_stats[away_team]['Form'][-5:]
    #points in the last 5 games
    home_points_5 = sum([x[0] for x in home_form_5])
    away_points_5 = sum([x[0] for x in away_form_5])
    #goal difference (home & away side)
    home_GD = sum([x[1]-x[2] for x in home_form_10])
    away_GD = sum([x[1]-x[2] for x in away_form_10])
    #points difference last 10 games
    points_diff_10 = home_points_10 - away_points_10
    #points difference last 3 games
    points_diff_3 = home_points_3 - away_points_3
    #goal difference last 10 games
    gd_diff_10 = home_GD - away_GD
    #avg goals/game
    home_avg_GF = team_stats[home_team]['HomeGF'] / team_stats[home_team]['HomeMatches'] if team_stats[home_team]['HomeMatches']>0 else 0
    home_avg_GA = team_stats[home_team]['HomeGA'] / team_stats[home_team]['HomeMatches'] if team_stats[home_team]['HomeMatches']>0 else 0
    away_avg_GF = team_stats[away_team]['AwayGF'] / team_stats[away_team]['AwayMatches'] if team_stats[away_team]['AwayMatches']>0 else 0
    away_avg_GA = team_stats[away_team]['AwayGA'] / team_stats[away_team]['AwayMatches'] if team_stats[away_team]['AwayMatches']>0 else 0
    #winrate
    home_win_rate = sum(1 for x in team_stats[home_team]['Form'] if x[0] == 3) / max (1,len(team_stats[home_team]['Form']))
    away_win_rate = sum(1 for x in team_stats[away_team]['Form'] if x[0] == 3) / max (1,len(team_stats[away_team]['Form']))
    #CALCULATING HOME ADVANTAGE (MORE REALISTIC INSTEAD OF ASSIGNING A STATIC VALUE)
    home_advantage = 0.8
    match_features = pd.DataFrame(
        [[home_points_10, away_points_10, home_points_3, away_points_3, home_GD, away_GD, home_advantage,
                    home_avg_GF, home_avg_GA, away_avg_GF, away_avg_GA, home_GF_last10, home_GA_last10, away_GF_last10, away_GA_last10,
                     home_points_5, away_points_5, points_diff_10, points_diff_3, gd_diff_10, home_win_rate, away_win_rate]],
        columns=['HomePointsLast10', 'AwayPointsLast10','HomePointsLast3', 'AwayPointsLast3','HomePointsLast5','AwayPointsLast5',
                 'HomeGDLast10', 'AwayGDLast10', 'home_advantage',
                'HomeAvgGF','HomeAvgGA','AwayAvgGF','AwayAvgGA','HomeGFLast10', 'HomeGALast10',
                'AwayGFLast10','AwayGALast10','PointsDiff10',
                'PointsDiff3','GDDiff10','HomeWinRate','AwayWinRate'
                                    ]
    )
    pred = model.predict(match_features)[0]
    probs = model.predict_proba(match_features)[0]
    outcome_proba = dict(zip(model.classes_, probs))
    outcome_proba_betteroutput = {
        ('Home Team Wins' if k=='H' else 'Away Team Wins' if k=='A' else 'Draw'):
        f"{v*100:.1f}%" for k,v in outcome_proba.items()
    }

    print(f"Prediction for {home_team} V {away_team} : {pred}")
    print("Probabilites:")
    for k,v in outcome_proba_betteroutput.items():
        print(f" {k} : {v}")
    print("\nAvergae Goals per game ")
    print(f" {home_team} -> GF: {home_avg_GF:.2f}, GA: {home_avg_GA:.2f}")
    print(f" {away_team} -> GF: {away_avg_GF:.2f}, GA: {away_avg_GA:.2f}")

home = input("insert the Home team : ")
away = input("insert the Away team : ")
predict_match(home, away)
compare_teams(home,away)
team_form(home)
team_form(away)
