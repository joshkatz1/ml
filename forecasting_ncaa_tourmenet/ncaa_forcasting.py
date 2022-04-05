import os
import re
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.model_selection import *

DATA_PATH = '/media/senswim/9e0b629a-c266-4be1-9310-1212944ade0a/mens-march-mania-2022/MDataFiles_Stage1/'

for filename in os.listdir(DATA_PATH):
    print(filename)
# ranking before the start of the playoff
df_seeds = pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv")

df_season_results = pd.read_csv(DATA_PATH + "MRegularSeasonCompactResults.csv")
df_season_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
df_season_results['ScoreGap'] = df_season_results['WScore'] - df_season_results['LScore']

num_win = df_season_results.groupby(['Season', 'WTeamID']).count()
num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(
    columns={"DayNum": "NumWins", "WTeamID": "TeamID"})

num_loss = df_season_results.groupby(['Season', 'LTeamID']).count()
num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(
    columns={"DayNum": "NumLoss", "LTeamID": "TeamID"})
gap_win = df_season_results.groupby(['Season', "WTeamID"]).mean().reset_index()
gap_win = gap_win[['Season', "WTeamID", "ScoreGap"]].rename(columns={"WTeamID": "TeamID", "ScoreGap": "GapWins"})

gap_loss = df_season_results.groupby(['Season', "LTeamID"]).mean().reset_index()
gap_loss = gap_loss[['Season', "LTeamID", "ScoreGap"]].rename(columns={"LTeamID": "TeamID", "ScoreGap": "GapLoss"})
df_features_season_w = df_season_results.groupby(['Season', 'WTeamID']).count().reset_index()[
    ['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
df_features_season_l = df_season_results.groupby(['Season', 'LTeamID']).count().reset_index()[
    ['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})
df_features_season = pd.concat([df_features_season_w, df_features_season_l], 0).drop_duplicates().sort_values(
    ['Season', 'TeamID']).reset_index(drop=True)

df_features_season = df_features_season.merge(num_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(num_loss, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_win, on=['Season', 'TeamID'], how='left')
df_features_season = df_features_season.merge(gap_loss, on=['Season', 'TeamID'], how='left')

df_features_season.fillna(0, inplace=True)

df_features_season["WinRatio"] = df_features_season["NumWins"] / (
            df_features_season["NumWins"] + df_features_season["NumLoss"])
df_features_season['GapAvg'] = (
        (df_features_season['NumWins'] * df_features_season['GapWins'] -
         df_features_season['NumLoss'] * df_features_season['GapLoss'])
        / (df_features_season['NumWins'] + df_features_season['NumLoss']))
# df_features_season.drop(['NumWins', 'NumLoss', 'GapWins', 'GapLoss'], axis=1, inplace=True)
print(df_features_season.head())
df_tourney_results = pd.read_csv(DATA_PATH + "MNCAATourneyCompactResults.csv")
df_tourney_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)

df = df_tourney_results.copy()
# df = df[df['Season'] >= 2016].reset_index(drop=True)

df.head()
for key, val in df.iterrows():
    print(key, val)
df = pd.merge(df, df_seeds, how='left', left_on=['Season', 'WTeamID'],
              right_on=['Season', 'TeamID']).drop('TeamID', axis=1).rename(columns={"Seed": "SeedW"})

df = pd.merge(df, df_seeds, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID']).drop('TeamID',
                                                                                                           axis=1).rename(
    columns={"Seed": "SeedL"})


def treat_seed(seed):
    return int(re.sub("[^0-9]", "", seed))


df['SeedW'] = df['SeedW'].apply(treat_seed)
df['SeedL'] = df['SeedL'].apply(treat_seed)

df.head()
# pd .DataFrame(data=[df["SeedW"].value_counts().index.value(),df["SeedL"].value_counts()[1]],index=df["SeedW"].value_counts()[0])
df1 = pd.DataFrame(columns = ['seed', 'precent_of_wininig', 'precent_of_loosing'])

for key, val in df["SeedW"].value_counts().iteritems():
    for idx, values in df["SeedL"].value_counts().iteritems():
        if idx == key:
            seed = key
            precent_of_wininig = val /(val+values)
            precent_of_loosing = 1 - precent_of_wininig

            df1 = df1.append({"seed": seed, "precent_of_wininig": precent_of_wininig,
                    "precent_of_loosing": precent_of_loosing}, ignore_index=True)


print(df["SeedL"].value_counts())
df = pd.merge(
    df,
    df_features_season,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsW',
    'NumLosses': 'NumLossesW',
    'GapWins': 'GapWinsW',
    'GapLosses': 'GapLossesW',
    'WinRatio': 'WinRatioW',
    'GapAvg': 'GapAvgW',
}).drop(columns='TeamID', axis=1)

df = pd.merge(
    df,
    df_features_season,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsL',
    'NumLosses': 'NumLossesL',
    'GapWins': 'GapWinsL',
    'GapLosses': 'GapLossesL',
    'WinRatio': 'WinRatioL',
    'GapAvg': 'GapAvgL',
}).drop(columns='TeamID', axis=1)
print(df.head(10))
df_538 = pd.read_csv("/media/senswim/9e0b629a-c266-4be1-9310-1212944ade0a/a)/538ratingsMen.csv")
df_538.drop('TeamName', axis=1, inplace=True)

df_538.head()
df = pd.merge(
    df,
    df_538,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'538rating': '538ratingW'})

df = pd.merge(
    df,
    df_538,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'538rating': '538ratingL'})
df.head()


def add_loosing_matches(df):
    win_rename = {
        "WTeamID": "TeamIdA",
        "WScore": "ScoreA",
        "LTeamID": "TeamIdB",
        "LScore": "ScoreB",
    }
    win_rename.update({c: c[:-1] + "A" for c in df.columns if c.endswith('W')})
    win_rename.update({c: c[:-1] + "B" for c in df.columns if c.endswith('L')})

    lose_rename = {
        "WTeamID": "TeamIdB",
        "WScore": "ScoreB",
        "LTeamID": "TeamIdA",
        "LScore": "ScoreA",
    }
    lose_rename.update({c: c[:-1] + "B" for c in df.columns if c.endswith('W')})
    lose_rename.update({c: c[:-1] + "A" for c in df.columns if c.endswith('L')})

    win_df = df.copy()
    lose_df = df.copy()

    win_df = win_df.rename(columns=win_rename)
    lose_df = lose_df.rename(columns=lose_rename)

    return pd.concat([win_df, lose_df], 0, sort=False)


df = add_loosing_matches(df)

df.head()

cols_to_diff = [
    'Seed', 'WinRatio', 'GapAvg', '538rating'
]

for col in cols_to_diff:
    df[col + 'Diff'] = df[col + 'A'] - df[col + 'B']
df_test = pd.read_csv(DATA_PATH + "MSampleSubmissionStage1.csv")
df_test['Season'] = df_test['ID'].apply(lambda x: int(x.split('_')[0]))
df_test['TeamIdA'] = df_test['ID'].apply(lambda x: int(x.split('_')[1]))
df_test['TeamIdB'] = df_test['ID'].apply(lambda x: int(x.split('_')[2]))
df_test.head()
df_test = pd.merge(
    df_test,
    df_seeds,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedA'})

df_test = pd.merge(
    df_test,
    df_seeds,
    how='left',
    left_on=['Season', 'TeamIdB'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedB'})

df_test['SeedA'] = df_test['SeedA'].apply(treat_seed)
df_test['SeedB'] = df_test['SeedB'].apply(treat_seed)

df_test = pd.merge(
    df_test,
    df_features_season,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsA',
    'NumLosses': 'NumLossesA',
    'GapWins': 'GapWinsA',
    'GapLosses': 'GapLossesA',
    'WinRatio': 'WinRatioA',
    'GapAvg': 'GapAvgA',
}).drop(columns='TeamID', axis=1)

df_test = pd.merge(
    df_test,
    df_features_season,
    how='left',
    left_on=['Season', 'TeamIdB'],
    right_on=['Season', 'TeamID']
).rename(columns={
    'NumWins': 'NumWinsB',
    'NumLosses': 'NumLossesB',
    'GapWins': 'GapWinsB',
    'GapLosses': 'GapLossesB',
    'WinRatio': 'WinRatioB',
    'GapAvg': 'GapAvgB',
}).drop(columns='TeamID', axis=1)

df_test = pd.merge(
    df_test,
    df_538,
    how='left',
    left_on=['Season', 'TeamIdA'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'538rating': '538ratingA'})

df_test = pd.merge(
    df_test,
    df_538,
    how='left',
    left_on=['Season', 'TeamIdB'],
    right_on=['Season', 'TeamID']
).drop('TeamID', axis=1).rename(columns={'538rating': '538ratingB'})

for col in cols_to_diff:
    df_test[col + 'Diff'] = df_test[col + 'A'] - df_test[col + 'B']

df_test.head()

df['ScoreDiff'] = df['ScoreA'] - df['ScoreB']
df['WinA'] = (df['ScoreDiff'] > 0).astype(int)
df.drop(['538ratingA', '538ratingB','538ratingDiff'], axis=1, inplace=True)
features = df.columns.tolist()
features.remove('ScoreB')
features.remove( 'ScoreDiff')
features.remove( 'DayNum')
features.remove('ScoreA', )
features.remove( 'WinA')
print(features)


# def rescale(features, df_train, df_val, df_test=None):
#     min_ = df_train[features].min()
#     max_ = df_train[features].max()
#
#     df_train[features] = (df_train[features] - min_) / (max_ - min_)
#     df_val[features] = (df_val[features] - min_) / (max_ - min_)
#
#     if df_test is not None:
#         df_test[features] = (df_test[features] - min_) / (max_ - min_)
#
#     return df_train, df_val, df_test


def kfold(df, df_test_=None, plot=False, verbose=0, mode="reg"):
    seasons = df['Season'].unique()
    cvs = []
    pred_tests = []
    target = "ScoreDiff" if mode == "reg" else "WinA"

    for season in seasons[1:]:
        if verbose:
            print(f'\nValidating on season {season}')

        df_train = df[df['Season'] < season].reset_index(drop=True).copy()
        df_val = df[df['Season'] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy()

        # df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)

        if mode == "reg":
            model = ElasticNet(alpha=1, l1_ratio=0.5)
        else:
            model = LogisticRegression(C=100)

        model.fit(df_train[features], df_train[target])

        if mode == "reg":
            pred = model.predict(df_val[features])
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        else:
            pred = model.predict_proba(df_val[features])[:, 1]

        if df_test is not None:
            if mode == "reg":
                pred_test = model.predict(df_test[features])
                pred_test = (pred_test - pred_test.min()) / (pred_test.max() - pred_test.min())
            else:
                pred_test = model.predict_proba(df_test[features])[:, 1]

            pred_tests.append(pred_test)

        if plot:
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(pred, df_val['ScoreDiff'].values, s=5)
            plt.grid(True)
            plt.subplot(1, 2, 2)
            sns.histplot(pred)
            plt.show()

        loss = log_loss(df_val['WinA'].values, pred)
        cvs.append(loss)

        if verbose:
            print(f'\t -> Scored {loss:.3f}')

    print(f'\n Local CV is {np.mean(cvs):.3f}')

    return pred_tests


pred_tests = kfold(df, df_test, plot=False, verbose=1, mode="cls")
pred_test = np.mean(pred_tests, 0)

_ = sns.displot(pred_test)
sub = df_test[['ID', 'Season', 'Pred', 'TeamIdA', 'TeamIdB', 'SeedA', 'SeedB']].copy()
sub['Pred'] = pred_test
df_teams = pd.read_csv(DATA_PATH + "MTeams.csv")
sub = sub.merge(df_teams, left_on="TeamIdA", right_on="TeamID").drop('TeamID', axis=1).rename(columns={"TeamName": "TeamA"})
sub = sub.merge(df_teams, left_on="TeamIdB", right_on="TeamID").drop('TeamID', axis=1).rename(columns={"TeamName": "TeamB"})
df_seeds['Seed'] = df_seeds['Seed'].apply(lambda x:x[0])

sub = sub.merge(df_seeds, left_on=["TeamIdA", "Season"], right_on=["TeamID", "Season"]).drop('TeamID', axis=1).rename(columns={"Seed": "RegionA"})
sub = sub.merge(df_seeds, left_on=["TeamIdB", "Season"], right_on=["TeamID", "Season"]).drop('TeamID', axis=1).rename(columns={"Seed": "RegionB"})
print(sub.head(10))
best_teams = ['Stanford', 'South Carolina', 'Connecticut', 'Baylor', 'Maryland']  # considered for buff

strong_teams_safe = best_teams + ['NC State', 'Louisville']  # win 1st round
strong_teams_risky = strong_teams_safe + ['Texas A&M', 'Arizona', 'Georgia', 'UCLA']  # win 1st round

def overwrite_pred_risky(sub, eps=1e-5):
    new_sub = []

    for i, row in sub.iterrows():

        # Buff Stanford
        if row['TeamA'] == 'Stanford' and row['SeedB'] >= 3:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Stanford' and row['SeedA'] >= 3:
            row['Pred'] = eps

        # Buff South Carolina
        if row['TeamA'] == 'South Carolina' and row['SeedB'] >= 4:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'South Carolina' and row['SeedA'] >= 4:
            row['Pred'] = eps

        # Buff Connecticut
        if row['TeamA'] == 'Connecticut' and row['SeedB'] >= 4:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Connecticut' and row['SeedA'] >= 4:
            row['Pred'] = eps

        # Buff Baylor
        if row['TeamA'] == 'Baylor' and row['SeedB'] >= 3:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Baylor' and row['SeedA'] >= 3:
            row['Pred'] = eps

        # Buff Maryland
        if row['TeamA'] == 'Maryland' and row['SeedB'] >= 7:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Maryland' and row['SeedA'] >= 7:
            row['Pred'] = eps

        # Strong teams (risky) win their first round
        if row['TeamA'] in strong_teams_risky and row['SeedB'] >= 13:
            row['Pred'] = 1 - eps
        elif row['TeamB'] in strong_teams_risky and row['SeedA'] >= 13:
            row['Pred'] = eps

        new_sub.append(row)

    return pd.DataFrame(np.array(new_sub), columns=sub.columns)


def overwrite_pred_safe(sub, eps=1e-2):
    new_sub = []

    for i, row in sub.iterrows():
        row['Pred'] = np.clip(row['Pred'], 0.1, 0.9)  # clip for safety

        # Buff Stanford
        if row['TeamA'] == 'Stanford' and row['SeedB'] >= 6:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Stanford' and row['SeedA'] >= 6:
            row['Pred'] = eps

        # Buff South Carolina
        if row['TeamA'] == 'South Carolina' and row['SeedB'] >= 6:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'South Carolina' and row['SeedA'] >= 6:
            row['Pred'] = eps

        # Buff Connecticut
        if row['TeamA'] == 'Connecticut' and row['SeedB'] >= 6:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Connecticut' and row['SeedA'] >= 6:
            row['Pred'] = eps

        # Buff Baylor
        if row['TeamA'] == 'Baylor' and row['SeedB'] >= 7:
            row['Pred'] = 1 - eps
        elif row['TeamB'] == 'Baylor' and row['SeedA'] >= 7:
            row['Pred'] = eps

        # Strong teams (safe) win their first rounds
        if row['TeamA'] in strong_teams_safe and row['SeedB'] >= 13:
            row['Pred'] = 1 - eps
        elif row['TeamB'] in strong_teams_safe and row['SeedA'] >= 13:
            row['Pred'] = eps

        new_sub.append(row)

    return pd.DataFrame(np.array(new_sub), columns=sub.columns)


# sub_pp = overwrite_pred_safe(sub)
sub_pp = overwrite_pred_risky(sub)

sub_pp = sub

final_sub = sub_pp[['ID', 'Pred']].copy()
final_sub.to_csv('submission.csv', index=False)