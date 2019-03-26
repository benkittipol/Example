import pandas as pd
import numpy as np
import tensorflow.keras as K

# disable WARNING:
import warnings
#warnings.simplefilter("ignore")
pd.options.mode.chained_assignment = None


###############################
# EPL Funtion by Ben Kittipol #
###############################


# process data for predicting
def predict_data(home, away, df, predict=False, tor="default"):

    # match data
    match_df = df

    # home team stat
    home_df = match_df.loc[match_df["HomeTeam"] == home]
    home_df["Goal"] = home_df["FTHG"] # goal
    home_df["Con"] = home_df["FTAG"] # conceded
    home_df2 = match_df.loc[match_df["AwayTeam"] == home]
    home_df2["Goal"] = home_df2["FTAG"]
    home_df2["Con"] = home_df2["FTHG"]

    # away team stat
    away_df = match_df.loc[match_df["AwayTeam"] == away]
    away_df["Goal"] = away_df["FTAG"]
    away_df["Con"] = away_df["FTHG"]
    away_df2 = match_df.loc[match_df["HomeTeam"] == away]
    away_df2["Goal"] = away_df2["FTHG"]
    away_df2["Con"] = away_df2["FTAG"]

    # combine df
    h_all_df = pd.concat([home_df, home_df2]).sort_values(by="Date") # all match
    #print(h_all_df)

    a_all_df = pd.concat([away_df, away_df2]).sort_values(by="Date")
    #print(a_all_df)

    # calculate average goal and conceded last 5 match
    if predict:
        h_avg_g = home_df.iloc[-5:]["Goal"].values.sum() / 5
        h_avg_g2 = h_all_df.iloc[-8:]["Goal"].values.sum() / 8
        h_avg_c = home_df.iloc[-5:]["Con"].values.sum() / 5
        h_avg_c2 = h_all_df.iloc[-8:]["Con"].values.sum() / 8

        a_avg_g = away_df.iloc[-5:]["Goal"].values.sum() / 5
        a_avg_g2 = a_all_df.iloc[-8:]["Goal"].values.sum() / 8
        a_avg_c = away_df.iloc[-5:]["Con"].values.sum() / 5
        a_avg_c2 = a_all_df.iloc[-8:]["Con"].values.sum() / 8
    else:
        h_avg_g = home_df.iloc[-6:-1]["Goal"].values.sum() / 5
        h_avg_g2 = h_all_df.iloc[-9:-1]["Goal"].values.sum() / 8
        h_avg_c = home_df.iloc[-6:-1]["Con"].values.sum() / 5
        h_avg_c2 = h_all_df.iloc[-9:-1]["Con"].values.sum() / 8

        a_avg_g = away_df.iloc[-6:-1]["Goal"].values.sum() / 5
        a_avg_g2 = a_all_df.iloc[-9:-1]["Goal"].values.sum() / 8
        a_avg_c = away_df.iloc[-6:-1]["Con"].values.sum() / 5
        a_avg_c2 = a_all_df.iloc[-9:-1]["Con"].values.sum() / 8

        if len(a_all_df.iloc[-9:-1]) < 8:
            x_data = np.array(["con"])
            return x_data



    #print(h_avg_g, "{0:.1f}".format(h_avg_g2), "-", h_avg_c, h_avg_c2)
    #print(a_avg_g, a_avg_g2, "-", a_avg_c, a_avg_c2)

    x_data = [h_avg_g, h_avg_g2, h_avg_c, h_avg_c2, a_avg_g, a_avg_g2,
                a_avg_c, a_avg_c2]

    if tor != "default":
        raka = tor
        x_data.append(raka)

    x_arr = np.array(x_data)

    # normalize to scale 0-1
    x_arr /= 10

    return x_arr



# process data for training
def train_data():
    match_df = pd.read_csv("data/epl.csv")

    ndf = len(match_df)
    features = []
    labels = []
    for i in range(130,ndf):
        home = match_df.iloc[i]["HomeTeam"]
        away = match_df.iloc[i]["AwayTeam"]

        # x data aka features
        x_data = predict_data(home, away, match_df.iloc[:i])
        if len(x_data) == 1:
            #print("Skip")
            continue
        features.append(x_data)

        # y data aka labels
        m0 = match_df.iloc[i]["HMargin"] <= -3
        m1 = match_df.iloc[i]["HMargin"] == -2
        m2 = match_df.iloc[i]["HMargin"] == -1
        m3 = match_df.iloc[i]["HMargin"] == 0
        m4 = match_df.iloc[i]["HMargin"] == 1
        m5 = match_df.iloc[i]["HMargin"] == 2
        m6 = match_df.iloc[i]["HMargin"] >= 3

        y_data = np.array([m0,m1,m2,m3,m4,m5,m6])
        labels.append(y_data)


    # make it to array
    features = np.array(features)
    labels = np.array(labels)

    return features, labels


# check แต้มต่อ
def chk_bet(tt, pred):
    tt = float(tt)
    tt = tt + 0.001 if tt >= 0 else tt - 0.001
    temtor = int(round(-tt))
    argtt = temtor + 3 if temtor >= 0 else temtor + 4
    if argtt > 6: argtt = 6
    if argtt < 1: argtt = 1
    result = sum(pred[argtt:])

    return result
