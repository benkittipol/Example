import tensorflow.keras as K
from tensorflow import set_random_seed
import pandas as pd
import numpy as np
import os, io

import epl_func as ef

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable warning

# set random seed
np.random.seed(456)
set_random_seed(123)

all_team = ["Arsenal", "Bournemouth", "Brighton", "Burnley", "Cardiff", "Chelsea",
            "Crystal Palace", "Everton", "Fulham", "Huddersfield", "Leicester",
            "Liverpool", "Man City", "Man United", "Newcastle", "Southampton",
            "Tottenham", "Watford", "West Ham", "Wolves"]

while True:
    # print instruction
    print("Input team number")
    for i in range(20):
        print(i, ":", all_team[i])

    # get home and away team
    print()
    homeid = input("Home Team: ")
    awayid = input("Away Team: ")

    # assign team id
    htm = all_team[int(homeid)]
    atm = all_team[int(awayid)]
    print(htm, "-", atm)

    # data
    epl_path = os.path.join("data","epl.csv")
    epl_df = pd.read_csv(epl_path)

    # load_model
    model_path = os.path.join("model","epl.h5")
    try:
        model = K.models.load_model(model_path)
    except:
        print("Error loading Model")
        raise SystemExit

    # features
    features = ef.predict_data(home=htm,away=atm,df=epl_df,predict=True)
    features = features.reshape(-1,8)
    pred_result = model.predict(features)

    # print result
    plus = ["+","","","","","","+"]
    for i in range(7):
        print(htm, "chance to end match with goal margin", f"{-3+i}{plus[i]}",
                ": {0:.2f}%".format(pred_result[0][i]*100))

    while True:
        tor = input("\nTor/Handicap: ")
        try: tor = float(tor);break
        except:print("Please input only number")

    # bet home probability
    bet_h_prob = ef.chk_bet(tor, pred_result[0])
    print("Home Prob: {0:.3f}%".format(bet_h_prob*100))

    print("\npress ENTER to continue or type 'quit' to quit")
    qq = input()
    if qq.lower() in ["yes", "quit", "q", "exit"]:break
