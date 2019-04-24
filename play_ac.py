import numpy as np
import cv2
import os
import time
from mss import mss
from PIL import Image
import pyautogui as auto

import s_function as sfn
#np.set_printoptions(threshold=np.inf) # print full numpy array

# top-left screen position for web -> http://thesimplearcade.com/play/cubefield.html
# change scrnPos to fit your screen
scrnPos = {'top': 190, 'left': 5, 'width': 720, 'height': 520} # -> change this
sct = mss()


# save / load model
loadmodel = True
adjust_scrn = False
actor_model_path = os.path.join("model", "actor.h5")


# parameter
auto.PAUSE = 0.005
action = {0:"right", 1:"left"}
#action = {0:"right", 1:"left", 2:"up"} # up is doing nothing
nAct = len(action) # number of action


# variable for collect data
notAct = True
gen = 0
lastAct = 0
freeze = 0
total_reward = 0


# create actor critic model
actor_model = sfn.create_cnn_actor(nAct)

# load model
if loadmodel:
    try:
        actor_model.load_weights(actor_model_path)
        print("Model loaded")
    except:
        print("Error Loading Model, Create new one")


# test model
xTest = np.zeros((1,100,180,1))
prob = actor_model.predict(xTest)
del xTest

# play
while(True):
    if gen == 1000:break

    # capture screen
    sct_img = sct.grab(scrnPos)
    color_img = np.array(Image.frombytes('RGB', sct_img.size, sct_img.rgb))

    # display picture(video)
    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY) # convert to gray
    resizeimg = cv2.resize(img,(180,130)) # Width x Height
    resizeimg[resizeimg == 149] = [110] # change yellow-อ่อน box color
    _,thresh = cv2.threshold(resizeimg,130,255,cv2.THRESH_BINARY)
    cv2.imshow('Window', thresh)

    # check score
    scoreScrn = resizeimg[:15,:25] # height * width

    # if meet condition -> break
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    # action
    if notAct:
        notAct = False
        time.sleep(2)
        #auto.click(x=175, y=400)
        auto.press("space")
    else:
        xFeature = thresh / 255 # normalize -> scale to 0-1
        xFeature = xFeature[30:,:] # cut some sky, don't need this info
        xFeature = xFeature.reshape(-1,100,180,1) # height * width
        prob = actor_model.predict(xFeature)
        act = np.random.choice(np.arange(nAct),p=prob[0])
        #act = np.argmax(prob[0])

        # print action
        #print(action[act])

        # press keyboard
        if not adjust_scrn:
            try: auto.keyUp(action[lastAct])
            except: pass
            auto.keyDown(action[act])
        lastAct = act

        # check freeze
        try:
            freezeScrn = np.array_equal(lastScore, scoreScrn)
        except:
            freezeScrn = False

        if freezeScrn:
            freeze += 1
        else:
            freeze = 0

        #collect last score screen
        lastScore = scoreScrn

        total_reward += 1

        if freeze >= 10:

            # release key
            try: auto.keyUp(action[lastAct])
            except: pass

            # print stats
            gen += 1
            print("Gen:", gen, "- Reward:", total_reward)

            # reset and start new game
            time.sleep(2)
            for _ in range(2):
                auto.press("space")
            freeze = 0
            total_reward = 0


# try to release key
try: auto.keyUp(action[lastAct])
except: pass
