# Ai learn to play cubefield

This is example of 1-hour trained model that learned to play cubefield game on browser. This model is just example and will not learn after playing.

## requirement
- tensorflow (I used tensorflow-gpu 1.10.0)
- numpy
- cv2
- mss
- Pillow
- pyautogui

## Instructions
- Open your browser and enter game url -> http://thesimplearcade.com/play/cubefield.html
- Move your browser to the leftmost of your screen. The default position of top-left of flash element is 190 pixels from top and 5 pixels from left.
- If you want to change the flash element screen position, change line 14 to fit your preference.
- To see the captured screen for adjusting browser window, change `adjust_scrn = False` to `adjust_scrn = True` in line 20.
  - run `play_ac.py` file. The new small windows of processed captured screen will pop up. This window show what AI see.
  - move your browser screen to fit captured position or change your captured position to fit your screen.
  - the image in this window should be similar to this image below.
    
    ![1559a639071988af44e5d335de5a8fdf.jpg](https://www.img.in.th/images/1559a639071988af44e5d335de5a8fdf.jpg)
  - If you finish adjusting your screen, reverse it back to `adjust_scrn = False`.
 - Click new game on game screen and let it die 1 time.
 - Run `play_ac.py` and immediately click on flash element on browser. **BE CAREFUL, AI will press left or right arrow key. If you are not on game browser window, it might doom your PC**.
   - To quit, click on small window that show what AI see and press 'q'
 - Enjoys!!
  
