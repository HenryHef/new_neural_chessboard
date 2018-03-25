This project is based on Maciej Czyzewski's Neural Chessboard program found at 
https://github.com/maciejczyzewski/neural-chessboard

__Dependencies Installation (macOS):__
```
$ brew install opencv3               # toolkit for computer vision
$ pip3 install -r requirements.txt   # toolkit for machine learning
```

on linux:
https://www.learnopencv.com/install-opencv3-on-ubuntu/

__Dataset & Training:__
```
$ python3 dataset.py
$ python3 train.py 50
```
The detect_img(im) method in detector.py takes ina image loded throguh opencv and returns the transform from world to board space. Note that thoguh it also returns a cropped image, you would be better served to recrop the image from the original, using the transform as there will be slightly less bluring.

__Dependencies:__

- [Python 3](https://www.python.org/downloads/)
- [Scipy 0.19.1](https://www.scipy.org/)
- [OpenCV 3](http://opencv.org/)
- [Tensorflow](https://www.tensorflow.org/) (with [tflearn](https://github.com/tflearn/tflearn) support)
- [Pyclipper](https://github.com/greginvm/pyclipper)
