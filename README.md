# Real-time face detection and emotion recognition
Real time face detection and emotion recognition

This source code is highly encouraged by [`Deepface`](https://github.com/serengil/deepface).

For face detection, [`SSD`](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/) is used.
For facial expression recognition, [`emotion`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) is used.

The execution speed (face detection + emotion recognition) is approx 0.06 sec per frame. To measure the execution speed, we basically assumed that only one face is detected. However, this source code also can work with multiple results of face detection.

This source code have been implimented based on Ubuntu 20.04 (64bit)


Running environment is as follows:
```
GPU: Geforce RTX 2080ti
Driver: 460.73.01
CUDA version: 11.2
```

## How to prepare.
Install custom library of Deepface.
```
cd ./libs/deepface
python setup.py install
```

## How to run demo
```
python main.py
```
*it is possible that an extra library needs to be installed to run this source code.

## Citation

Please cite [deepface](https://ieeexplore.ieee.org/document/9259802) and [DDRL](https://ieeexplore.ieee.org/abstract/document/8451494) in your publications if it helps your research. Here is an example BibTeX entry:

```BibTeX
@inproceedings{serengil2020lightface,
  title={LightFace: A Hybrid Deep Face Recognition Framework},
  author={Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages={23-27},
  year={2020},
  doi={10.1109/ASYU50717.2020.9259802},
  organization={IEEE}
}
```

```BibTeX
@inproceedings{yu2018deep,
  title={Deep discriminative representation learning for face verification and person re-identification on unconstrained condition},
  author={Yu, Jongmin and Ko, Donghwuy and Moon, Hangyul and Jeon, Moongu},
  booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
  pages={1658--1662},
  year={2018},
  organization={IEEE}
}
```
