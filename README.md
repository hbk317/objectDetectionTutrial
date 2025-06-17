# objectDetectionTutrial 
2025/06/19, 26日に行う物体検出モデルチュートリアルのサンプルコード

## セットアップ方法
```
git clone https://github.com/hbk317/objectDetectionTutrial.git
cd objectDetectionTutrial
pipenv --python 3.11.3
pipenv shell
pipenv install lxml==5.4.0 PyQt5==5.15.11 torch==2.2.0 ultralytics==8.1.0 numpy==1.26.0 mediapipe==0.10.21 retina-face==0.0.17 tf-keras==2.19.0 lapx==0.5.11
```
※私はpyenv + pipenvでpython環境を構築しているので上記のコマンドです．それ以外を利用されている場合は，お手数ですが仮想環境の立ち上げ方，ライブラリのインストール等を適宜読み替えてください．

## mp4ファイルと教師データ，google colab上で動作するソースコードの共有
下記のGoogleドライブで共有しておりますので適宜ご利用ください．data/はダウンロードした後，リポジトリ内に置いてください（空のディレクトリを用意してありますので，それを置換するように置いていただければ問題無いです）．colab_notobook/中のファイルについては，pythonのローカル環境をお持ちでない方向けに，google colab上でyoloの推論を行うためのコードを記載してあります．

[Googleドライブのリンク](https://drive.google.com/drive/folders/1aOykHCs_N18W5DKGXuzZRaAzknChPKiA?usp=sharing)．

## アノテーションソフトlabelImgの実行
```
python labelImg/labelImg.py
```
を実行する事で，labelImgのウィンドウが表示されます．詳細はチュートリアル中に説明いたします．

## YOLOをリアルタイムで実行
```
python yolo_realtime.py
```
を実行すると，PCのカメラを使用してリアルタイムでYOLOの推論を行います．終了する際はカメラ映像が表示されているウィンドウを選択した状態で'q'キーを入力してください．事前学習時点でのモデルを利用しているため，検出可能な物体の一覧は以下となります[（参考）](https://docs.ultralytics.com/ja/datasets/detect/coco8/#dataset-yaml)．
```
0: person
1: bicycle
2: car
3: motorcycle
4: airplane
5: bus
6: train
7: truck
8: boat
9: traffic light
10: fire hydrant
11: stop sign
12: parking meter
13: bench
14: bird
15: cat
16: dog
17: horse
18: sheep
19: cow
20: elephant
21: bear
22: zebra
23: giraffe
24: backpack
25: umbrella
26: handbag
27: tie
28: suitcase
29: frisbee
30: skis
31: snowboard
32: sports ball
33: kite
34: baseball bat
35: baseball glove
36: skateboard
37: surfboard
38: tennis racket
39: bottle
40: wine glass
41: cup
42: fork
43: knife
44: spoon
45: bowl
46: banana
47: apple
48: sandwich
49: orange
50: broccoli
51: carrot
52: hot dog
53: pizza
54: donut
55: cake
56: chair
57: couch
58: potted plant
59: bed
60: dining table
61: toilet
62: tv
63: laptop
64: mouse
65: remote
66: keyboard
67: cell phone
68: microwave
69: oven
70: toaster
71: sink
72: refrigerator
73: book
74: clock
75: vase
76: scissors
77: teddy bear
78: hair drier
79: toothbrush
```
