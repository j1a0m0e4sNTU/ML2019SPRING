# python3 main.py train -epoch 50 -save ../../weights/01.pkl
# python3 main.py train -epoch 50 -save ../../weights/vgg_b.pkl
# python3 main.py train -epoch 50 -save ../../weights/vgg_c.pkl
# python3 main.py train -epoch 50 -save ../../weights/vgg_c_2.pkl
# python3 main.py train -epoch 50 -save ../../weights/vgg_c_3.pkl 
# transforamtion: (horizontal-flip, degree:10, translate:(0.1, 0.1)), validation acc: 0.6688 after 50 epochs
# python3 main.py train -epoch 50 -save ../../weights/vgg_c_4.pkl -record results/vgg_c_4.csv 
# transforamtion: (horizontal-flip, degree:10, translate:(0.1, 0.1), scale:(0.9, 1.1))
# python3 main.py train -epoch 50 -save ../../weights/vgg_c_5.pkl -record results/vgg_c_5.csv 
# transformation :flip degrees= 20, translate= (0.2, 0.2), scale= (0.8, 1.2)
# python3 main.py train -epoch 50 -save ../../weights/vgg_c_c.pkl -record results/vgg_c_c.csv 
# python3 main.py predict -dataset ../../data_hw3/test.csv -load ../../weights/vgg_c_c.pkl -tencrop 1 -csv vgg_c_c.csv
# python3 main.py train -epoch 60 -save ../../weights/vgg_cc_2.pkl -record results/vgg_cc_2.csv 
# python3 main.py predict -dataset ../../data_hw3/test.csv -load ../../weights/vgg_cc_2.pkl -tencrop 1 -csv vgg_cc_2.csv
python3 main.py train -epoch 60 -save ../../weights/dnn.pkl -record results/dnn.csv 
# python3 main.py predict -dataset ../../data_hw3/test.csv -load ../../weights/dnn.pkl -tencrop 1 -csv predictions/dnn.csv