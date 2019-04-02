# python3 main.py train -epoch 50 -save ../../weights/01.pkl
# python3 main.py train -epoch 50 -save ../../weights/vgg_b.pkl
# python3 main.py train -epoch 50 -save ../../weights/vgg_c.pkl
# python3 main.py train -epoch 50 -save ../../weights/vgg_c_2.pkl
# python3 main.py train -epoch 50 -save ../../weights/vgg_c_3.pkl 
# transforamtion: (horizontal-flip, degree:10, translate:(0.1, 0.1)), validation acc: 0.6688 after 50 epochs
python3 main.py train -epoch 50 -save ../../weights/vgg_c_4.pkl -record results/vgg_c_4.csv 
# transforamtion: (horizontal-flip, degree:10, translate:(0.1, 0.1), scale:(0.9, 1.1))