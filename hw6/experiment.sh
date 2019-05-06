#python3 main.py train simple -epoch 30 -record records/test.txt -save ../../weights/test.pkl
#python3 main.py predict simple -load ../../weights/test.pkl -predict predictions/test.csv
python3 main.py train simple -epoch 30 -record records/0506_1_new_word_model.txt -save ../../weights/0506_1.pkl
#python3 main.py predict simple -load ../../weights/0506_1.pkl -predict predictions/0506_1.csv
