#python3 main.py train simple -epoch 30 -record records/test.txt -save ../../weights/test.pkl
python3 main.py predict simple -load ../../weights/test.pkl -predict predictions/test.csv
