#python3 main.py train simple -epoch 30 -record records/test.txt -save ../../weights/test.pkl
#python3 main.py predict simple -load ../../weights/test.pkl -predict predictions/test.csv
#python3 main.py train simple -epoch 30 -record records/0506_2_new_word_model.txt -save ../../weights/0506_2.pkl
python3 main.py train simple -seq_len 20 -record records/0507_1_seq_len_20.txt -save ../../weights/0507_1.pkl
python3 main.py train simple -seq_len 40 -record records/0507_2_seq_len_40.txt -save ../../weights/0507_2.pkl
