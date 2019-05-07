#python3 main.py train simple -epoch 30 -record records/test.txt -save ../../weights/test.pkl
#python3 main.py predict simple -load ../../weights/test.pkl -predict predictions/test.csv
#python3 main.py train simple -epoch 30 -record records/0506_2_new_word_model.txt -save ../../weights/0506_2.pkl
# python3 main.py train simple -seq_len 20 -record records/0507_1_seq_len_20.txt -save ../../weights/0507_1.pkl
# python3 main.py train simple -seq_len 40 -record records/0507_2_seq_len_40.txt -save ../../weights/0507_2.pkl
# python3 main.py train simple -record records/0507_0_baseline.txt 
# python3 main.py train A -record records/0507_3_A.txt -save ../../weights/0507_3.pkl
# python3 main.py train B -record records/0507_4_B.txt -save ../../weights/0507_4.pkl
python3 main.py train simple -seq_len 60 -record records/0507_5_seq_len_60.txt -save ../../weights/0507_5.pkl
python3 main.py train simple -seq_len 80 -record records/0507_6_seq_len_80.txt -save ../../weights/0507_6.pkl
python3 main.py train C -record records/0507_7_C.txt -save ../../weights/0507_7.pkl