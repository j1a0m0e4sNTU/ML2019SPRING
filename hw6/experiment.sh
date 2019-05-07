#python3 main.py train simple -epoch 30 -record records/test.txt -save ../../weights/test.pkl
#python3 main.py predict simple -load ../../weights/test.pkl -predict predictions/test.csv
#python3 main.py train simple -epoch 30 -record records/0506_2_new_word_model.txt -save ../../weights/0506_2.pkl
# python3 main.py train simple -seq_len 20 -record records/0507_1_seq_len_20.txt -save ../../weights/0507_1.pkl
# python3 main.py train simple -seq_len 40 -record records/0507_2_seq_len_40.txt -save ../../weights/0507_2.pkl
# python3 main.py train simple -record records/0507_0_baseline.txt 
# python3 main.py train A -record records/0507_3_A.txt -save ../../weights/0507_3.pkl
# python3 main.py train B -record records/0507_4_B.txt -save ../../weights/0507_4.pkl
# python3 main.py train simple -seq_len 60 -record records/0507_5_seq_len_60.txt -save ../../weights/0507_5.pkl
# python3 main.py train simple -seq_len 80 -record records/0507_6_seq_len_80.txt -save ../../weights/0507_6.pkl
# python3 main.py train C -record records/0507_7_C.txt -save ../../weights/0507_7.pkl
# python3 main.py train simple -epoch 10 -record records/0507_8_dropout.txt 
# python3 main.py train D -epoch 10 -record records/0507_9_D.txt -save ../../weights/0507_9.pkl
# python3 main.py train E -epoch 10 -record records/0507_10_E.txt -save ../../weights/0507_10.pkl
# python3 main.py train A -seq_len 40 -record records/0507_11_A_seq40.txt -save ../../weights/0507_11.pkl
# python3 main.py train B -seq_len 40 -record records/0507_12_B_seq40.txt -save ../../weights/0507_12.pkl
# python3 main.py train C -seq_len 40 -record records/0507_13_C_seq40.txt -save ../../weights/0507_13.pkl

# python3 main.py predict simple -seq_len 20 -load ../../weights/0507_2.pkl -predict predictions/0.csv
# python3 main.py predict simple -seq_len 40 -load ../../weights/0507_2.pkl -predict predictions/1.csv
# python3 main.py predict simple -seq_len 60 -load ../../weights/0507_2.pkl -predict predictions/2.csv
# python3 main.py predict simple -seq_len 80 -load ../../weights/0507_2.pkl -predict predictions/3.csv
# python3 main.py predict simple -seq_len 100 -load ../../weights/0507_2.pkl -predict predictions/4.csv
python3 main.py ensemble -predict ensemble.csv
