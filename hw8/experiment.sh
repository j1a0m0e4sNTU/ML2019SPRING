# python3 main.py train -save ../../weights/0529_test.pkl -record records/0529_test.txt
# python3 main.py train -conv A -fc A -save ../../weights/0529_A_A.pkl -record records/0529_A_A.txt
python3 main.py train -conv B -fc A -save ../../weights/0529_B_A.pkl -record records/0529_B_A.txt
python3 main.py train -conv B -fc B -save ../../weights/0529_B_B.pkl -record records/0529_B_B.txt