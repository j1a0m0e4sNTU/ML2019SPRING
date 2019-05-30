# python3 main.py train -save ../../weights/0529_test.pkl -record records/0529_test.txt
# python3 main.py train -conv A -fc A -save ../../weights/0529_A_A.pkl -record records/0529_A_A.txt
# python3 main.py train -conv B -fc A -lr 1e-3 -save ../../weights/0529_B_A.pkl -record records/0529_B_A.txt
# python3 main.py train -conv B -fc B -lr 1e-3 -save ../../weights/0529_B_B.pkl -record records/0529_B_B.txt
python3 main.py train -conv B -fc A -lr 1e-3 -save ../../weights/0530_B_A.pkl -record records/0530_B_A.txt -info "without rotation & scaling"
