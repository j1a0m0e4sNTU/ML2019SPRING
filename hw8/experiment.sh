# python3 main.py train -save ../../weights/0529_test.pkl -record records/0529_test.txt
# python3 main.py train -conv A -fc A -save ../../weights/0529_A_A.pkl -record records/0529_A_A.txt
# python3 main.py train -conv B -fc A -lr 1e-3 -save ../../weights/0529_B_A.pkl -record records/0529_B_A.txt
# python3 main.py train -conv B -fc B -lr 1e-3 -save ../../weights/0529_B_B.pkl -record records/0529_B_B.txt
# python3 main.py train -conv B -fc A -lr 1e-3 -save ../../weights/0530_B_A.pkl -record records/0530_B_A.txt -info "without rotation & scaling"
# python3 main.py train -conv C -fc C -lr 1e-3 -save ../../weights/0530_C_C.pkl -record records/0530_C_C.txt -info "Parameter num: 50339"
# python3 main.py train -conv C -fc C -lr 1e-3 -save ../../weights/0530_C_C_2.pkl -record records/0530_C_C_2.txt -info "use nn.Conv2d in first layer, parameter num: 50455"
# python3 main.py train -conv C -fc C2 -lr 1e-3 -save ../../weights/0530_C_C2.pkl -record records/0530_C_C2.txt -info "parameter num: 63511"
python3 main.py train -conv D -fc C -lr 1e-3 -save ../../weights/0530_D_C.pkl -record records/0530_D_C.txt -info "parameter num: 106791"