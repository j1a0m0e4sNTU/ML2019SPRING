# python3 main.py train -id test -save ../../weights/test.pkl
# python3 main.py train -E A -D A -id 0514_A_A -save ../../weights/0514_A_A.pkl
# python3 main.py train -E B -D B -id 0514_B_B -save ../../weights/0514_B_B.pkl
# python3 main.py train -E C -D C -id 0514_C_C -save ../../weights/0514_C_C.pkl
# python3 main.py train -E D -D D -id 0514_D_D -save ../../weights/0514_D_D.pkl
# python3 main.py cluster -E D -D D -load ../../weights/0514_D_D.pkl -csv test.csv
# python3 main.py cluster -cluster_num 4 -csv test4.csv
# python3 main.py train -id simple_all -save ../../weights/simple_all.pkl
# python3 main.py cluster -load ../../weights/simple_all.pkl 
# python3 main.py train -E D -D D -id DD_all -save ../../weights/DD_all.pkl
# python3 main.py cluster -E D -D D -load ../../weights/DD_all.pkl 
# python3 main.py test -E A -D A -load ../../weights/0514_A_A.pkl
python3 main.py cluster -load ../../weights/simple_all.pkl -csv check.csv