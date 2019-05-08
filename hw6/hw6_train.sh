wget https://www.dropbox.com/s/dv8bxi4a91ozy23/word2vec_2.model
python3 main.py train simple -train_x $1 -train_y $2 -dict $4 -word_model word2vec_2.model -save train.pkl
python3 main.py predict simple -word_model word2vec_2.model -load train.pkl -test_x $3 -predict ans.csv