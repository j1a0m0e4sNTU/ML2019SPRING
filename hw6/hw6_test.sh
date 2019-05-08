wget https://www.dropbox.com/s/dv8bxi4a91ozy23/word2vec_2.model
wget https://www.dropbox.com/s/mtnz0pa2yw3yxg0/0508_1.pkl
wget https://www.dropbox.com/s/aqd525bq17ljxvi/0508_3.pkl
wget https://www.dropbox.com/s/yot8rnfy083s9oy/0508_4.pkl
python3 main.py ensemble simple -test_x $1 -dict $2 -word_model word2vec_2.model -predict $3