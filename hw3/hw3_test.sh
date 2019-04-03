wget https://www.dropbox.com/s/gmnw9m8e9emkvwr/vgg_cc_2.pkl
python3 main.py predict -dataset $1 -load vgg_cc_2.pkl -tencrop 1 -csv $2