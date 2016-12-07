###############################################################################################
#
# Script for training good word and phrase vector model using public corpora, version 1.0.
# The training time will be from several hours to about a day.
#
# Downloads about 8 billion words, makes phrases using two runs of word2phrase, trains
# a 500-dimensional vector model and evaluates it on word and phrase analogy tasks.
#
###############################################################################################

# This function will convert text to lowercase and remove special characters
normalize_text() {
  awk '{print tolower($0);}' | sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
  -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
  -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
  -e 's/«/ /g' | tr 0-9 " "
}

#mkdir paragraph2vec_l
cd paragraph2vec_l

#wget http://ai.stanford.edu/~amaas//data/sentiment/aclImdb_v1.tar.gz
#tar -xvf aclImdb_v1.tar.gz

#for i in `ls aclImdb/train/neg`; do
#    normalize_text < aclImdb/train/neg/$i >> data_p2v.txt
#done
#for i in `ls aclImdb/train/pos`; do
#    normalize_text < aclImdb/train/pos/$i >> data_p2v.txt
#done
#for i in `ls aclImdb/train/unsup`; do
#    normalize_text < aclImdb/train/unsup/$i >> data_p2v.txt
#done

cd ../

#python2 run.py
cd paragraph2vec_l
#for i in `ls aclImdb/test/neg`; do
#    normalize_text < aclImdb/test/neg/$i >> test_data_p2v.txt
#done

#for i in `ls aclImdb/test/pos`; do
#    normalize_text < aclImdb/test/pos/$i >> test_data_p2v.txt
#done
cd ../
python2 classification.py
