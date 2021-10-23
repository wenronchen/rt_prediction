#$1 is the file path for input
#$2 is the number of replicate
##The charge count and experimental mobility should be calculated firstly, after the normalization, the normalized experimental mobility should be converted to migration time. 


codebase=~/rt/code


for i in f1 f2 f3 f4 f5 f6
do
	python3  ${codebase}/model/empirical/train_empirical.py $1/${i}.tsv $1/${i}_fixed.tsv
done

cat $1/f1_norm.tsv > $1/combined_norm.tsv
for i in 2 3 4 5 6
do
	grep "480_F${i}_0$2" $1/f${i}_norm.tsv >> $1/combined_norm.tsv
done 

cat $1/f1_fixed_norm.tsv > $1/combined_fixed_norm.tsv
for i in 2 3 4 5 6
do
	grep "480_F${i}_0$2" $1/f${i}_fixed_norm.tsv >> $1/combined_fixed_norm.tsv
done

python3  ${codebase}/model/empirical/train_empirical.py $1/combined_norm.tsv $1/combined_fixed_norm.tsv


