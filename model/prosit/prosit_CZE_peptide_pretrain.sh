max_seq=200 
max_mt=1
min_mt=0
dropout_rate=0.5
dense_dropout=0.1
pretrained=0
pretrain_path=none
y_attr='MT norm'

cnt=1
for GRU_unit in 128
do
	for embed_out in 24  
	do 
		for dense_feature in 256
		do
			python3 bigru_model_kfold.py $1 $2_$cnt $max_seq $max_mt $min_mt $GRU_unit $embed_out $dense_feature $dropout_rate $dense_dropout "$y_attr" $pretrained $pretrain_path
			echo "$cnt training complete"
			cnt=$((cnt+1))
			echo Embed_out_dim=$embed_out, GRU_unit=$GRU_unit, Dense_Feature=$dense_feature, dropout_rate=$dropout_rate, dense_dropout=$dense_dropout, max_seq=$max_seq, max_mt=$max_mt, min_mt=$min_mt, pretrained=$pretrained, pretrain_path=$pretrain_path
		done
	done
done



