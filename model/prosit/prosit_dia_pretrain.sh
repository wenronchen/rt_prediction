max_seq=200 ##max_seq=153 for SW480 CZE 
max_rt=11040 ##14400 #10800 ##max_rt=0.07 for SW480 CZE
min_rt=-3600
dropout_rate=0.5
dense_dropout=0.1
pretrained=0
pretrain_path=none
y_attr=RT

cnt=1
for GRU_unit in 512
do
	for embed_out in 24
	do 
		for dense_feature in 256
		do
			python3 bigru_model_kfold.py $1 $2_$cnt $max_seq $max_rt $min_rt $GRU_unit $embed_out $dense_feature $dropout_rate $dense_dropout $y_attr $pretrained $pretrain_path
			echo "$cnt training complete"
			cnt=$((cnt+1))
			echo Embed_out_dim=$embed_out, GRU_unit=$GRU_unit, Dense_Feature=$dense_feature, dropout_rate=$dropout_rate, dense_dropout=$dense_dropout, max_seq=$max_seq, max_rt=$max_rt, min_rt=$min_rt, pretrained=$pretrained, pretrain_path=$pretrain_path
		done
	done
done



