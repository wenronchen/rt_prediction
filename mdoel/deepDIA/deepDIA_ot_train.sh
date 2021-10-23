max_seq=200
max_rt=10800
min_rt=0
dropout_rate=0.5
y_attr='Feature apex'


cnt=1
for dense_feature in 128
do
	for lstm_feature in 64
	do
		for filter_size in 64
		do
			for kernel_size in 4
			do
				python3 train_model_split.py $1 $2_$cnt $max_seq $max_rt $min_rt $filter_size $kernel_size $lstm_feature $dropout_rate $dense_feature "$y_attr"
				echo "$cnt training complete"
				cnt=$((cnt+1))
				echo filter_size=$filter_size, kernel_size=$kernel_size, LSTM_Feature=$lstm_feature, dropout_rate=$dropout_rate, dense_feature=$dense_feature, max_seq=$max_seq, max_rt=$max_rt
			done
		done
	done
done

