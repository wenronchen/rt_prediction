input_file=$1
output=$2
fold_k=1
batch_size=8
epoch=12000
activation_function=sigmoid
y_object='RT norm'
lr=1e-6
max_rt=1 
hidden_layer=3

cnt=1
for coding_method in elude_custom
do
	for feature_size in 62
	do
		for dense_feature in 128
		do 
			for drop_rate in 0
			do
				python3 torch_fnn_kfold.py $1 $2_$cnt $coding_method $fold_k $batch_size $epoch $activation_function $dense_feature "$y_object" $lr  $drop_rate $feature_size $max_rt
				echo "$cnt training complete"
                		cnt=$((cnt+1))
				echo hidden_layer=$hidden_layer, dropout_rate=$drop_rate,coding_method=$coding_method, batch_size=$batch_size,activation_function=$activation_function, predict_object="$y_object", learning_rate=$lr, epoch=$epoch, dense_feature=$dense_feature, feature_size=$feature_size, max_rt=$max_rt
			done
		done
	done
done		





