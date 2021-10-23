input_file=$1
output=$2
batch_size=8
fold_k=1
epoch=12000
fix=0
activation_function=sigmoid
y_object=normalized_experimental_mt
lr=1e-6
batch_norm=0
max_rt=24972 ## for SW480 data only 
hidden_layer=2

cnt=1
for coding_method in mz+DEN ##input features for FNN model
do
	for feature_size in 62
	do
		for dense_feature in 256
		do 
			for drop_rate in 0
			do
				python3 torch_fnn_kfold.py $1 $2_$cnt $coding_method $fold_k $batch_size $epoch $activation_function $dense_feature $y_object $lr $drop_rate $feature_size $max_rt
				echo "$cnt training complete"
                		cnt=$((cnt+1))
				echo hidden_layer=$hidden_layer, dropout_rate=$drop_rate,coding_method=$coding_method, batch_size=$batch_size,activation_function=$activation_function, predict_object=$y_object, learning_rate=$lr, epoch=$epoch, feature_size=$feature_size, max_rt=$max_rt
			done
		done
	done
done		





