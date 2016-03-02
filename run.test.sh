# Sift RBF
#g=(0.005 0.005 0.01)
#for k in {1..3}; do
#    for i in {1..3};do
#        ./scripts/train_svm.py P00${i} ${k} siftbow_features/all_avg.vectors 200 sift_pred/P00${i}_avg_${k}.model -g ${g[i-1]} -k rbf 
#        ./scripts/test_svm.py sift_pred/P00${i}_avg_${k}.model ${k} siftbow_features/all_avg.vectors 200 sift_pred/P00${i}_avg_${k}.pred
#        ../hw1/mAP/ap list/P00${i}_test_${k} sift_pred/P00${i}_avg_${k}.pred >> log
#    done;
#done

# CNN RBF

#g=(0.001 0.0005 0.001)
#for k in {1..3};do
#for i in {1..3};do
#./scripts/train_svm.py P00${i} ${k} cnn_fc7_features/all_avg.vectors 4096 cnn_pred/fc7_P00${i}_${k}.model -k rbf -g ${g[i-1]} 
#./scripts/test_svm.py cnn_pred/fc7_P00${i}_${k}.model ${k} cnn_fc7_features/all_avg.vectors 4096 cnn_pred/fc7_P00${i}_${k}.pred
#../hw1/mAP/ap list/P00${i}_test_${k} cnn_pred/fc7_P00${i}_${k}.pred >> log2
#done;
#done

# Imtraj linear 
for k in {1..3};do
for i in {1..3}; do
./scripts/train_svm.py P00${i} ${k} imtraj 32748 imtraj_pred/P00${i}_${k}.model -f imtraj 
./scripts/test_svm.py imtraj_pred/P00${i}_${k}.model ${k} imtraj 32748 imtraj_pred/P00${i}_${k}.pred -f imtraj
../hw1/mAP/ap list/P00${i}_test_${k} imtraj_pred/P00${i}_${k}.pred >> log
done;
done
