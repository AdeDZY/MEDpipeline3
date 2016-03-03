# cross validating early fusion
g=(0.0001 0.0001 0.0001)
for k in {1..3};do
for i in {1..3};do
./scripts/train_svm.py P00${i} ${k} all_early.vectors 200 early_pred/P00${i}_${k}.model -k rbf -g ${g[i-1]} 
./scripts/test_svm.py early_pred/P00${i}_${k}.model ${k} all_early.vectors 200 early_pred/P00${i}_${k}.pred
../hw1/mAP/ap list/P00${i}_test_${k} early_pred/P00${i}_${k}.pred >> log
done;
done

# predict test videos
g=(0.0001 0.0001 0.0001)
for i in {1..3};do
./scripts/train_svm.py P00${i} 0 all_early.vectors 200 early_pred/P00${i}.model -k rbf -g ${g[i-1]} 
./scripts/test_svm.py early_pred/P00${i}.model 0 all_early.vectors 200 early_pred/P00${i}.pred
done;
