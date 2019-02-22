initial_size=100
batch_size=100
iterations=20
idx=0
for idx in {0..9}
do
python3 main.py $idx "dd" $batch_size $initial_size $iterations "DualDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/DualDensity/" --gpu 1
done
