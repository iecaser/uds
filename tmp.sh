initial_size=100
batch_size=1000
iterations=5
idx=0
for idx in {0..9}
do
python3 main.py $idx "mnist" $batch_size $initial_size $iterations "DualDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/DualDensity/" --gpu 4
done
