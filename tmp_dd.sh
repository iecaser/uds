initial_size=1000
batch_size=3000
iterations=4
idx=0
dataset="dd"
echo "------- DualDensity $idx... -------"
python3 main.py $idx $dataset $batch_size $initial_size $iterations "DualDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/DualDensity/" --gpu 4
