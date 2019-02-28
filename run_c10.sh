initial_size=5000
batch_size=5000
iterations=4
dataset=cifar10
visible='2,3'
for idx in {0..9}
do
    echo "------- Random $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "Random" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/Random/" --visible $visible

    echo "------- Uncertainty $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "Uncertainty" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/Uncertainty/" --visible $visible

    echo "------- CoreSet $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "CoreSet" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/CoreSet/" --visible $visible

    echo "------- UncertaintyDensity $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "UncertaintyDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/UncertaintyDensity/" --visible $visible

    echo "------- DualDensity $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "DualDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/DualDensity/" --visible $visible
done
python3 plot.py --dataset $dataset --init $initial_size --batch $batch_size
