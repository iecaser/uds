initial_size=1000
batch_size=1000
iterations=30
dataset="cifar10"
visible="0,1"
exp="exp.1000.1000.min"
mkdir $exp
mkdir $exp/Random $exp/Uncertainty $exp/CoreSet $exp/UncertaintyDensity $exp/DualDensity
for idx in {0..5}
do
    echo "------- Random $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "Random" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/Random/" --visible $visible

    echo "------- Uncertainty $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "Uncertainty" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/Uncertainty/" --visible $visible

    echo "------- CoreSet $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "CoreSet" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/CoreSet/" --visible $visible

    echo "------- UncertaintyDensity $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "UncertaintyDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/UncertaintyDensity/" --visible $visible

    echo "------- DualDensity $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "DualDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/DualDensity/" --visible $visible

    echo "------ ploting ...."
    python3 plot.py --idx $idx --exp $exp --dataset $dataset --init $initial_size --batch $batch_size
done
python3 plot.py --exp $exp --dataset $dataset --init $initial_size --batch $batch_size
