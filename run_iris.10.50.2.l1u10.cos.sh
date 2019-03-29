initial_size=10
batch_size=50
iterations=2
visible="0"
dataset="iris"
exp="iris.${initial_size}.${batch_size}.${iterations}.l1u10.cos"
mkdir $exp
mkdir $exp/results $exp/Random $exp/Uncertainty $exp/CoreSet $exp/UncertaintyDensity $exp/DualDensity
for idx in {0..200}
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
