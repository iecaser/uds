initial_size=5
batch_size=5
iterations=20
dataset="iris"
exp="iris.5.5.20.0"
mkdir $exp
mkdir $exp/results $exp/Random $exp/Uncertainty $exp/CoreSet $exp/UncertaintyDensity $exp/DualDensity
for idx in {0..100}
do
    echo "------- Random $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "Random" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/Random/"

    echo "------- Uncertainty $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "Uncertainty" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/Uncertainty/"

    echo "------- CoreSet $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "CoreSet" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/CoreSet/"

    echo "------- UncertaintyDensity $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "UncertaintyDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/UncertaintyDensity/"

    echo "------- DualDensity $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "DualDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/DualDensity/"

    echo "------ ploting ...."
    python3 plot.py --idx $idx --exp $exp --dataset $dataset --init $initial_size --batch $batch_size
done
python3 plot.py --exp $exp --dataset $dataset --init $initial_size --batch $batch_size
