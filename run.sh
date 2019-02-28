initial_size=1000
batch_size=5000
iterations=4
visible='0,1'
for idx in {0..9}
do
    echo "------- Random $idx... -------"
    python3 main.py $idx "mnist" $batch_size $initial_size $iterations "Random" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/Random/"

    echo "------- Uncertainty $idx... -------"
    python3 main.py $idx "mnist" $batch_size $initial_size $iterations "Uncertainty" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/Uncertainty/"

    echo "------- CoreSet $idx... -------"
    python3 main.py $idx "mnist" $batch_size $initial_size $iterations "CoreSet" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/CoreSet/"

    echo "------- DualDensity $idx... -------"
    python3 main.py $idx "mnist" $batch_size $initial_size $iterations "DualDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/DualDensity/"

    # echo "------- DiscriminativeLearned $idx... -------"
    # python3 main.py $idx "mnist" $batch_size $initial_size $iterations "DiscriminativeLearned" "/home/zxf/workspace/DiscriminativeActiveLearning/exp/DiscriminativeLearned/"
done
python3 plot.py --dataset mnist --init $initial_size --batch $batch_size
