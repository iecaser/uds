#!/bin/bash

initial_size=10
batch_size=10
iterations=10
visible="0"
dataset="iris"
exp="iris.${initial_size}.${batch_size}.${iterations}.laua"
smooth=${dataset}_${initial_size}_${batch_size}_999
mkdir $exp
mkdir $exp/results $exp/Random $exp/Uncertainty $exp/CoreSet\
      $exp/UncertaintyDensity $exp/DualDensity $exp/UncertaintyDualDensity\
      $exp/UncertaintyEntropy $exp/Distance $exp/UncertaintyDistance
for idx in {0..500};
do
    echo "------- Random $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "Random" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/Random/" --visible $visible

    echo "------- Uncertainty $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "Uncertainty" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/Uncertainty/" --visible $visible

    echo "------- UncertaintyEntropy $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "UncertaintyEntropy" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/UncertaintyEntropy/" --visible $visible
#
    echo "------- CoreSet $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "CoreSet" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/CoreSet/" --visible $visible
#
    echo "------- UncertaintyDensity $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "UncertaintyDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/UncertaintyDensity/" --visible $visible

    echo "------- Distance $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "Distance" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/Distance/" --visible $visible

    echo "------- UncertaintyDistance $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "UncertaintyDistance" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/UncertaintyDistance/" --visible $visible

    echo "------- DualDensity $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "DualDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/DualDensity/" --visible $visible

    echo "------- UncertaintyDualDensity $idx... -------"
    python3 main.py $idx $dataset $batch_size $initial_size $iterations "UncertaintyDualDensity" "/home/zxf/workspace/DiscriminativeActiveLearning/$exp/UncertaintyDualDensity/" --visible $visible

    echo "------ ploting ...."
    python3 plot.py --idx $idx --exp $exp --dataset $dataset --init $initial_size --batch $batch_size

    python3 plot.py --exp $exp --dataset $dataset --init $initial_size --batch $batch_size
    mv $exp/results/${smooth}.png $exp/results/${smooth}.$idx.png
done
python3 plot.py --exp $exp --dataset $dataset --init $initial_size --batch $batch_size
