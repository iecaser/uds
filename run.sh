#!/bin/bash

initial_size=$1
batch_size=$2
iterations=$3
visible=$4
dataset=$5
exp="${dataset}.${initial_size}.${batch_size}.${iterations}"
smooth=${dataset}_${initial_size}_${batch_size}_999
mkdir $exp $exp/results
for idx in {200..500}
    do
    for method in Random\
                    DualDensityBeam2\
                    DualDensityBeam3\
                    DualDensity\
                    CoreSet\
                    UncertaintyDensity\
                    Uncertainty\
                    # EGL\
                    # DualDensityBeam4\
                    # Adversarial\
                    # UncertaintyDualDensity\
                    # DynamicUncertaintyDualDensity\
                    # UncertaintyDistance\
                    # AntiUncertaintyDualDensity\
                    # UncertaintyEntropy\
                    # Distance\
    do
        mexp=$exp/$method
        if [ ! -d $mexp ]; then
            mkdir $mexp
        fi
        echo "------- $method $idx -------"
        python3 main.py $idx $dataset $batch_size $initial_size $iterations $method\
                "/home/zxf/workspace/DiscriminativeActiveLearning/$mexp/" --visible $visible
        # python3 scatter.py --dir $mexp
    done

    echo "------ ploting ...."
    python3 plot.py --idx $idx --exp $exp --dataset $dataset --init $initial_size --batch $batch_size
    python3 plot.py --exp $exp --dataset $dataset --init $initial_size --batch $batch_size
    mv $exp/results/${smooth}.png $exp/results/${smooth}.$idx.png
done
