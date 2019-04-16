# initial_size=500
# batch_size=3000
# iterations=3
# dataset="mnist"
initial_size=10
batch_size=10
iterations=9
dataset="iris"
exp="${dataset}.${initial_size}.${batch_size}.${iterations}"
python3 plotb.py --exp $exp --dataset $dataset --init $initial_size --batch $batch_size
# python3 plot.py --exp $exp --dataset $dataset --init $initial_size --batch $batch_size
sz $exp/results/${dataset}_${initial_size}_${batch_size}_999b.png
