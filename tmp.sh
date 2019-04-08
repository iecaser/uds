initial_size=500
batch_size=3000
iterations=3
dataset="mnist"
exp="${dataset}.${initial_size}.${batch_size}.${iterations}.l0ua"
python3 plot.py --exp $exp --dataset $dataset --init $initial_size --batch $batch_size
