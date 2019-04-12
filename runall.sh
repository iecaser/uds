# nohup ./run.sh 500 1000 6 "0" "mnist" > mnist.500.1000.6.nohup 2>&1 &
nohup ./run.sh 500 3000 3 "0" "mnist" > nohup.mnist.500.3000.3 2>&1 &
# nohup ./run.sh 500 2000 4 "1" "mnist" > mnist.500.2000.4.nohup 2>&1 &
# nohup ./run.sh 100 200 10 "3" "mnist" > mnist.100.200.10.nohup 2>&1 &
# nohup ./run.sh 500 2000 10 "3" "cifar10" > cifar10.500.2000.10.nohup 2>&1 &
nohup ./run.sh 500 3000 3 "1" "cifar10" > nohup.cifar10.500.3000.3 2>&1 &
nohup ./run.sh 5 5 15 "3" "iris" > nohup.iris.5.5.15 2>&1 &
