!/bin/bash

# download all data for experiments

echo "MINIST data..."
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ./data/minist/
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ./data/minist/
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ./data/minist/
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ./data/minist/

echo "fashion-MINIST data..."
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -P ./data/fashion-minist/
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz -P ./data/fashion-minist/
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -P ./data/fashion-minist/
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -P ./data/fashion-minist/

echo "finished."