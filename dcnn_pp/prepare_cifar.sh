echo "downloading original dataset mirror..."
cd ./data
if [ ! -f cifar.tgz ]; then
    wget https://pjreddie.com/media/files/cifar.tgz
fi

 
if [ ! -d cifar ]; then
    echo "unpacking..."
    tar xzf cifar.tgz
    echo "packing into RecordIO format..."
    python ../cifar-10.py --path ./cifar
fi       

echo "making Record format data file..."
if [ ! -d RecordIO-cifar ]; then
    mkdir RecordIO-cifar
fi

mkdir tmp
# prefix and images directory (they must be absolute path)
# train data
python ../im2rec.py --list --recursive ./tmp/train ./cifar/train/
python ../im2rec.py --pass-through  ./tmp/ ./cifar/train
mv ./tmp/train.* ./RecordIO-cifar/
# test data
python ../im2rec.py --list --recursive ./tmp/test ./cifar/test
python ../im2rec.py --pass-through  ./tmp/ ./cifar/test
mv ./tmp/test.* ./RecordIO-cifar/

rm -rf tmp
rm -rf cifar

mv RecordIO-cifar cifar

cd ..

echo "Done."