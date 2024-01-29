arch=$1
dist_url=$2
dataroot=$3
# dataroot=/media/hdd2/data/thermal_ssl/
# dist_url=tcp://localhost:10001
# arch=efficient-vit
# arch=mobilenetv3_small_075
# arch=resnet18
python main_moco.py ${dataroot} \
    --workers 26 \
    --epochs 200 \
    --batch-size 512 \
    --multiprocessing-distributed \
    --rank 0 \
    --world-size 1 \
    --dist-url ${dist_url} \
    --moco-k 4096 \
    --mlp \
    --aug-plus \
    --cos \
    --arch $arch \
    --exp-name ${arch}-test-more-aug
