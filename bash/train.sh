arch=mobilenetv3_small_075
# arch=resnet18
python main_moco.py /media/hdd2/data/thermal_ssl/ \
    --workers 8 \
    --epochs 200 \
    --batch-size 256 \
    --multiprocessing-distributed \
    --rank 0 \
    --world-size 1 \
    --dist-url 'tcp://localhost:10001' \
    --moco-k 4096 \
    --mlp \
    --aug-plus \
    --cos \
    --arch $arch \
    --exp-name ${arch}-test \
    --epochs 5
