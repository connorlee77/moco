dataroot=/media/hdd2/data/thermal_ssl/
dist_url=tcp://localhost:10010
arch=fast-scnn
python main_moco.py ${dataroot} \
    --workers 8 \
    --epochs 150 \
    --batch-size 256 \
    --multiprocessing-distributed \
    --rank 0 \
    --world-size 1 \
    --dist-url ${dist_url} \
    --moco-k 4096 \
    --mlp \
    --aug-plus \
    --cos \
    --arch $arch \
    --exp-name ${arch}-test-moreaug

# dist_url=tcp://localhost:10011
# arch=mobilenetv3_small_075
# python main_moco.py ${dataroot} \
#     --workers 8 \
#     --epochs 300 \
#     --batch-size 256 \
#     --multiprocessing-distributed \
#     --rank 0 \
#     --world-size 1 \
#     --dist-url ${dist_url} \
#     --moco-k 4096 \
#     --mlp \
#     --aug-plus \
#     --cos \
#     --arch $arch \
#     --exp-name ${arch}-test-moreaug
