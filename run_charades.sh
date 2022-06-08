export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python moment_localization/train.py --cfg experiments/charades/i3d-finetune-mgpn256.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/vgg-mgpn256.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/i3d-raw-mgpn256.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/i3d-finetune-mgpn512.yaml --verbose

