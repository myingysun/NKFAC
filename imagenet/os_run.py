#imagenet
import os,time

os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4  --master_addr 127.0.0.2 --master_port 29502 main.py /data2/sunying/ImageNet  --model r18  -b 256 --lr 0.05 --wd 0.0002  --alg nkfac --tconv 20 --tinv 200 --stat_decay 0.95 --d 0.01 >logout/r18_nkfac_lr25_wd42.log &")
os.system("CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=4  --master_addr 127.0.0.6 --master_port 29506 main.py /data2/sunying/ImageNet  --model r50  -b 256 --lr 0.05 --wd 0.0002  --alg nkfac --tconv 20 --tinv 200 --stat_decay 0.95 --d 0.01 >logout/r50_nkfac_lr25_wd42.log ")
time.sleep(100)


