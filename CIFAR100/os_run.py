#cifar100
import os,time
### nkfac
os.system("CUDA_VISIBLE_DEVICES=0  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.1 --master_port 29501 main_cos.py --lr 0.05 --wd 0.001   --alg nkfac   --epochs 200  --model r18 --tconv 20 --tinv 200 --stat_decay 0.95 --d 0.01 > logout/r18_nkfac_1.log &")
os.system("CUDA_VISIBLE_DEVICES=1  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.2 --master_port 29502 main_cos.py --lr 0.05 --wd 0.001   --alg nkfac   --epochs 200  --model r18 --tconv 20 --tinv 200 --stat_decay 0.95 --d 0.01 > logout/r18_nkfac_2.log &")
os.system("CUDA_VISIBLE_DEVICES=2  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.3 --master_port 29503 main_cos.py --lr 0.05 --wd 0.001   --alg nkfac   --epochs 200  --model r18 --tconv 20 --tinv 200 --stat_decay 0.95 --d 0.01 > logout/r18_nkfac_3.log &")
os.system("CUDA_VISIBLE_DEVICES=3  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.4 --master_port 29504 main_cos.py --lr 0.05 --wd 0.001   --alg nkfac   --epochs 200  --model r18 --tconv 20 --tinv 200 --stat_decay 0.95 --d 0.01 > logout/r18_nkfac_4.log ")
time.sleep(600)

