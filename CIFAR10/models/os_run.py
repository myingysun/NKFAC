#cifar10 e200 bs128 
import os,time
#############################
# #r18
# ##############
#
# os.system("CUDA_VISIBLE_DEVICES=0  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.1 --master_port 29501 main_wg.py --lr 0.05 --wd 0.001   --alg sgdwg   --epochs 200  --model r18 --d 0.001  --st 0.9 > logout/r18_sgdwg_lr25_wd31_d31_st9_1.log &")
# os.system("CUDA_VISIBLE_DEVICES=1  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.2 --master_port 29502 main_wg.py --lr 0.05 --wd 0.001   --alg sgdwg   --epochs 200  --model r18 --d 0.001  --st 0.9 > logout/r18_sgdwg_lr25_wd31_d31_st9_2.log &")
# os.system("CUDA_VISIBLE_DEVICES=2  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.3 --master_port 29503 main_wg.py --lr 0.05 --wd 0.001   --alg sgdwg   --epochs 200  --model r18 --d 0.001  --st 0.9 > logout/r18_sgdwg_lr25_wd31_d31_st9_3.log &")
# os.system("CUDA_VISIBLE_DEVICES=3  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.4 --master_port 29504 main_wg.py --lr 0.05 --wd 0.001   --alg sgdwg   --epochs 200  --model r18 --d 0.001  --st 0.9 > logout/r18_sgdwg_lr25_wd31_d31_st9_4.log ")
# #
# time.sleep(1000)
#
#
# os.system("CUDA_VISIBLE_DEVICES=0  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.1 --master_port 29501 main_wg.py --lr 0.001 --wd 0.5   --alg adamwg   --epochs 200  --model r18 --d 0.001  --st 0.9 > logout/r18_adamwg_lr31_wd11_d31_st9_1.log &")
# os.system("CUDA_VISIBLE_DEVICES=1  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.2 --master_port 29502 main_wg.py --lr 0.001 --wd 0.5   --alg adamwg   --epochs 200  --model r18 --d 0.001  --st 0.9 > logout/r18_adamwg_lr31_wd11_d31_st9_2.log &")
# os.system("CUDA_VISIBLE_DEVICES=2  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.3 --master_port 29503 main_wg.py --lr 0.001 --wd 0.5   --alg adamwg   --epochs 200  --model r18 --d 0.001  --st 0.5 > logout/r18_adamwg_lr31_wd11_d31_st9_3.log &")
# os.system("CUDA_VISIBLE_DEVICES=3  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.4 --master_port 29504 main_wg.py --lr 0.001 --wd 0.5   --alg adamwg   --epochs 200  --model r18 --d 0.001  --st 0.9 > logout/r18_adamwg_lr31_wd11_d31_st9_4.log ")
#

os.system("CUDA_VISIBLE_DEVICES=4  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.5 --master_port 29505 main_wg.py --lr 0.1 --wd 0.0005   --alg shampoo   --epochs 200  --model r18 --d 1e-4 > logout/r18_shampoo_lr11_wd45_d41_1.log &")
os.system("CUDA_VISIBLE_DEVICES=5  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.6 --master_port 29506 main_wg.py --lr 0.1 --wd 0.0005   --alg shampoo   --epochs 200  --model r18 --d 1e-4 > logout/r18_shampoo_lr11_wd45_d41_2.log &")
os.system("CUDA_VISIBLE_DEVICES=6  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.7 --master_port 29507 main_wg.py --lr 0.1 --wd 0.0005   --alg shampoo   --epochs 200  --model r18 --d 1e-4 > logout/r18_shampoo_lr11_wd45_d41_3.log &")
os.system("CUDA_VISIBLE_DEVICES=7  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.8 --master_port 29508 main_wg.py --lr 0.1 --wd 0.0005   --alg shampoo   --epochs 200  --model r18 --d 1e-4 > logout/r18_shampoo_lr11_wd45_d41_4.log ")
time.sleep(200)

os.system("CUDA_VISIBLE_DEVICES=4  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.5 --master_port 29505 main_wg.py --lr 0.05 --wd 0.001   --alg shampoo   --epochs 200  --model r18 --d 1e-4 > logout/r18_shampoo_lr25_wd31_d41_1.log &")
os.system("CUDA_VISIBLE_DEVICES=5  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.6 --master_port 29506 main_wg.py --lr 0.05 --wd 0.001   --alg shampoo   --epochs 200  --model r18 --d 1e-4 > logout/r18_shampoo_lr25_wd31_d41_2.log &")
os.system("CUDA_VISIBLE_DEVICES=6  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.7 --master_port 29507 main_wg.py --lr 0.05 --wd 0.001   --alg shampoo   --epochs 200  --model r18 --d 1e-4 > logout/r18_shampoo_lr25_wd31_d41_3.log &")
os.system("CUDA_VISIBLE_DEVICES=7  nohup python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.8 --master_port 29508 main_wg.py --lr 0.05 --wd 0.001   --alg shampoo   --epochs 200  --model r18 --d 1e-4 > logout/r18_shampoo_lr25_wd31_d41_4.log ")
time.sleep(200)