#!/bin/bash

# ==============================================================================
# 🛠️ 炼丹自动化并行挂机脚本（最多3个并行任务）
# ==============================================================================

# 1. 在这里【按顺序】定义你想跑的指令（你想加多少个都可以，格式为 "指令"）
COMMANDS=(
    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1"

    #"python main.py -algo FedAvg -fopt adam -lbs 100  -data MNIST -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 100 -ls 1 -atk label_flip -amr 0.25 -amon"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_unbalanced -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1"
    
    #"python main.py -algo FedAvg -fopt adam -lbs 100  -data MNIST_unbalanced -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_unbalanced -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_dir10 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1"
    
    #"python main.py -algo FedAvg -fopt adam -lbs 100  -data MNIST_dir10 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_dir10 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_dir5 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1"
    
    #"python main.py -algo FedAvg -fopt adam -lbs 100  -data MNIST_dir5 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_dir5 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_dir1 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1"
    
    #"python main.py -algo FedAvg -fopt adam -lbs 100  -data MNIST_dir1 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_dir1 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_dir0.5 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1"
    
    #"python main.py -algo FedAvg -fopt adam -lbs 100  -data MNIST_dir0.5 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon"
    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_dir0.5 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_dir0.1 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1"
    
    #"python main.py -algo FedAvg -fopt adam -lbs 100  -data MNIST_dir0.1 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon"

    #"python main.py -algo FedSGD -fopt adam -lbs 100  -data MNIST_dir0.1 -ncl 10 -m FedAvgMLP -nc 20 -jr 1 -gr 300 -ls 1 -atk label_flip -amr 0.25 -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -amon"
    
    # "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    # "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 100 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #0725凌晨尝试
    
    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    #"python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    #"python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    #"python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"





    #"python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -amon"
    
    #"python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    #"python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    #"python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    #"python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    #"python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    #"python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    #"python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"





    # "python main.py -algo FedAvg -data Cifar10_100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedAvg -data Cifar10_100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_100_dir0.1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_100_dir0.1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedAvg -data Cifar10_100_dir0.1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_100_dir0.1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_100_dir0.1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_100_dir0.1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_100_dir0.1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_100_dir1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_100_dir1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedAvg -data Cifar10_100_dir1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_100_dir1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_100_dir1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_100_dir1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_100_dir1 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_100_unbalanced -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_100_unbalanced -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedAvg -data Cifar10_100_unbalanced -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"
    
    # "python main.py -algo FedAvg -data Cifar10_100_unbalanced -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_100_unbalanced -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedAvg -data Cifar10_100_unbalanced -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    # "python main.py -algo FedAvg -data Cifar10_100_unbalanced -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_100  -lbs 100 -ncl 10 -m ResNet18 -nc 100 -jr 1 -gr 500 -ls 1 -amon -fopt adam -atk minmax -amr 0.25 -adev sign"

    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m VGG11 -nc 100 -jr 1 -gr 500 -ls 1 -amon -atk minmax -amr 0.25 -adev sign "

    #"python main.py -algo FedSGD -data Cifar10_100  -lbs 100 -ncl 10 -m ResNet18 -nc 100 -jr 1 -gr 300 -ls 1 -amon -fopt adam -atk minmax -amr 0.25 -adev sign -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m VGG11 -nc 100 -jr 1 -gr 300 -ls 1 -amon -atk minmax -amr 0.25 -adev sign -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100  -lbs 100 -ncl 10 -m ResNet18 -nc 100 -jr 1 -gr 300 -ls 1 -amon -fopt adam -atk minmax -amr 0.25 -adev sign -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m VGG11 -nc 100 -jr 1 -gr 300 -ls 1 -amon -atk minmax -amr 0.25 -adev sign -def cosine_cluster -cciter 50 -cctol 1e-6"

    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 3 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100_dir0.1 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 3 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100_dir1 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 3 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_100_unbalanced -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 3 -db 10000 -dseed 0"

    #"python main.py -algo FedSGD -data Cifar10_50 -fopt adam -lbs 50 -ncl 10 -m CNN -nc 50 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.2 -adev sign -amon -def dnc -dmar 0.2 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedSGD -data Cifar10_50 -fopt adam -lbs 150 -ncl 10 -m CNN -nc 50 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.2 -adev sign -amon -def dnc -dmar 0.2 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedSGD -data Cifar10_50 -fopt adam -lbs 200 -ncl 10 -m CNN -nc 50 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.2 -adev sign -amon -def dnc -dmar 0.2 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedSGD -data Cifar10_50 -fopt adam -lbs 300 -ncl 10 -m CNN -nc 50 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.2 -adev sign -amon -def dnc -dmar 0.2 -df 1 -di 1 -db 10000 -dseed 0"



    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"

    #"python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedSGD -data Cifar10_100_dir0.1 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedSGD -data Cifar10_100_dir0.1 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedSGD -data Cifar10_100_dir1 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedSGD -data Cifar10_100_dir1 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedSGD -data Cifar10_100_unbalanced -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedSGD -data Cifar10_100_unbalanced -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"


    # "python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedSGD -data Cifar10_dir0.1 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedSGD -data Cifar10_dir0.1 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedSGD -data Cifar10_dir1 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedSGD -data Cifar10_dir1 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    # "python main.py -algo FedSGD -data Cifar10_unbalanced -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon"

    # "python main.py -algo FedSGD -data Cifar10_unbalanced -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev sign -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"



#0727 std


    
    "python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    "python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_100 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    "python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_100_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    "python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_100_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -amon"
    
    "python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_100_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 100 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"





    "python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -amon"
    
    "python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10 -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -amon"
    
    "python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_dir0.1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -amon"
    
    "python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_dir1  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -amon"
    
    "python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedSGD -data Cifar10_unbalanced  -fopt adam -lbs 100 -ncl 10 -m CNN -nc 20 -jr 1 -gr 500 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

 #avg
    "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -amon "
    
    "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedAvg -data Cifar10 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -amon"
    
    "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedAvg -data Cifar10_dir0.1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -amon"
    
    "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedAvg -data Cifar10_dir1 -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -amon"
    
    "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon"

    "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon"
    
    "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def dnc -dmar 0.25 -df 1 -di 1 -db 10000 -dseed 0"

    "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minmax -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"

    "python main.py -algo FedAvg -data Cifar10_unbalanced -ncl 10 -m CNN -nc 20 -jr 1 -gr 200 -ls 1 -atk minsum -amr 0.25 -adev std -amon -def cosine_cluster -cciter 50 -cctol 1e-6"
)


# 2. 配置参数
MAX_PARALLEL=8         # 最大并行任务数
SLEEP_BETWEEN_START=10  # 每个任务启动之间的间隔（秒）
EXIT_ON_ERROR=0         # 是否在任务失败时退出（1: 是，0: 否）

# GPU admission control. A task starts only when its selected GPU has at least
# this much free memory. Set this to the measured peak of one experiment plus
# a safety margin; 6000 MiB is a conservative default for the current models.
DEFAULT_GPU_ID=0
DEFAULT_REQUIRED_FREE_MIB=2000
GPU_MEMORY_POLL_SECONDS=600  # Recheck every 10 minutes while waiting.

# Optional per-command override: prefix a command in COMMANDS with
# "GPU_MEM_MIB=8000 ", e.g. "GPU_MEM_MIB=8000 python main.py ... -did 0".

# ==============================================================================
# 🚀 并行任务管理核心逻辑
# ==============================================================================

# 创建工作目录存放日志和状态
WORK_DIR="/home/adslab/zzj/PFLlib_poison_try/PFLlib-master/results/parallel_logs"
WORK_DIR="/home/adslab/zzj/PFLlib_poison_try/PFLlib-master/results/parallel_logs"
mkdir -p "$WORK_DIR"
LOG_PREFIX="$WORK_DIR/task"

# 全局变量
declare -A RUNNING_PIDS  # 记录正在运行的进程ID
declare -A TASK_NUMBERS  # 记录每个PID对应的任务编号
NEXT_TASK_INDEX=0        # 下一个要启动的任务索引
TOTAL_CMDS=${#COMMANDS[@]}
NEXT_TASK_INDEX=0
TOTAL_CMDS=${#COMMANDS[@]}
COMPLETED_COUNT=0
FAILED_COUNT=0
GLOBAL_START_TIME=$(date +%s)

# Use an explicit -did/--device_id in a command when it targets another GPU.
# Commands without one follow main.py's CUDA:0 default.
gpu_id_for_command() {
    local cmd="$1"
    if [[ " $cmd " =~ [[:space:]](-did|--device_id)[[:space:]]+([0-9]+) ]]; then
        echo "${BASH_REMATCH[2]}"
    else
        echo "$DEFAULT_GPU_ID"
    fi
}

required_memory_for_command() {
    local cmd="$1"
    if [[ "$cmd" =~ GPU_MEM_MIB=([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "$DEFAULT_REQUIRED_FREE_MIB"
    fi
}

gpu_free_memory_mib() {
    local gpu_id="$1"
    nvidia-smi --id="$gpu_id" --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]'
}

# Block before launching a task, without occupying a MAX_PARALLEL slot.
wait_for_gpu_memory() {
    local task_num="$1"
    local gpu_id="$2"
    local required_mib="$3"
    local free_mib

    while true; do
        free_mib=$(gpu_free_memory_mib "$gpu_id")
        if [[ ! "$free_mib" =~ ^[0-9]+$ ]]; then
            echo "[task $task_num] Cannot query free memory of GPU $gpu_id with nvidia-smi." >&2
            return 2
        fi
        if (( free_mib >= required_mib )); then
            echo "[task $task_num] GPU $gpu_id admitted: ${free_mib} MiB free (requires ${required_mib} MiB)."
            return 0
        fi

        echo "[task $task_num] GPU $gpu_id has ${free_mib} MiB free; requires ${required_mib} MiB. Rechecking in ${GPU_MEMORY_POLL_SECONDS}s..."
        sleep "$GPU_MEMORY_POLL_SECONDS"
    done
}

echo "===================================================================="
echo "🔥 自动化并行实验启动！"
echo "   总任务数: $TOTAL_CMDS"
echo "   最大并行数: $MAX_PARALLEL"
echo "   启动间隔: ${SLEEP_BETWEEN_START}秒"
echo "   日志目录: $WORK_DIR"
echo "===================================================================="

# 函数：启动一个新任务
start_task() {
    local task_index=$1
    local task_num=$((task_index + 1))
    local cmd="${COMMANDS[$task_index]}"
    local log_file="${LOG_PREFIX}_${task_num}.log"
    local gpu_id
    local required_mib
    gpu_id=$(gpu_id_for_command "$cmd")
    required_mib=$(required_memory_for_command "$cmd")

    if ! wait_for_gpu_memory "$task_num" "$gpu_id" "$required_mib"; then
        echo "[task $task_num] GPU admission check failed; stopping the queue." >&2
        exit 2
    fi

    echo ""
    echo "--------------------------------------------------------------------"
    echo "🟢 [任务 $task_num/$TOTAL_CMDS] 启动"
    echo "   时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "   日志: $log_file"
    echo "   指令: $cmd"
    echo "--------------------------------------------------------------------"

    # 在后台执行任务，输出重定向到日志文件
    {
        echo "===================================================================="
        echo "任务 $task_num 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "===================================================================="
        echo "执行指令: $cmd"
        echo ""

        eval "$cmd"
        local status=$?

        echo ""
        echo "===================================================================="
        echo "任务 $task_num 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "退出状态: $status"
        echo "===================================================================="

        exit $status
    } > "$log_file" 2>&1 &

    local pid=$!
    RUNNING_PIDS[$pid]=1
    TASK_NUMBERS[$pid]=$task_num

    echo "   ✓ 进程ID: $pid"
}

# 函数：检查并处理完成的任务
check_completed_tasks() {
    for pid in "${!RUNNING_PIDS[@]}"; do
        # 检查进程是否还在运行
        if ! kill -0 "$pid" 2>/dev/null; then
            # 进程已结束，获取退出状态
            wait "$pid"
            local status=$?
            local task_num=${TASK_NUMBERS[$pid]}

            if [ $status -eq 0 ]; then
                echo ""
                echo "✅ [任务 $task_num/$TOTAL_CMDS] 成功完成 (PID: $pid)"
                ((COMPLETED_COUNT++))
            else
                echo ""
                echo "❌ [任务 $task_num/$TOTAL_CMDS] 执行失败 (PID: $pid, 退出码: $status)"
                ((FAILED_COUNT++))

                if [ $EXIT_ON_ERROR -eq 1 ]; then
                    echo "🚨 检测到任务失败，终止所有任务..."
                    kill_all_tasks
                    exit $status
                fi
            fi

            # 从运行列表中移除
            unset RUNNING_PIDS[$pid]
            unset TASK_NUMBERS[$pid]

            # 尝试启动下一个任务
            if [ $NEXT_TASK_INDEX -lt $TOTAL_CMDS ]; then
                echo "   准备启动下一个任务 (等待 ${SLEEP_BETWEEN_START}秒)..."
                sleep "$SLEEP_BETWEEN_START"
                start_task $NEXT_TASK_INDEX
                ((NEXT_TASK_INDEX++))
            fi
        fi
    done
}

# 函数：杀死所有运行中的任务
kill_all_tasks() {
    echo "正在终止所有运行中的任务..."
    for pid in "${!RUNNING_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  终止进程 $pid (任务 ${TASK_NUMBERS[$pid]})"
            kill "$pid" 2>/dev/null
        fi
    done
}

# 函数：显示当前状态
show_status() {
    local running_count=${#RUNNING_PIDS[@]}
    local remaining=$((TOTAL_CMDS - NEXT_TASK_INDEX))
    echo ""
    echo "📊 当前状态: 运行中 $running_count | 已完成 $COMPLETED_COUNT | 失败 $FAILED_COUNT | 待启动 $remaining"
}

# 捕获中断信号，优雅退出
trap 'echo ""; echo "⚠️  收到中断信号，正在清理..."; kill_all_tasks; exit 130' INT TERM

# ==============================================================================
# 主循环：启动初始任务并监控
# ==============================================================================

# 启动前 MAX_PARALLEL 个任务
initial_tasks=$((MAX_PARALLEL < TOTAL_CMDS ? MAX_PARALLEL : TOTAL_CMDS))
for ((i=0; i<initial_tasks; i++)); do
    start_task $NEXT_TASK_INDEX
    ((NEXT_TASK_INDEX++))

    # 如果不是最后一个初始任务，等待间隔时间
    if [ $i -lt $((initial_tasks - 1)) ]; then
        echo "   等待 ${SLEEP_BETWEEN_START}秒后启动下一个任务..."
        sleep "$SLEEP_BETWEEN_START"
    fi
done

# 主监控循环
while [ ${#RUNNING_PIDS[@]} -gt 0 ]; do
    sleep 5  # 每5秒检查一次
    check_completed_tasks
    show_status
done

# ==============================================================================
# 完成总结
# ==============================================================================
GLOBAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((GLOBAL_END_TIME - GLOBAL_START_TIME))

echo ""
echo "===================================================================="
echo "🎉 所有任务已完成！"
echo "   总任务数: $TOTAL_CMDS"
echo "   成功: $COMPLETED_COUNT"
echo "   失败: $FAILED_COUNT"
echo "   总耗时: $((TOTAL_DURATION / 3600)) 小时 $(( (TOTAL_DURATION % 3600) / 60 )) 分 $((TOTAL_DURATION % 60)) 秒"
echo "   日志目录: $WORK_DIR"
echo "===================================================================="

# 退出码：如果有任务失败则返回1，否则返回0
[ $FAILED_COUNT -eq 0 ] && exit 0 || exit 1
