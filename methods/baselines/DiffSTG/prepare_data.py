#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据准备脚本：将FlowGNN的h5数据转换为DiffSTG所需的格式
- 生成flow.npy: (T, V, D) 其中D=3 (原始数据, tod, dow)
- 转换adj_mx.pkl为adj.npy: (V, V)
- 数据划分: 7:1:2 (训练:验证:测试)
- 只对第一个feature（流量）做标准化
"""

import numpy as np
import pandas as pd
import h5py
import pickle
import os
from pathlib import Path

def load_h5_data(file_path):
    """从h5文件加载数据"""
    print(f"Loading data from {file_path}")
    with h5py.File(file_path, 'r') as f:
        # 打印h5文件的结构
        print(f"Keys in h5 file: {list(f.keys())}")

        # 根据不同的h5文件结构加载数据
        if 'df' in f.keys():
            data = f['df']['block0_values'][:]  # (T, V) - metr-la格式
        elif 'speed' in f.keys():
            data = f['speed']['block0_values'][:]  # (T, V) - pems-bay格式
        else:
            raise ValueError(f"Unknown h5 file structure. Keys: {list(f.keys())}")

    print(f"Data shape: {data.shape}")
    return data

def load_pkl_adj(file_path):
    """从pkl文件加载邻接矩阵"""
    print(f"Loading adjacency matrix from {file_path}")
    with open(file_path, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
    print(f"Adjacency matrix shape: {adj_mx.shape}")
    print(f"Number of sensors: {len(sensor_ids)}")
    return adj_mx, sensor_ids, sensor_id_to_ind

def compute_tod_dow(num_samples, points_per_hour=12):
    """
    计算time of day和day of week特征

    Args:
        num_samples: 总时间步数
        points_per_hour: 每小时的采样点数（默认12，即5分钟一个点）

    Returns:
        tod: (T,) 一天中的时间索引 [0, points_per_hour*24)
        dow: (T,) 一周中的天索引 [0, 7)
    """
    # 假设数据从周一开始
    tod = np.array([i % (points_per_hour * 24) for i in range(num_samples)])
    dow = np.array([(i // (points_per_hour * 24)) % 7 for i in range(num_samples)])
    return tod, dow

def normalize_data(data, train_ratio=0.7):
    """
    只对训练集计算mean和std，然后标准化整个数据集

    Args:
        data: (T, V) 原始流量数据
        train_ratio: 训练集比例

    Returns:
        normalized_data: (T, V) 标准化后的数据
        mean: 训练集均值
        std: 训练集标准差
    """
    train_size = int(data.shape[0] * train_ratio)
    train_data = data[:train_size]

    mean = np.mean(train_data)
    std = np.std(train_data)

    print(f"Training data statistics - Mean: {mean:.4f}, Std: {std:.4f}")

    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def prepare_dataset(dataset_name, h5_path, pkl_path, output_dir, points_per_hour=12):
    """
    准备单个数据集

    Args:
        dataset_name: 数据集名称 (metr-la 或 pems-bay)
        h5_path: h5文件路径
        pkl_path: pkl邻接矩阵文件路径
        output_dir: 输出目录
        points_per_hour: 每小时采样点数
    """
    print(f"\n{'='*60}")
    print(f"Preparing dataset: {dataset_name}")
    print(f"{'='*60}")

    # 创建输出目录
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 加载原始流量数据
    raw_data = load_h5_data(h5_path)  # (T, V)
    T, V = raw_data.shape

    # 2. 计算tod和dow特征
    tod, dow = compute_tod_dow(T, points_per_hour)

    # 3. 标准化流量数据（只对第一个特征）
    normalized_flow, mean, std = normalize_data(raw_data, train_ratio=0.7)

    # 4. 组合3个特征：标准化后的流量、tod、dow
    # 注意：tod和dow不需要标准化，保持原始值
    flow_3d = np.stack([
        normalized_flow,  # (T, V) - 标准化后的流量
        np.tile(tod[:, np.newaxis], (1, V)),  # (T, V) - tod
        np.tile(dow[:, np.newaxis], (1, V))   # (T, V) - dow
    ], axis=-1)  # (T, V, 3)

    print(f"Flow data shape: {flow_3d.shape}")
    print(f"Feature 0 (normalized flow) - min: {flow_3d[:,:,0].min():.4f}, max: {flow_3d[:,:,0].max():.4f}")
    print(f"Feature 1 (tod) - min: {flow_3d[:,:,1].min():.0f}, max: {flow_3d[:,:,1].max():.0f}")
    print(f"Feature 2 (dow) - min: {flow_3d[:,:,2].min():.0f}, max: {flow_3d[:,:,2].max():.0f}")

    # 5. 保存flow.npy
    flow_output = output_path / 'flow.npy'
    np.save(flow_output, flow_3d)
    print(f"Saved flow data to {flow_output}")

    # 6. 加载并保存邻接矩阵
    adj_mx, sensor_ids, sensor_id_to_ind = load_pkl_adj(pkl_path)
    adj_output = output_path / 'adj.npy'
    np.save(adj_output, adj_mx)
    print(f"Saved adjacency matrix to {adj_output}")

    # 7. 保存数据集信息
    info = {
        'dataset_name': dataset_name,
        'num_samples': T,
        'num_nodes': V,
        'num_features': 3,
        'points_per_hour': points_per_hour,
        'train_ratio': 0.7,
        'val_ratio': 0.1,
        'test_ratio': 0.2,
        'train_size': int(T * 0.7),
        'val_size': int(T * 0.1),
        'test_size': T - int(T * 0.7) - int(T * 0.1),
        'mean': float(mean),
        'std': float(std),
        'sensor_ids': sensor_ids.tolist() if hasattr(sensor_ids, 'tolist') else list(sensor_ids)
    }

    info_output = output_path / 'dataset_info.txt'
    with open(info_output, 'w') as f:
        for key, value in info.items():
            if key != 'sensor_ids':
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {len(value)} sensors\n")
    print(f"Saved dataset info to {info_output}")

    # 8. 创建readme.txt
    readme_output = output_path / 'readme.txt'
    with open(readme_output, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"adj.npy: ({V}, {V})\n")
        f.write(f"flow.npy: ({T}, {V}, 3)\n\n")
        f.write(f"T: {T} (total number of time steps)\n")
        f.write(f"V: {V} (number of nodes)\n")
        f.write(f"D: 3 (number of features)\n\n")
        f.write(f"Features:\n")
        f.write(f"  - Feature 0: Normalized flow (mean={mean:.4f}, std={std:.4f})\n")
        f.write(f"  - Feature 1: Time of day (0-{points_per_hour*24-1})\n")
        f.write(f"  - Feature 2: Day of week (0-6)\n\n")
        f.write(f"Data split (7:1:2):\n")
        f.write(f"  - Train: 0 to {info['train_size']} ({info['train_ratio']*100:.0f}%)\n")
        f.write(f"  - Val: {info['train_size']} to {info['train_size']+info['val_size']} ({info['val_ratio']*100:.0f}%)\n")
        f.write(f"  - Test: {info['train_size']+info['val_size']} to {T} ({info['test_ratio']*100:.0f}%)\n\n")
        f.write(f"Temporal settings:\n")
        f.write(f"  - Points per hour: {points_per_hour}\n")
        f.write(f"  - Historical steps: 144\n")
        f.write(f"  - Prediction steps: 144\n")
    print(f"Saved readme to {readme_output}")

    print(f"\n{dataset_name} dataset preparation completed!")
    return info

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prepare METR-LA and PEMS-BAY data for DiffSTG')
    parser.add_argument('--source_data_dir', type=str, default='data/raw',
                        help='Directory containing metr-la.h5, pems-bay.h5, and sensor_graph/')
    parser.add_argument('--output_dir', type=str, default='data/processed/DiffSTG',
                        help='Output directory for DiffSTG processed data')
    args = parser.parse_args()

    # 设置路径
    flowgnn_data_dir = args.source_data_dir
    diffstg_data_dir = args.output_dir

    # 准备METR-LA数据集
    metr_info = prepare_dataset(
        dataset_name='metr-la',
        h5_path=os.path.join(flowgnn_data_dir, 'metr-la.h5'),
        pkl_path=os.path.join(flowgnn_data_dir, 'sensor_graph', 'adj_mx.pkl'),
        output_dir=diffstg_data_dir,
        points_per_hour=12  # 5分钟采样
    )

    # 准备PEMS-BAY数据集
    pems_info = prepare_dataset(
        dataset_name='pems-bay',
        h5_path=os.path.join(flowgnn_data_dir, 'pems-bay.h5'),
        pkl_path=os.path.join(flowgnn_data_dir, 'sensor_graph', 'adj_mx_bay.pkl'),
        output_dir=diffstg_data_dir,
        points_per_hour=12  # 5分钟采样
    )

    print(f"\n{'='*60}")
    print("All datasets prepared successfully!")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"  METR-LA: {metr_info['num_samples']} samples, {metr_info['num_nodes']} nodes")
    print(f"  PEMS-BAY: {pems_info['num_samples']} samples, {pems_info['num_nodes']} nodes")
    print(f"\nOutput directory: {diffstg_data_dir}")

if __name__ == '__main__':
    main()
