import pandas as pd
import numpy as np
import os
import argparse

# PEMS 系列数据集的标准起始时间配置 (用于处理没有时间戳的 .npz 文件)
PEMS_START_DATES = {
    'PEMS03': '2018-09-01',
    'PEMS04': '2018-01-01',
    'PEMS07': '2017-05-01',
    'PEMS08': '2016-07-01'
}

def convert_to_tslib_format(input_path, output_path=None):
    """
    将交通数据集转换为 Time-Series-Library (TSLib) 要求的 CSV 格式。
    格式要求: [date, feature1, feature2, ..., OT]
    """
    
    if output_path is None:
        filename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{filename}.csv"

    print(f"🚀 开始处理: {input_path}")
    df = None

    # ================= 1. 读取数据 =================
    try:
        if input_path.endswith('.h5'):
            print("   ↳ 检测为 HDF5 格式 (METR-LA / PEMS-BAY)")
            try:
                df = pd.read_hdf(input_path)
            except KeyError:
                # 尝试常见的 key
                try:
                    df = pd.read_hdf(input_path, key='df')
                except KeyError:
                    df = pd.read_hdf(input_path, key='data')
            
            # H5 文件通常索引是 datetime，需要 reset 出来作为一列
            df.reset_index(inplace=True)
            # 强制将第一列（原索引）命名为 'date'
            df.rename(columns={df.columns[0]: 'date'}, inplace=True)

        elif input_path.endswith('.npz'):
            print("   ↳ 检测为 NPZ 格式 (PEMS03/04/07/08)")
            data = np.load(input_path)
            
            # 提取数据矩阵
            if 'data' in data:
                array_3d = data['data']
            else:
                array_3d = data[list(data.keys())[0]]
            
            print(f"   ↳ 原始维度: {array_3d.shape}")

            # 处理维度: (Time, Nodes, Channels) -> (Time, Nodes)
            # 默认取 Channel 0 (Traffic Flow)
            if len(array_3d.shape) == 3:
                df_data = array_3d[:, :, 0]
            else:
                df_data = array_3d

            # 生成时间轴 (因为 npz 里没有时间)
            filename_base = os.path.basename(input_path).upper()
            start_date = '2018-01-01' # 默认兜底
            
            # 自动匹配起始时间
            for key, date_str in PEMS_START_DATES.items():
                if key in filename_base:
                    start_date = date_str
                    print(f"   ↳ 匹配到 {key}，使用起始时间: {start_date}")
                    break
            
            # 生成时间序列 (5分钟间隔)
            time_index = pd.date_range(start=start_date, periods=df_data.shape[0], freq='5T')
            
            df = pd.DataFrame(df_data)
            # 插入 date 列
            df.insert(0, 'date', time_index)

        else:
            print("❌ 错误: 不支持的文件格式，仅支持 .h5 或 .npz")
            return

    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # ================= 2. 格式适配 (关键步骤) =================
    # TS-Library 强制要求：必须有一列叫 'OT' (Output Target)
    # 对于多变量预测 (M 任务)，我们把最后一列重命名为 OT 即可
    
    print("   ↳ 正在执行列重命名适配 (Last Column -> OT)...")
    
    # 获取当前最后一列的名字
    cols = list(df.columns)
    last_col = cols[-1]
    
    if last_col != 'OT':
        df.rename(columns={last_col: 'OT'}, inplace=True)
        print(f"     已将 [{last_col}] 重命名为 [OT]")
    else:
        print("     最后一列已经是 OT，跳过重命名。")

    # ================= 3. 保存文件 =================
    print(f"   ↳ 正在保存至: {output_path}")
    df.to_csv(output_path, index=False)
    
    print("-" * 40)
    print(f"✅ 转换成功！")
    print(f"   数据形状: {df.shape}")
    print(f"   前3行预览:\n{df.head(3)}")
    print("-" * 40)

if __name__ == "__main__":
    # 使用方法：直接在下方修改文件名，或者通过命令行传参
    # 示例: python prepare_data.py --input metr-la.h5
    
    parser = argparse.ArgumentParser(description='Convert Traffic Data to TSLib CSV format')
    parser.add_argument('--input', type=str, default='metr-la.h5', help='Input file path (.h5 or .npz)')
    parser.add_argument('--output', type=str, default='../dataset', help='Output CSV path (optional)')
    
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        convert_to_tslib_format(args.input, args.output)
    else:
        print(f"❌ 找不到文件: {args.input}")
        print("提示: 请将脚本放在数据目录下，或指定完整路径。")