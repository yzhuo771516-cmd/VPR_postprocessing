import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt


def compute_descriptor_distances(npz_file):
    # 加载描述符数据
    data = np.load(npz_file)
    descriptors = data["descriptors"]  # shape: (N, 512)
    image_names = data["image_names"]  # list of N image names

    print(f"Loaded {len(image_names)} descriptors with shape {descriptors.shape}")

    # 计算两两L2距离
    distances = np.sqrt(((descriptors[:, np.newaxis, :] - descriptors[np.newaxis, :, :]) ** 2).sum(axis=-1))
    # 或者使用更简洁的方式:
    # from scipy.spatial.distance import pdist, squareform
    # distances = squareform(pdist(descriptors, metric='euclidean'))

    return distances, image_names


def compute_cosine_similarity(npz_file):
    # 加载描述符数据
    data = np.load(npz_file)
    descriptors = data["descriptors"]
    image_names = data["image_names"]

    # L2归一化
    descriptors = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
    # 计算余弦相似度矩阵
    cosine_sim = np.dot(descriptors, descriptors.T)

    return cosine_sim, image_names


def plot_heatmap(matrix, image_names, output_file, title):
    # 创建DataFrame
    df = pd.DataFrame(matrix, index=image_names, columns=image_names)

    # 四舍五入到两位小数
    df = df.round(2)

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="viridis", center=0 if 'cosine' in title.lower() else None)  # 对于余弦相似度，中心设为0
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Heatmap saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_file", type=str,
                        default="D:/projects/cosplace/result/hkustgz_seq1_descriptors.npz",
                        help="Path to descriptors NPZ file")
    parser.add_argument("--output_l2", type=str, default="l2_heatmap.png",
                        help="Output file for L2 distance heatmap")
    parser.add_argument("--output_cosine", type=str, default="cosine_heatmap.png",
                        help="Output file for cosine similarity heatmap")

    args = parser.parse_args()

    # 计算L2距离并绘制热力图
    l2_distances, image_names = compute_descriptor_distances(args.npz_file)
    plot_heatmap(l2_distances, image_names, args.output_l2, "L2 Distance Heatmap")

    # 计算余弦相似度并绘制热力图
    cosine_sim, image_names = compute_cosine_similarity(args.npz_file)
    plot_heatmap(cosine_sim, image_names, args.output_cosine, "Cosine Similarity Heatmap")