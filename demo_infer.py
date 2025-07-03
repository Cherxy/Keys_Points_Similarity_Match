import cv2
import numpy as np
from pathlib import Path
from onnx_runner import LightGlueRunner, viz2d
import math


def read_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def resize_image(
    image: np.ndarray,
    size: list[int] | int,
    fn: str = "max",
    interp: str | None = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return image / 255.0



def load_image(
    path: str,
    crop_coords,
    grayscale: bool = False,
    resize: int = None,
    fn: str = "max",
    interp: str = "area",
):
    img = read_image(path, grayscale=grayscale)
    scales = [1, 1]
    if resize is not None:
        img, scales = resize_image(img, resize, fn=fn, interp=interp)
    if crop_coords:
        img, min_rectangle = crop_image(img,crop_coords)
    return normalize_image(img)[None].astype(np.float32), np.asarray(scales)


def crop_image(image, coords) -> np.ndarray:
    """
    根据任意多个点构成的多边形区域裁剪图像。
    将多边形区域外的像素置为0，保留多边形内的图像内容。

    Args:
        image: 输入图像
        coords: 多边形顶点坐标列表，每个元素为(x,y)坐标点，至少需要3个点

    Returns:
        裁剪后的图像，多边形区域外的像素被置为0
    """
    # 检查是否有足够的点构成多边形
    if len(coords) < 3:
        raise ValueError("至少需要3个点才能构成多边形")

    # 转换坐标点为numpy数组
    points = np.array(coords, dtype=np.int32)
    
    # 计算多边形的边界框
    x_min = max(0, np.min(points[:, 0]))
    y_min = max(0, np.min(points[:, 1]))
    x_max = min(image.shape[1], np.max(points[:, 0]))
    y_max = min(image.shape[0], np.max(points[:, 1]))

    min_rectangle = ((x_min, y_min), (x_max, y_max))
    
    # 创建掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # 使用多边形点绘制填充多边形
    cv2.fillPoly(mask, [points], 255)
    
    # 提取包含多边形区域的矩形部分
    cropped = image[y_min:y_max, x_min:x_max].copy()
    mask = mask[y_min:y_max, x_min:x_max]
    
    # 应用掩码
    if len(image.shape) == 3:  # 彩色图像
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)
    cropped[mask == 0] = 0
    
    return cropped, min_rectangle





def calculate_similarity1(kpts0, kpts1, threshold=100):
    """
    计算两组匹配关键点之间的相似度
    参数:
        kpts0: 第一张图片的关键点
        kpts1: 第二张图片的关键点
        threshold: 判断为好的匹配点的距离阈值
    返回:
        float: 0.0-1.0之间的相似度分数
    """
    if len(kpts0) == 0 or len(kpts1) == 0:
        return 0.0
    
    # 计算所有匹配点对之间的欧氏距离
    distances = []
    for pt0, pt1 in zip(kpts0, kpts1):
        dist = math.sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)
        distances.append(dist)
    
    # 统计好的匹配点（距离小于阈值的点）
    good_matches = sum(1 for d in distances if d < threshold)
    
    # 计算相似度分数
    # 考虑匹配点的数量和质量
    num_matches = len(kpts0)
    if num_matches < 10:  # 匹配点太少，认为相似度很低
        base_score = 0.1
    else:
        # 基础分数基于匹配点数量，最高0.7
        base_score = min(0.7, num_matches / 100)
    
    # 质量分数基于好的匹配点占比，最高0.3
    quality_score = (good_matches / num_matches) * 0.3
    
    # 总分 = 基础分数 + 质量分数
    similarity = base_score + quality_score
    
    return min(1.0, max(0.0, similarity))



def calculate_similarity2(m_kpts0, m_kpts1, n_kpts0, n_kpts1):
    similarity = (m_kpts0.shape[0]+m_kpts1.shape[0]) / (n_kpts0 + n_kpts1)
    
    return similarity



def match_images(img_path1, img_path2, output_path, size=1024):
    """
    使用LightGlue在difficult模式下进行图像特征匹配（纯CPU版本）
    参数:
        img_path1: 第一张图片路径
        img_path2: 第二张图片路径
        output_path: 输出图片路径
        size: 输入图片大小，使用较大尺寸以获得更好的匹配效果
    """
    # 设置模型路径
    extractor_type = "disk"  # disk特征提取器在视角变化大的场景下表现更好
    weights_path = Path("weights")
    extractor_path = str(weights_path / f"{extractor_type}.onnx")
    lightglue_path = str(weights_path / f"{extractor_type}_lightglue.onnx")

    crop_coords = [(299,497),(695,235),(918,264),(797,678)]
    # # 加载图片
    # image0, scales0 = load_image(img_path1, resize=size)
    # image1, scales1 = load_image(img_path2, resize=size)
    image0, scales0 = load_image(img_path1,crop_coords)
    image1, scales1 = load_image(img_path2,crop_coords)




    # 初始化模型（只使用CPU）
    providers = ["CPUExecutionProvider"]
    runner = LightGlueRunner(
        extractor_path=extractor_path,
        lightglue_path=lightglue_path,
        providers=providers
    )

    # 运行推理
    m_kpts0, m_kpts1, n_kpts0, n_kpts1 = runner.run(image0, image1, scales0, scales1)

    # 加载原始图像用于可视化
    orig_image0, _ = load_image(img_path1,crop_coords)
    orig_image1, _ = load_image(img_path2,crop_coords)

    # 创建可视化结果
    viz2d.plot_images([
        orig_image0[0].transpose(1, 2, 0),
        orig_image1[0].transpose(1, 2, 0)
    ])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)

    # 保存结果
    viz2d.plt.savefig(output_path, dpi=300, bbox_inches='tight')
    viz2d.plt.close()

    # 计算相似度
    # similarity = calculate_similarity1(m_kpts0, m_kpts1)
    similarity = calculate_similarity2(m_kpts0, m_kpts1, n_kpts0, n_kpts1)

    return m_kpts0, m_kpts1, similarity

if __name__ == "__main__":
    # 使用示例图片进行测试
    img1 = "stand.png"
    img2 = "stand.png"
    output = "matched_result.jpg"

    print("Starting feature matching...")
    m_kpts0, m_kpts1, similarity = match_images(img1, img2, output)
    print(f"Found {len(m_kpts0)} matches.")
    print(f"Image similarity score: {similarity:.3f}")