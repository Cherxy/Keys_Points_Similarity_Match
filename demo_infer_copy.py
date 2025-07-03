import cv2
import numpy as np
from pathlib import Path
from onnx_runner import LightGlueRunner, viz2d

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
    grayscale: bool = False,
    resize: int = None,
    fn: str = "max",
    interp: str = "area",
):
    img = read_image(path, grayscale=grayscale)
    scales = [1, 1]
    if resize is not None:
        img, scales = resize_image(img, resize, fn=fn, interp=interp)
    return normalize_image(img)[None].astype(np.float32), np.asarray(scales)





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

    # 加载图片
    image0, scales0 = load_image(img_path1, resize=size)
    image1, scales1 = load_image(img_path2, resize=size)

    # 初始化模型（只使用CPU）
    providers = ["CPUExecutionProvider"]
    runner = LightGlueRunner(
        extractor_path=extractor_path,
        lightglue_path=lightglue_path,
        providers=providers
    )

    # 运行推理
    m_kpts0, m_kpts1 = runner.run(image0, image1, scales0, scales1)

    # 加载原始图像用于可视化
    orig_image0, _ = load_image(img_path1)
    orig_image1, _ = load_image(img_path2)

    # 创建可视化结果
    viz2d.plot_images([
        orig_image0[0].transpose(1, 2, 0),
        orig_image1[0].transpose(1, 2, 0)
    ])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)

    # 保存结果
    viz2d.plt.savefig(output_path, dpi=300, bbox_inches='tight')
    viz2d.plt.close()

    return m_kpts0, m_kpts1

if __name__ == "__main__":
    # 使用示例图片进行测试
    img1 = "assets/54omoeq9y6d873r88zxclq3vw_0.jpg"
    img2 = "assets/74xpeiee8ysxkx2ikk5q0tx3t_0.jpg"
    output = "matched_result.jpg"

    print("Starting feature matching...")
    m_kpts0, m_kpts1 = match_images(img1, img2, output)
    print(f"Found {len(m_kpts0)} matches.")
    print(f"Results saved to {output}")
