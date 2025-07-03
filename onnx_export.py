from export import export_onnx



export_onnx(
        img_size=1024,
        extractor_type="disk",
        dynamic=True,  # 支持动态图片尺寸
        max_num_keypoints=None,  # 不限制特征点数量
    )
print("Models exported successfully.")