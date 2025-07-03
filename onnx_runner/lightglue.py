import numpy as np
import onnxruntime as ort


def lqy_filter_below_threshold(arr, threshold):
    """
    过滤numpy数组中小于指定阈值的元素，返回需要过滤的元素的索引
    
    参数:
    arr: numpy数组
    threshold: 阈值
    
    返回:
    filtered_indices: 小于阈值的元素的索引数组
    """
    # 找到小于阈值的元素的索引
    filtered_indices = np.where(arr < threshold)[0]
    return filtered_indices


def lqy_normalize_and_filter(kpts_arr,desc_arr,scores_arr, threshold):
    """
    对数组进行最大值归一化，并返回小于阈值的元素索引
    
    参数:
    arr: numpy数组，维度为(1, 2501)
    threshold: 阈值，用于过滤元素
    
    返回:
    normalized_arr: 归一化后的数组
    indices: 小于阈值的元素索引
    """
    # 找到数组中的最大值
    max_value = np.max(scores_arr)
    
    # 避免除零错误
    if max_value == 0:
        print("警告：数组最大值为0，无法进行归一化")
        return kpts_arr,desc_arr
    
    # 根据最大值进行归一化
    normalized_arr = scores_arr[0] / max_value
    
    # 找到小于阈值的元素索引
    indices = np.where(normalized_arr < threshold)[0]
    scores_arr = np.delete(scores_arr, indices,axis=1)[0]
    kpts_arr = np.delete(kpts_arr, indices, axis=1)
    desc_arr = np.delete(desc_arr, indices, axis=1)
    
    return kpts_arr, scores_arr, desc_arr






class LightGlueRunner:
    def __init__(
        self,
        lightglue_path: str,
        extractor_path=None,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        self.extractor = (
            ort.InferenceSession(
                extractor_path,
                providers=providers,
            )
            if extractor_path is not None
            else None
        )
        sess_options = ort.SessionOptions()
        self.lightglue = ort.InferenceSession(
            lightglue_path, sess_options=sess_options, providers=providers
        )

        # Check for invalid models.
        lightglue_inputs = [i.name for i in self.lightglue.get_inputs()]
        if self.extractor is not None and "image0" in lightglue_inputs:
            raise TypeError(
                f"The specified LightGlue model at {lightglue_path} is end-to-end. Please do not pass the extractor_path argument."
            )
        elif self.extractor is None and "image0" not in lightglue_inputs:
            raise TypeError(
                f"The specified LightGlue model at {lightglue_path} is not end-to-end. Please pass the extractor_path argument."
            )

    def run(self, image0: np.ndarray, image1: np.ndarray, scales0, scales1):
        kpts_threshold = 0.5
        if self.extractor is None:
            kpts0, kpts1, matches0, mscores0 = self.lightglue.run(
                None,
                {
                    "image0": image0,
                    "image1": image1,
                },
            )
            m_kpts0, m_kpts1 = self.post_process(
                kpts0, kpts1, matches0, scales0, scales1
            )
            return m_kpts0, m_kpts1
        else:
            kpts0, scores0, desc0 = self.extractor.run(None, {"image": image0})
            kpts_arr0, scores_arr0, desc_arr0= lqy_normalize_and_filter(kpts_arr=kpts0,desc_arr=desc0,scores_arr=scores0, threshold=kpts_threshold)
            kpts1, scores1, desc1 = self.extractor.run(None, {"image": image1})
            kpts_arr1, scores_arr1, desc_arr1 = lqy_normalize_and_filter(kpts_arr=kpts1,desc_arr=desc1,scores_arr=scores1, threshold=kpts_threshold)

            matches0, mscores0 = self.lightglue.run(
                None,
                {
                    "kpts0": self.normalize_keypoints(
                        kpts_arr0, image0.shape[2], image0.shape[3]
                    ),
                    "kpts1": self.normalize_keypoints(
                        kpts_arr1, image1.shape[2], image1.shape[3]
                    ),
                    "desc0": desc_arr0,
                    "desc1": desc_arr1,
                },
            )

            # indices = lqy_filter_below_threshold(mscores0, 0.5) # 过滤掉小于0.5的匹配
            # matches0 = np.delete(matches0, indices,axis=0) # 删除小于0.5的匹配
            m_kpts0, m_kpts1 = self.post_process(kpts_arr0, kpts_arr1, matches0, scales0, scales1)

            return m_kpts0, m_kpts1, kpts_arr0.shape[1], kpts_arr1.shape[1]

    @staticmethod
    def normalize_keypoints(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
        size = np.array([w, h])
        shift = size / 2
        scale = size.max() / 2
        kpts = (kpts - shift) / scale
        return kpts.astype(np.float32)

    @staticmethod
    def post_process(kpts0, kpts1, matches, scales0, scales1):
        kpts0 = (kpts0 + 0.5) / scales0 - 0.5
        kpts1 = (kpts1 + 0.5) / scales1 - 0.5
        # create match indices
        m_kpts0, m_kpts1 = kpts0[0][matches[..., 0]], kpts1[0][matches[..., 1]]
        return m_kpts0, m_kpts1
