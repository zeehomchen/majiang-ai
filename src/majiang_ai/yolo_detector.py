"""
YOLO11物体检测模块
支持手牌和副露的端到端识别
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


class TileState(Enum):
    """牌的状态"""
    HAND = "hand"  # 手牌
    MELD = "meld"  # 副露


class TileSuit(Enum):
    """牌的花色"""
    WAN = "m"
    TONG = "p"
    TIAO = "s"
    ZI = "z"


# 麻将牌列表（34种）
ALL_TILES = [
    # 万子 1-9
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    # 筒子 1-9
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    # 条子 1-9
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    # 字牌 东南西北白发中
    "1z", "2z", "3z", "4z", "5z", "6z", "7z",
]


# 生成68类标签（34 × 2种状态）
CLASS_NAMES = []
for tile in ALL_TILES:
    CLASS_NAMES.append(f"hand_{tile}")
for tile in ALL_TILES:
    CLASS_NAMES.append(f"meld_{tile}")


@dataclass
class YOLOTileDetection:
    """YOLO检测结果"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    tile_code: str  # 牌的编码如 "1m"
    tile_state: TileState
    confidence: float
    class_name: str  # 完整类别名


@dataclass
class YOLOInferenceResult:
    """YOLO推理结果"""
    tiles: List[YOLOTileDetection]
    inference_time_ms: float
    input_shape: Tuple[int, int]


class YOLO11Detector:
    """
    YOLO11麻将牌检测器
    支持68类检测（34种牌 × 手牌/副露状态
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        use_onnx: bool = True,
        device: str = "cpu",
        debug: bool = False
    ):
        """
        初始化检测器

        Args:
            model_path: 模型文件路径（.pt或.onnx）
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            use_onnx: 是否使用ONNX推理
            device: 设备 ('cpu' 或 'cuda')
            debug: 是否调试模式
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_onnx = use_onnx
        self.device = device
        self.debug = debug

        self._model: Optional[YOLO] = None
        self._onnx_session: Optional[ort.InferenceSession] = None
        self._class_names = CLASS_NAMES
        self._input_size = (640, 640)

    def load_model(self) -> bool:
        """
        加载模型

        Returns:
            是否成功
        """
        if self.model_path is None:
            # 没有模型路径，使用预训练模型
            print("警告: 未指定模型路径")
            return False

        model_path = Path(self.model_path)

        if not model_path.exists():
            print(f"错误: 模型文件不存在: {model_path}")
            return False

        try:
            if self.use_onnx and model_path.suffix == ".onnx":
                # ONNX模型
                if not ONNXRUNTIME_AVAILABLE:
                    print("警告: onnxruntime未安装，降级为PyTorch模式")
                    self.use_onnx = False
                else:
                    self._load_onnx_model(model_path)
                    return True

            # PyTorch模型
            if not ULTRALYTICS_AVAILABLE:
                print("错误: ultralytics未安装")
                return False

            self._model = YOLO(str(model_path))
            print(f"✅ YOLO模型加载成功: {model_path}")
            return True

        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def _load_onnx_model(self, model_path: Path) -> None:
        """加载ONNX模型"""
        providers = ['CPUExecutionProvider']
        if self.device == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self._onnx_session = ort.InferenceSession(str(model_path), providers=providers)
        print(f"✅ ONNX模型加载成功: {model_path}")

    def predict(self, image: np.ndarray) -> YOLOInferenceResult:
        """
        进行预测

        Args:
            image: 输入图像（BGR格式）

        Returns:
            YOLOInferenceResult
        """
        start_time = time.time()

        if self._onnx_session is not None:
            detections = self._predict_onnx(image)
        elif self._model is not None:
            detections = self._predict_yolo(image)
        else:
            detections = []

        inference_time_ms = (time.time() - start_time) * 1000

        return YOLOInferenceResult(
            tiles=detections,
            inference_time_ms=inference_time_ms,
            input_shape=image.shape[:2]
        )

    def _predict_yolo(self, image: np.ndarray) -> List[YOLOTileDetection]:
        """使用YOLO进行预测"""
        results = self._model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=self.debug
        )

        detections: List[YOLOTileDetection] = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    if cls_id < len(self._class_names):
                        class_name = self._class_names[cls_id]
                        tile_code, tile_state = self._parse_class_name(class_name)

                        detections.append(YOLOTileDetection(
                            bbox=(x1, y1, x2, y2),
                            tile_code=tile_code,
                            tile_state=tile_state,
                            confidence=conf,
                            class_name=class_name
                        ))

        return detections

    def _predict_onnx(self, image: np.ndarray) -> List[YOLOTileDetection]:
        """使用ONNX进行预测（简化版）"""
        # 预处理
        input_tensor = self._preprocess(image)

        # 推理
        outputs = self._onnx_session.run(None, {self._onnx_session.get_inputs()[0].name: input_tensor})

        # 后处理（需要根据具体ONNX模型输出格式调整）
        detections = self._postprocess_onnx(outputs, image.shape)

        return detections

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 调整大小
        resized = cv2.resize(image, self._input_size)
        # BGR -> RGB
        rgb = resized[:, :, ::-1]
        # 归一化
        normalized = rgb.astype(np.float32) / 255.0
        # 通道优先 (H, W, C) -> (1, C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))
        # 添加batch维度
        input_tensor = np.expand_dims(transposed, axis=0)
        return input_tensor

    def _postprocess_onnx(
        self,
        outputs: List[np.ndarray],
        original_shape: Tuple[int, int]
    ) -> List[YOLOTileDetection]:
        """ONNX输出后处理（占位实现，需要根据具体模型调整"""
        detections: List[YOLOTileDetection] = []
        # 这里需要根据具体ONNX模型的输出格式实现后处理
        return detections

    @staticmethod
    def _parse_class_name(class_name: str) -> Tuple[str, TileState]:
        """
        解析类别名

        Args:
            class_name: 类别名如 "hand_1m" 或 "meld_5p"

        Returns:
            (tile_code, TileState)
        """
        if class_name.startswith("hand_"):
            tile_code = class_name[5:]
            state = TileState.HAND
        elif class_name.startswith("meld_"):
            tile_code = class_name[5:]
            state = TileState.MELD
        else:
            return class_name, TileState.HAND

        return tile_code, state

    def export_to_onnx(
        self,
        pt_model_path: str,
        output_path: Optional[str] = None,
        imgsz: int = 640,
        simplify: bool = True
    ) -> str:
        """
        导出PyTorch模型为ONNX

        Args:
            pt_model_path: PyTorch模型路径
            output_path: 输出ONNX路径
            imgsz: 输入图像尺寸
            simplify: 是否简化模型

        Returns:
            输出路径
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("需要ultralytics: pip install ultralytics")

        if output_path is None:
            output_path = Path(pt_model_path).with_suffix(".onnx")

        model = YOLO(pt_model_path)
        model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=simplify,
            opset=12
        )

        print(f"✅ 模型已导出为ONNX: {output_path}")
        return str(output_path)

    def visualize(
        self,
        image: np.ndarray,
        result: YOLOInferenceResult
    ) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image: 原始图像
            result: 检测结果

        Returns:
            标注后的图像
        """
        vis = image.copy()

        for tile in result.tiles:
            x1, y1, x2, y2 = tile.bbox

            # 选择颜色
            if tile.tile_state == TileState.HAND:
                color = (0, 255, 0)  # 绿色 - 手牌
            else:
                color = (0, 0, 255)  # 红色 - 副露

            # 绘制bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{tile.tile_code} {tile.confidence:.2f}"
            cv2.putText(
                vis, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # 绘制推理时间
        cv2.putText(
            vis, f"{result.inference_time_ms:.1f}ms", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        return vis

    @staticmethod
    def generate_class_names_file(output_path: str = "class_names.txt") -> None:
        """
        生成类别名文件（用于训练时标注

        Args:
            output_path: 输出文件路径
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for name in CLASS_NAMES:
                f.write(f"{name}\n")
        print(f"✅ 类别名文件已生成: {output_path}")


def create_training_template() -> YOLO11Detector:
    """
    创建用于训练的模板检测器

    Returns:
        YOLO11Detector实例
    """
    detector = YOLO11Detector(
        model_path=None,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    # 生成类别名文件
    YOLO11Detector.generate_class_names_file()
    return detector
