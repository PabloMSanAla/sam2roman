"""
SAM 2 Model Wrapper for Hugging Face Transformers and PyTorch.
"""

from __future__ import annotations

import logging
from typing import Optional, Union, Any

import torch
from transformers import (
    AutoModel,
    AutoProcessor,
    pipeline,
    Pipeline,
)

logger = logging.getLogger(__name__)

DEFAULT_SAM2_MODEL = "facebook/sam2.1-hiera-tiny"


def get_optimal_device() -> torch.device:
    """
    Detect the most performant available PyTorch accelerator:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon Metal Performance Shaders)
    3. CPU (Fallback)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SAM2ModelWrapper:
    """
    Unified loader and abstraction for Hugging Face SAM 2 models.

    Parameters
    ----------
    model_id : str, default 'facebook/sam2.1-hiera-tiny'
        Hugging Face Hub model repository or local checkpoint path.
    device : str or torch.device, optional
        Target execution device ('cuda', 'mps', 'cpu'). Auto-detected if None.
    torch_dtype : torch.dtype, optional
        Data precision (e.g., torch.bfloat16, torch.float32). Defaults to float32 on CPU/MPS.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_SAM2_MODEL,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_id = model_id
        if device is None:
            self.device = get_optimal_device()
        else:
            self.device = torch.device(device)

        if torch_dtype is None:
            # MPS and CUDA can benefit from bfloat16 or float32; float32 is safest default
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._mask_generator: Optional[Pipeline] = None

    @property
    def model(self) -> Any:
        """Lazy-loaded PyTorch SAM 2 model instance."""
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def processor(self) -> Any:
        """Lazy-loaded Hugging Face SAM 2 processor."""
        if self._processor is None:
            self.load_processor()
        return self._processor

    def load_model(self) -> Any:
        """Load the PyTorch SAM 2 model to the target device."""
        logger.info(f"Loading SAM 2 model '{self.model_id}' onto {self.device}...")
        try:
            # Attempt loading via specialized Sam2Model if available in transformers
            from transformers import Sam2Model  # type: ignore
            model_cls = Sam2Model
        except (ImportError, AttributeError):
            # Fallback to AutoModel
            model_cls = AutoModel

        self._model = model_cls.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
        )
        self._model.to(self.device)
        self._model.eval()
        return self._model

    def load_processor(self) -> Any:
        """Load the SAM 2 processor for input/output formatting."""
        logger.info(f"Loading SAM 2 processor for '{self.model_id}'...")
        try:
            from transformers import Sam2Processor  # type: ignore
            proc_cls = Sam2Processor
        except (ImportError, AttributeError):
            proc_cls = AutoProcessor

        self._processor = proc_cls.from_pretrained(self.model_id)
        return self._processor

    def get_mask_generator_pipeline(
        self,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.85,
        **kwargs: Any,
    ) -> Pipeline:
        """
        Instantiate or return the Hugging Face automatic mask generation pipeline.

        Parameters
        ----------
        points_per_batch : int, default 64
            Number of points per forward pass for grid sampling.
        pred_iou_thresh : float, default 0.7
            Minimum predicted IoU threshold for keeping a mask.
        stability_score_thresh : float, default 0.85
            Mask stability threshold.
        """
        if self._mask_generator is None:
            # Setup device index or string for transformers pipeline
            if self.device.type == "cuda":
                pipeline_device = 0
            elif self.device.type == "mps":
                pipeline_device = "mps"
            else:
                pipeline_device = "cpu"

            try:
                self._mask_generator = pipeline(
                    task="mask-generation",
                    model=self.model_id,
                    device=pipeline_device,
                    torch_dtype=self.torch_dtype,
                    points_per_batch=points_per_batch,
                    pred_iou_thresh=pred_iou_thresh,
                    stability_score_thresh=stability_score_thresh,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(
                    f"Direct pipeline('mask-generation') setup encountered: {e}. "
                    "Inference will fall back to model + processor grid prompting."
                )
                self._mask_generator = None

        return self._mask_generator
