from dataclasses import dataclass
from typing import Type

from model_hub.model_classes import ModelBuilderBase, HFModelBuilder, HFModelBuilder4bit


@dataclass
class ModelMetainfo:
    builder: Type[ModelBuilderBase]
    checkpoint: str


MODEL_REGISTRY = {
    'starcoderbase-1b': ModelMetainfo(builder=HFModelBuilder, checkpoint="bigcode/starcoderbase-1b"),
    'starcoderbase-3b': ModelMetainfo(builder=HFModelBuilder, checkpoint="bigcode/starcoderbase-3b"),
    'starcoderbase-7b': ModelMetainfo(builder=HFModelBuilder, checkpoint="bigcode/starcoderbase-7b"),
    'starcoderbase': ModelMetainfo(builder=HFModelBuilder, checkpoint="bigcode/starcoderbase"),

    'starcoderbase-1b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="bigcode/starcoderbase-1b"),
    'starcoderbase-3b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="bigcode/starcoderbase-3b"),
    'starcoderbase-7b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="bigcode/starcoderbase-7b"),
    'starcoderbase-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="bigcode/starcoderbase"),

    'codellama-7b': ModelMetainfo(builder=HFModelBuilder, checkpoint="codellama/CodeLlama-7b-hf"),
    'codellama-13b': ModelMetainfo(builder=HFModelBuilder, checkpoint="codellama/CodeLlama-13b-hf"),
    'codellama-34b': ModelMetainfo(builder=HFModelBuilder, checkpoint="codellama/CodeLlama-34b-hf"),

    'codellama-7b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="codellama/CodeLlama-7b-hf"),
    'codellama-13b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="codellama/CodeLlama-13b-hf"),
    'codellama-34b-4bit': ModelMetainfo(builder=HFModelBuilder4bit, checkpoint="codellama/CodeLlama-34b-hf"),
}