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

    'starcoder2-3b': ModelMetainfo(builder=HFModelBuilder, checkpoint="bigcode/starcoder2-3b"),

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

    'deepseek-coder-1b': ModelMetainfo(builder=HFModelBuilder, checkpoint="deepseek-ai/deepseek-coder-1.3b-base"),
    'deepseek-coder-7b': ModelMetainfo(builder=HFModelBuilder, checkpoint="deepseek-ai/deepseek-coder-6.7b-base"),

    'granite-3b-code-base': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-3b-code-base"),
    'granite-3b-code-base-128k': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-3b-code-base-128k"),
    'granite-3b-code-instruct': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-3b-code-instruct"),
    'granite-3b-code-instruct-128k': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-3b-code-instruct-128k"),
    'granite-8b-code-base': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-8b-code-base"),
    'granite-8b-code-base-128k': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-8b-code-base-128k"),
    'granite-8b-code-instruct': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-8b-code-instruct"),
    'granite-8b-code-instruct-128k': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-8b-code-instruct-128k"),
    'granite-20b-code-base': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-20b-code-base"),
    'granite-20b-code-instruct': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-20b-code-instruct"),
    'granite-34b-code-base': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-34b-code-base"),
    'granite-34b-code-instruct': ModelMetainfo(builder=HFModelBuilder, checkpoint="ibm-granite/granite-34b-code-instruct"),
}
