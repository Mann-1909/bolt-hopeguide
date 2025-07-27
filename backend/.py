from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_model_name = "openchat/openchat_3.5"
lora_dir = "/workspace/Bolt/project/backend/openchat-phq9-lora/checkpoint-564"

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    logger.info("Base model loaded")
    model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
        torch_dtype=torch.float16,
        is_trainable=False,
        device_map="auto",
    )
    logger.info("PEFT model loaded")
except Exception as e:
    logger.error(f"Error: {str(e)}")
    logger.error(traceback.format_exc())