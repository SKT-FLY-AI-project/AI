import torch
from transformers import (
    AutoModelForVision2Seq, AutoProcessor,
    AutoModelForCausalLM, AutoTokenizer
)
from concurrent.futures import ThreadPoolExecutor

# ✅ 모델 로드 시 ThreadPoolExecutor 활용 (멀티스레딩)
executor = ThreadPoolExecutor(max_workers=2)

# ✅ 모델 이름 설정
VL_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
CODER_MODEL_NAME = "Qwen/Qwen2.5-Coder-32B"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔄 Qwen2.5-VL 및 Qwen2.5-Coder-32B 모델 로딩 중...")

# ✅ Qwen2.5-VL (이미지 설명)
vlm_model = AutoModelForVision2Seq.from_pretrained(
    VL_MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
vlm_processor = AutoProcessor.from_pretrained(VL_MODEL_NAME)

# ✅ Qwen2.5-Coder-32B (텍스트 생성)
coder_model = AutoModelForCausalLM.from_pretrained(
    CODER_MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
coder_tokenizer = AutoTokenizer.from_pretrained(CODER_MODEL_NAME)

print("✅ Qwen2.5-VL 및 Qwen2.5-Coder-32B 모델 로딩 완료!")

# ✅ 공용 모델 반환 함수
def get_vlm_model():
    """ Qwen2.5-VL Vision-Language 모델 반환 """
    return vlm_model, vlm_processor

def get_coder_model():
    """ Qwen2.5-Coder-32B 모델 반환 """
    return coder_model, coder_tokenizer
