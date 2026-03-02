import torch
import re
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

_model = None
_processor = None
MODEL_PATH = "/workspace/Qwen2-VL-7B-Instruct"

def _get_model():
    global _model, _processor
    if _model is None:
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16, 
            device_map="cuda",         
            trust_remote_code=True
        ).eval()
        
        _processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return _model, _processor

def verify_video_content(video_path, target_label):
    model, processor = _get_model()

    prompt_content = (
        f"你是一个安防专家。请判断视频中是否发生了行为：【{target_label}】。\n"
        "规则：\n"
        "1. 视频源为监控，画面可能模糊。如果动作特征（如剧烈肢体接触）符合，请判为是。\n"
        "2. 只有完全无关（如正常走路）或无法辨认时，判为否。\n"
        "3. 仅输出一个数字：1 或 0。"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420, 
                    "fps": 2.0,
                },
                {"type": "text", "text": prompt_content},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        match = re.search(r'\d', output_text)
        if match:
            return int(match.group())
        else:
            return 0

    except Exception as e:
        print(f">>>大模型推理出错: {e}")
        return 0