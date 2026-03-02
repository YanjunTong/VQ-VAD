import torch
import torch.nn.functional as F
import numpy as np
import os
import shutil
import cv2
import clip
from PIL import Image
import re
import argparse
import sys
import pprint
from tqdm import tqdm

try:
    from model import CLIPVAD
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import CLIPVAD

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)

CONFIG = {
    'checkpoint_path': os.path.join(project_root,'VADinference', 'checkpoint_strong_final', 'best_model.pth'),
    'temp_dir': os.path.join(project_root, 'VADinference','temp_cmd_batch_cache'), 
    'visual_length': 2048,
    'stride': 4,
    'min_duration': 16,
    
    'feature_model': 'ViT-B/16',
    'clip_path': None, 
}

KNOWN_CLASSES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
    'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Riding',
    'Shoplifting', 'Stealing', 'Vandalism', 'Normal', 'Fire', 'Rotate',
    'Squat', 'Umbrella', 'Falling'
]

CLASS_THRESHOLDS = {
    'Arrest': 0.75,
    'Arson': 0.2,
    'Burglary': 0.8,
    'Explosion': 0.75,
    'Fighting': 0.75,
    'RoadAccidents': 0.51,
    'Shooting': 0.5,
    'Robbery': 0.8,
    'Shoplifting': 0.5,
    'Stealing': 0.5,
    'Falling': 0.4
}
DEFAULT_THRESHOLD = 0.4

DETAILED_DESCRIPTIONS = {
    'Normal': "normal surveillance scene with no abnormal activity",
    'Abuse': "cruel treatment or violence against another person, domestic violence",
    'Arrest': "police officers arresting a person, handcuffing suspect, wearing uniform",
    'Arson': "setting fire to property, huge fire and smoke, burning",
    'Assault': "sudden physical attack, hitting or punching someone aggressively",
    'Burglary': "breaking into a house or building to steal, entering through window",
    'Explosion': "sudden explosion with fire and smoke and debris",
    'Fighting': "people fighting each other, mutual physical combat, kicking and punching",
    'RoadAccidents': "car crash, vehicle collision on the road, car hitting person",
    'Robbery': "stealing from a person using force or threat, snatching bag",
    'Shooting': "person shooting with a gun, gunfire flash, holding weapon",
    'Shoplifting': "stealing goods from a shop shelf secretly",
    'Stealing': "theft of property without force",
    'Vandalism': "damaging public or private property, smashing things, breaking glass",
    'Riding': "person riding a bicycle or motorcycle",
    'Fire': "large fire outbreak, flames and smoke",
    'Rotate': "person rotating or spinning around",
    'Squat': "person squatting down close to the ground",
    'Umbrella': "person using an umbrella, opening or closing umbrella",
    'Falling': "person falling down to the ground, losing balance and collapsing"
}

def construct_all_class_prompts():
    display_names = ["Normal"]
    full_prompts = []
    target_indices = []

    full_prompts.append(f"A surveillance video showing {DETAILED_DESCRIPTIONS['Normal']}.")
    
    current_idx = 1
    for known_class in KNOWN_CLASSES:
        if known_class == 'Normal':
            continue
            
        display_names.append(known_class)
        target_indices.append(current_idx) 
        
        desc = DETAILED_DESCRIPTIONS.get(known_class, known_class)
        full_prompts.append(f"A surveillance video showing {desc}.")
        current_idx += 1
    
    return display_names, full_prompts, target_indices

def extract_feature_list(video_paths, temp_dir, device):
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    generated_files = []
    
    for i, video_path in enumerate(video_paths):
        save_name = os.path.splitext(os.path.basename(video_path))[0] + ".npy"
        save_path = os.path.join(temp_dir, save_name)
        
        if os.path.exists(save_path): 
            generated_files.append(save_path)
            continue

        try:
            model_target = CONFIG.get('clip_path')
            if not model_target:
                model_target = CONFIG['feature_model']
                
            clip_model, preprocess = clip.load(model_target, device=device)
            clip_model.eval()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): 
                print(f"[Error] 无法打开视频: {video_path}")
                continue
            
            frames = []
            cnt = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if cnt % CONFIG['stride'] == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    frames.append(preprocess(img))
                cnt += 1
            cap.release()
            
            if not frames: 
                continue
            
            tensor = torch.stack(frames).to(device)
            feat_list = []
            batch_size = 16 
            
            with torch.no_grad():
                for j in range(0, len(tensor), batch_size):
                    batch = tensor[j:j+batch_size]
                    feat = clip_model.encode_image(batch)
                    feat_list.append(feat.cpu().numpy())
            
            features = np.concatenate(feat_list, axis=0)
            np.save(save_path, features.astype(np.float32))
            generated_files.append(save_path)
            
            del clip_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[Error] 提取失败 {video_path}: {e}")
            
    return generated_files

def process_split(features, length):
    if features.shape[0] > length:
        features = features[:length]
        valid_len = length
    else:
        valid_len = features.shape[0]
        pad_len = length - valid_len
        if pad_len > 0:
            padding = np.zeros((pad_len, features.shape[1]))
            features = np.concatenate((features, padding), axis=0)
    return features, valid_len

def load_vad_model(num_classes, device):
    embed_dim = 768 if 'ViT-L' in CONFIG['feature_model'] else 512
    visual_head = 12 if embed_dim == 768 else 4
    
    model = CLIPVAD(
        num_class=num_classes, 
        embed_dim=embed_dim, 
        visual_length=CONFIG['visual_length'],
        visual_width=embed_dim, 
        visual_head=visual_head, 
        visual_layers=2, 
        attn_window=16, prompt_prefix=16, prompt_postfix=16, 
        device=device
    )
    
    if not os.path.exists(CONFIG['checkpoint_path']):
        raise FileNotFoundError(f"权重文件不存在: {CONFIG['checkpoint_path']}")
    
    checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    return model

def _run_batch_logic(feat_files, display_names, prompts, target_indices, device):
    events_dict = {}
    model = load_vad_model(len(prompts), device)
    
    for feat_path in tqdm(feat_files, desc="Inference"):
        try:
            features = np.load(feat_path)
        except: continue
        
        features, valid_len = process_split(features, CONFIG['visual_length'])
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        len_tensor = torch.tensor([valid_len]).to(device)
        
        with torch.no_grad():
            _, logits1, logits2 = model(feat_tensor, None, prompts, len_tensor)
            
        anomaly_scores = torch.sigmoid(logits1[0, :valid_len, 0]).cpu().numpy()
        text_logits = logits2[0, :valid_len, :] 
        best_indices = torch.argmax(text_logits, dim=-1).cpu().numpy()
        
        raw_candidates = []
        for t in range(valid_len):
            pred_idx = best_indices[t]
            score = float(anomaly_scores[t])
            
            if pred_idx in target_indices:
                class_name = display_names[pred_idx]
                
                threshold = DEFAULT_THRESHOLD
                for key, val in CLASS_THRESHOLDS.items():
                    if key.lower() == class_name.lower():
                        threshold = val
                        break
                
                if score >= threshold:
                    raw_candidates.append({
                        'frame_idx': t * CONFIG['stride'],
                        'class_name': class_name,
                        'score': score
                    })

        if not raw_candidates:
            continue
            
        current_event = None
        MAX_GAP = CONFIG['stride'] * 1 
        
        for cand in raw_candidates:
            start_f = cand['frame_idx']
            end_f = start_f + CONFIG['stride']
            cls = cand['class_name']
            sc = cand['score']
            
            if current_event is None:
                current_event = {
                    'label': cls,
                    'start': start_f,
                    'end': end_f,
                    'scores': [sc]
                }
            else:
                time_diff = start_f - current_event['end']
                if cls == current_event['label'] and time_diff <= MAX_GAP:
                    current_event['end'] = end_f 
                    current_event['scores'].append(sc)
                else:
                    avg_conf = np.mean(current_event['scores'])
                    interval = [current_event['start'], current_event['end'], float(avg_conf)]
                    
                    lbl = current_event['label']
                    if lbl not in events_dict:
                        events_dict[lbl] = []
                    events_dict[lbl].append(interval)
                    
                    current_event = {
                        'label': cls,
                        'start': start_f,
                        'end': end_f,
                        'scores': [sc]
                    }
        
        if current_event is not None:
            avg_conf = np.mean(current_event['scores'])
            interval = [current_event['start'], current_event['end'], float(avg_conf)]
            lbl = current_event['label']
            if lbl not in events_dict:
                events_dict[lbl] = []
            events_dict[lbl].append(interval)

    for class_name in events_dict:
        events_dict[class_name].sort(key=lambda x: x[2], reverse=True)
        events_dict[class_name] = events_dict[class_name][:4]

    return events_dict

def run_vad_inference(video_path=None, folder_path=None, text_query=None, text=None, clip_path=None):
    if clip_path:
        CONFIG['clip_path'] = clip_path

    if video_path and folder_path:
        raise ValueError("只能同时指定 video_path 或 folder_path 之一")
    if not video_path and not folder_path:
        raise ValueError("必须指定 video_path 或 folder_path")

    video_list = []
    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")
        video_list.append(video_path)
    else:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        valid_exts = ('.mp4', '.avi', '.mkv', '.mov')
        for f in os.listdir(folder_path):
            if f.lower().endswith(valid_exts):
                video_list.append(os.path.join(folder_path, f))
        if not video_list:
            raise ValueError("文件夹中未找到视频文件")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    feat_files = extract_feature_list(video_list, CONFIG['temp_dir'], device)
    if not feat_files:
        return {}
        
    display_names, prompts, target_indices = construct_all_class_prompts()
    
    if not prompts:
        raise ValueError("无法构建 Prompts")

    final_results = _run_batch_logic(feat_files, display_names, prompts, target_indices, device)
    
    if os.path.exists(CONFIG['temp_dir']):
        try: shutil.rmtree(CONFIG['temp_dir'])
        except: pass
        
    return final_results

def main():
    parser = argparse.ArgumentParser(description="VadCLIP 通用召回模式")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str, help="单个视频路径")
    group.add_argument('--path', type=str, help="文件夹路径")
    parser.add_argument('--text', type=str, default=None, help="[已弃用] 检测目标")
    parser.add_argument('--output', type=str, default='./result.txt', help="结果保存路径")
    parser.add_argument('--clip_model', type=str, default=None, help="指定本地 CLIP 模型路径 (e.g. /workspace/ViT-B-16.pt)")
    
    args = parser.parse_args()

    try:
        results = run_vad_inference(
            video_path=args.video, 
            folder_path=args.path, 
            text=args.text,
            clip_path=args.clip_model
        )
        
        print("\n" + "="*20 + " 检测结果预览 " + "="*20)
        pprint.pprint(results)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(str(results))
        print(f"\n结果已保存至: {args.output}")
        
    except Exception as e:
        print(f"执行出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()