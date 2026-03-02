import os
import shutil
import gc
import torch
import sys
import ffmpeg 
from vad_detect import run_vad_inference
from mllm import verify_video_content
#~/.cache
ANNOTATION_FILE = "/workspace/1/video_annotations.txt" 
VIDEO_FOLDER = "/workspace/1/video"              
OUTPUT_TXT = "result.txt"
TEMP_ROOT = '/workspace/VADinference/batch_temp'

def load_annotations(txt_path):
    annotations = {}
    if not os.path.exists(txt_path):
        print(f"[Error] File not found: {txt_path}")
        return annotations

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                video_key = os.path.splitext(parts[0])[0]
                annotations[video_key] = parts[1]
    return annotations

def cut_video_clips(video_path, vad_results, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for label, segments in vad_results.items():
        for i, seg in enumerate(segments):
            start_f = int(seg[0])
            end_f = int(seg[1])
            
            save_name = f"{label}_{start_f}_{end_f}.mp4"
            save_path = os.path.join(output_dir, save_name)
            
            start_t = start_f / 30.0
            end_t = end_f / 30.0
            
            try:
                (
                    ffmpeg
                    .input(video_path, ss=start_t, to=end_t)
                    .output(save_path, c='copy', loglevel='quiet')
                    .overwrite_output()
                    .run()
                )
            except Exception as e:
                print(f"  [Warn] 切片失败 {save_name}: {e}")

def run_task_driven_pipeline():
    task_map = load_annotations(ANNOTATION_FILE)
    
    if not task_map:
        print("未加载到任务描述，请检查 ANNOTATION_FILE 路径。")
        return

    target_videos = list(task_map.keys())
    target_videos.sort()

    processed_videos_map = {} 

    if os.path.exists(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT)
    os.makedirs(TEMP_ROOT)

    for i, video_name in enumerate(target_videos):
        video_path = os.path.join(VIDEO_FOLDER, video_name + ".mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(VIDEO_FOLDER, video_name + ".avi")
            
        if not os.path.exists(video_path):
            print(f"[{i+1}] Video not found: {video_name}")
            continue
        
        try:
            vad_result = run_vad_inference(video_path=video_path) 
            
            if vad_result:
                video_sub_dir = os.path.join(TEMP_ROOT, video_name)
                
                cut_video_clips(video_path, vad_result, video_sub_dir)
                
                processed_videos_map[video_name] = video_sub_dir
                
        except Exception as e:
            print(f"    -> [Error] VAD阶段出错: {e}")
    gc.collect()
    torch.cuda.empty_cache()

    
    if os.path.exists(OUTPUT_TXT):
        os.remove(OUTPUT_TXT)

    for i, video_name in enumerate(target_videos):
        specific_description = task_map.get(video_name, "abnormal event")
        
        final_segments = []
        
        if video_name in processed_videos_map:
            temp_dir = processed_videos_map[video_name]
            
            clips = [
                os.path.join(temp_dir, c) 
                for c in os.listdir(temp_dir) 
                if c.endswith('.mp4')
            ]
            
            for clip_path in clips:
                score = verify_video_content(video_path=clip_path, target_label=specific_description)
                
                if score == 1:
                    try:
                        base = os.path.splitext(os.path.basename(clip_path))[0]
                        parts = base.rsplit('_', 2)
                        if len(parts) == 3:
                            start_f = int(parts[1])
                            end_f = int(parts[2])
                            final_segments.append([start_f, end_f])
                    except:
                        pass

        with open(OUTPUT_TXT, "a", encoding="utf-8") as f:
            if not final_segments:
                output_str = f"{video_name} 0 -1"
            else:
                final_segments.sort(key=lambda x: x[0])
                
                top_segments = final_segments[:4]
                
                seg_str_list = []
                for s, e in top_segments:
                    seg_str_list.append(str(s))
                    seg_str_list.append(str(e))
                
                output_str = f"{video_name} 1 {' '.join(seg_str_list)} -1"

            f.write(output_str + "\n")

    print(f"result.txt is saved at {os.path.abspath(OUTPUT_TXT)}")

if __name__ == "__main__":
    if not os.path.exists(VIDEO_FOLDER):
        print(f"错误: 视频文件夹不存在 {VIDEO_FOLDER}")
    else:
        run_task_driven_pipeline()