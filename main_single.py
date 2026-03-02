import sys
import os
import shutil
import gc
import torch
from vad_detect import run_vad_inference
from videocut import copy_video
from mllm import verify_video_content

def run_pipeline():
    video_path = '/workspace/train/Fighting003_x264.mp4'
    target_text = "fighting" 
    
    TEMP_DIR = '/workspace/VADinference/video_temp'
    
    original_video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    vad_result = run_vad_inference(video_path=video_path)
    
    if not vad_result:
        print(f"{original_video_name} 0 -1")
        return


    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    copy_video(result=vad_result, video_path=video_path, output_dir=TEMP_DIR)
    del vad_result  
    gc.collect()    
    torch.cuda.empty_cache() 

    video_files = [
        os.path.join(TEMP_DIR, f) 
        for f in os.listdir(TEMP_DIR) 
        if f.lower().endswith(('.mp4', '.avi'))
    ]
    
    final_results = {}

    for video in video_files:
        print(f"正在核验: {os.path.basename(video)} ... ", end="")
        

        llm_score = verify_video_content(video_path=video, target_label=target_text)
        
        if llm_score == 1:
            basename = os.path.basename(video)
            name_no_ext = os.path.splitext(basename)[0]
            
            try:
                parts = name_no_ext.rsplit('_', 2)
                if len(parts) == 3:
                    event_key = parts[0]
                    start_f = int(parts[1])
                    end_f = int(parts[2])
                    
                    if event_key not in final_results:
                        final_results[event_key] = []
                    final_results[event_key].append([start_f, end_f])
            except Exception as e:
                print(f"文件名解析错误: {e}")

    all_segments = []
    if final_results:
        for k in final_results:
            for item in final_results[k]:
                all_segments.append(item)

    if not all_segments:
        print(f"{original_video_name} 0 -1")
    else:
       
        all_segments.sort(key=lambda x: x[0])
        
        final_segments = all_segments[:4]
        
        seg_str_list = []
        for start, end in final_segments:
            seg_str_list.append(str(start))
            seg_str_list.append(str(end))
            
        output_str = f"{original_video_name} 1 {' '.join(seg_str_list)} -1"

if __name__ == "__main__":
    run_pipeline()