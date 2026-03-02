import os
import ffmpeg

def process(video_path, save_path, start_frame, end_frame):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    start_time = start_frame / 30.0
    end_time = end_frame / 30.0
    
    (
        ffmpeg
        .input(video_path, ss=start_time, to=end_time)
        .output(save_path, c='copy')
        .global_args('-y', '-loglevel', 'error')
        .run()
    )

def copy_video(result, video_path, output_dir='./video_temp'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    for key in result.keys():
        for frame in result[key]:
            start_f = frame[0]
            end_f = frame[1]
            file_name = f"{key}_{start_f}_{end_f}.mp4"
            save_path = os.path.join(output_dir, file_name)
            
            process(video_path, save_path, start_f, end_f)