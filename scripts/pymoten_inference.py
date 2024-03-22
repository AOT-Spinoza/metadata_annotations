from tqdm import tqdm
import moten
import os

def infer_videos(video_files, model, config, task_type, model_name):
    outputs_all = {}
    for video in tqdm(video_files, desc=f"Processing videos for {model_name}"):
        luminance_images = moten.io.video2luminance(video, nimages=config.tasks[task_type][model_name].load_model.parameters_transformation.nimages)
        outputs_all[os.path.basename(video)] = model.project_stimulus(luminance_images)
    return outputs_all
