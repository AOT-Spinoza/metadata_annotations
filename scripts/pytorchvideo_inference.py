from pytorchvideo.data.encoded_video import EncodedVideo
from tqdm import tqdm
import torch
import numpy as np
import os
from src.save_data import load_data_from_hdf5


def get_person_bboxes(video_name):
    base_video_name = os.path.basename(video_name)
    base_video_name_without_extension = os.path.splitext(base_video_name)[0]
    task_map = "object_detection"
    model_name = "fasterrcnn_resnet50_fpn_v2"
    map_name = f"{task_map}/{model_name}"
    hdf5_file_name = f"./result/{base_video_name}/{map_name}/{base_video_name_without_extension}_{model_name}.hdf5"
    print(hdf5_file_name)
    if not os.path.exists(hdf5_file_name):
        raise FileNotFoundError("First do object detection on video")
        
    
    # Convert hdf5_file to a list of dictionaries using your custom function
    detection_data = load_data_from_hdf5(hdf5_file_name)

    has_label_1 = any((frame['labels'] == 1).any() for frame in detection_data)
    if not has_label_1:
        print("No person detected in video")
        return None

    return detection_data
    
    
def get_frame_boxes(detection_data, frame_number):
    # Get the bounding boxes for the frame number
    frame_data = detection_data[frame_number]
    # Create a boolean mask where True corresponds to a label of 1
    mask = (frame_data['labels'] == 1)

    # Use the mask to select the boxes with a label of 1
    
    person_boxes = frame_data['boxes'][mask]
    
    return person_boxes

def infer_videos(video_files, model, transformation, clip_duration, classes, model_name):
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval()
    print(f"Using device: {device}")
    # Move the model to the device
    model = model.to(device)
    # Initialize the output dictionary
    
    outputs_all = {}
    for video_path in tqdm(video_files, desc=f"Processing videos for {model_name}"):
        detection_data = get_person_bboxes(video_path)
        if detection_data is None:
            print("Skipping video, no person detected")
            outputs_all[os.path.basename(video_path)] = {}
            continue
        encoded_video = EncodedVideo.from_path(video_path)
        endbound = 2.6
        time_stamp_range = np.arange(0, endbound, clip_duration)
        frame_rate = 60
        for time_stamp in time_stamp_range:
            print("Generating predictions for time stamp: {} sec".format(time_stamp))

            # Generate clip around the designated time stamps
            inp_imgs = encoded_video.get_clip(
                time_stamp - clip_duration/2.0, # start second
                time_stamp + clip_duration/2.0  # end second
            )
            inp_imgs = inp_imgs['video']

            start_frame_number = int((time_stamp - clip_duration/2.0) * frame_rate)
            middle_frame_number = start_frame_number + inp_imgs.shape[1]//2


            # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
            # We use the the middle image in each clip to generate the bounding boxes.
            inp_img = inp_imgs[:,inp_imgs.shape[1]//2,:,:]
            inp_img = inp_img.permute(1,2,0)

            # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
            predicted_boxes = get_frame_boxes(detection_data, middle_frame_number)
            if len(predicted_boxes) == 0:
                print("Skipping clip no frames detected at time stamp: ", time_stamp)
                continue

            
            # Preprocess clip and bounding boxes for video action recognition.
            inputs, inp_boxes, _ = transformation(inp_imgs, predicted_boxes)
            # Prepend data sample id for each bounding box.
            # For more details refere to the RoIAlign in Detectron2
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)

            # Generate actions predictions for the bounding boxes in the clip.
            # The model here takes in the pre-processed video clip and the detected bounding boxes.
            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(device)
            preds = model(inputs, inp_boxes.float().to(device))

            preds= preds.to('cpu')
            # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
            preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)

            classes_dict = classes[0][0]
            # Get the class with the highest probability for each instance of a person
            # Get the class with the highest probability for each instance of a person
            max_probs, max_classes = torch.max(preds[:, 1:], dim=1)
            max_probs = max_probs.tolist()
            max_classes = max_classes.tolist()
            max_classes = [i+1 for i in max_classes]
            # Use the indices of the max probabilities to get the class names
            class_names = [classes_dict[i] for i in max_classes]
            print(class_names)
            # Store the results in the output dictionary
            outputs_all[os.path.basename(video_path)] = {middle_frame_number: {'max_probs': max_probs, 'max_classes': max_classes}}
            outputs_all[os.path.basename(video_path)] = {middle_frame_number: preds}
            
    return outputs_all