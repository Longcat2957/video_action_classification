import os
import cv2
import numpy as np
from typing import List, Tuple

def read_img(p:str) -> np.ndarray:
    try:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    except Exception as e:
        print(f"Failed to read image file {p}: {e}")
        return None
    
def read_vid(p:str) -> List[np.ndarray]:
    result = []
    try:
        cap = cv2.VideoCapture(p)
        while True:
            ret, frame = cap.read()
            # 읽은 프레임이 없으면  종료
            if not ret: break
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result.append(frame)
        cap.release()
        return result
    
    except Exception as e:
        print(f"Failed to read video file {p}: {e}")
        return None
    
def get_max_value(frames:List[np.ndarray]) -> Tuple[int, int]:
    max_h, max_w = 0, 0
    for frame in frames:
        h, w = frame.shape[:2]
        max_h, max_w = max(h, max_h), max(w, max_w)
    return max_h, max_w

def resize_with_padding(image:np.ndarray, target_h:int, target_w:int) -> np.ndarray:
    h, w = image.shape[:2]
    
    # no padding
    if h >= target_h and w >= target_w:
        return cv2.resize(image, (target_w, target_h))

    # padding (1)
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    
    # padding (2)
    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    return padded_image


    
if __name__ == '__main__':
    sample_video_path = "/home/junghyun/Desktop/wei/mvit_transfer_learning/data/Celeb-DF-V1/Celeb-real/id0_0000.mp4"
    vid_frames = read_vid(sample_video_path)
    get_max_value(vid_frames)