import cv2
from pathlib import Path

# List of image filenames
image_folder = Path("data-v2/evolution-monthwise/image")
images = [
    image_folder / f"map.{year}-{month}.png"
    for year in range(1990, 2024)
    for month in range(0, 12)
]

# Ensure there are images
if not images:
    raise ValueError("No PNG images found in the specified folder.")

# Specify the output video properties
frame = cv2.imread(str(images[0]))
height, width, layers = frame.shape
video_name = "output_video.mp4"
fps = 24  # frames per second

# Video Writer - using 'mp4v' codec for MP4 format
video = cv2.VideoWriter(
    video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

repeat_single_image = 6
# Adding images to video, each image repeated for 1 second
for image in images:
    if not image.exists():
        continue
    frame = cv2.imread(str(image))
    for _ in range(
        repeat_single_image
    ):  # Repeat the frame for 'fps' times to last for 1 second
        video.write(frame)

# Release the video writer
video.release()
