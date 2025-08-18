import cv2

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height and also the fps
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cam.get(cv2.CAP_PROP_FPS)

# Saves the camera properties to camera_outputs.txt
with open('assignment_1/solutions/camera_outputs.txt', 'w') as f:
    f.write(f"fps: {fps}\n")
    f.write(f"height: {frame_height}\n")
    f.write(f"width: {frame_width}\n")
    

# Release the capture and writer objects
cam.release()
