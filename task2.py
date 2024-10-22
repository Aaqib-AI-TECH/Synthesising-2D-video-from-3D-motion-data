import ezc3d
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the C3D file
c3d_file_path = r'C:\Users\Aaqib\Downloads\0038_Ankle_01.c3d'
c3d = ezc3d.c3d(c3d_file_path)

# Points Extraction (3D coordinates)
points = c3d['data']['points']

# X and Y extraction 
x_points = points[0, :, :] 
y_points = points[1, :, :] 

# Video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('2d_motion_video.mp4', fourcc, 30.0, (640, 480))

# Loop through each frame of the 3D
for frame in range(points.shape[2]):  # Loop over all frames
    fig, ax = plt.subplots()
    
    # Points for the frame
    ax.scatter(x_points[:, frame], y_points[:, frame], color='red')
    
    # Titles to X and Y axes
    ax.set_xlabel('X-axis (horizontal)')
    ax.set_ylabel('Y-axis (vertical)')
    ax.set_title(f"2D Projection - Frame {frame}")

    # Save the plot as an image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize of image if needed
    img = cv2.resize(img, (640, 480))

    # Write image to the video
    out.write(img)
    
    plt.close(fig)  

# Release video writer
out.release()
print("Video created: 2d_motion_video.mp4")
