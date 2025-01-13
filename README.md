# Nest Camera Person Detection with Vision-Language Model (VLM)

This project demonstrates how to capture frames from a Nest camera via RTSP and use a Vision-Language Model (VLM) to detect if a person is present in the scene and track how long they remain (dwell time). If a person’s dwell time exceeds a specified threshold, we save the frame locally.  

![Illustration of the person detection concept](snapshots/vlm_person_detection.png)

## How It Works

1. **RTSP Stream**:
   - We generate an RTSP URL using the Nest Smart Device Management (SDM) API.  
   - After authentication, the code calls the “CameraLiveStream.GenerateRtspStream” command to retrieve the URL.

2. **Frame Capture**:
   - OpenCV accesses the RTSP URL as a video stream.  
   - We periodically capture frames (e.g., every 5 seconds) to reduce load and avoid processing every single frame.

3. **Vision-Language Model**:
   - We use a Hugging Face pipeline for image captioning (e.g., “Salesforce/blip-image-captioning-base”).  
   - Each captured frame is converted to a PIL Image and passed through the pipeline to generate a textual caption.

4. **Person Dwell Time Logic**:
   - If the caption text indicates a person through keywords (like “person,” “man,” “woman,” etc.), we start/continue a dwell timer.  
   - Once the dwell time surpasses a preset threshold (e.g., 10 seconds), we save the frame with a timestamp.

5. **Saved Frames**:
   - Frames are stored locally in the “clips” directory.  
   - Filenames include a timestamp and the dwell time (e.g., “vlm_person_20240101_123000_dwell15s.jpg”).

## Requirements

1. Nest API Setup:
   - A Nest developer account with the SDM API enabled.  
   - OAuth 2.0 credentials (client ID, client secret, refresh token).
   - Environment variables set for NEST_PROJECT_ID, DEVICE_ID, etc.

2. Python Packages:
   - OpenCV for capturing RTSP streams and saving frames.
   - Hugging Face Transformers & Torch for the image captioning pipeline.
   - Requests for Nest API calls.
   - PIL (Pillow) for image conversions.


## Usage

1. Set Environment Variables (or replace them with your actual strings in the code):
   ```
   export NEST_PROJECT_ID="your_project_id"
   export DEVICE_ID="your_device_id"
   export NEST_CLIENT_ID="your_client_id"
   export NEST_CLIENT_SECRET="your_client_secret"
   export NEST_REFRESH_TOKEN="your_refresh_token"
   ```

2. Run the script:
   ```
   python nest_cam_object_tracker.py --mode download
   ```
   - By default, it will capture up to 20 frames when a person has been detected for more than 10 seconds total dwell time.

3. Check the “clips” directory for saved images.  
   - Filenames will show timestamps and the dwell time.

## Customization

- To change the dwell threshold, pass a different command-line argument:
  ```
  python nest_cam_object_tracker.py --mode download --min-dwell-sec 20
  ```
  This will now require 20 seconds of continuous person presence before saving a frame.

- Adjust how often frames are captured by modifying the `frame_interval` variable in the code.  
- You can detect different subjects by expanding the keywords in the code or by using a more advanced classifier/captioning approach.

## Troubleshooting

- “500 Internal Error” can occur if the Nest camera is unavailable or if there are issues with your OAuth token. 
- “404 Not Found” might mean your camera or project ID is incorrect. 
- If you see slow performance, consider reducing the resolution or using a GPU for your VLM.

## References

- [Google Nest Device Access: SDM API](https://developers.google.com/nest/device-access)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

---
© 2023 NestCamVLMProject. 