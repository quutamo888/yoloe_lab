# Design Spec: YOLOE Real-time Video Player inside Jupyter Notebook

**Date:** 2026-06-29  
**Status:** Approved  
**Topic:** Real-time object detection video player using PPYOLOE-26 and IPywidgets with external OpenCV window playback.

---

## 1. Goal Description
The objective is to implement a new interactive cell in the Jupyter Notebook `yoloe_video_lab.ipynb`. This cell will provide a dropdown list of available video files in the workspace directory, as well as a text input for arbitrary video paths. When triggered, it will load the selected video, run PP-YOLOE-26 object detection and segmentation on each frame in real-time, and display the processed stream in a native, external OpenCV window. The user must be able to stop the video at any time by pressing 'q' or 'Esc'.

---

## 2. Architecture & Components

### 2.1 Interactive UI Widget Layout (IPywidgets)
The notebook cell will render standard Jupyter widgets:
* **Dropdown Selection (`ipywidgets.Dropdown`)**: Scans `os.listdir(".")` for `.mp4`, `.avi`, `.mkv`, and `.mov` files to populate a list of available videos in the current workspace.
* **Text Input (`ipywidgets.Text`)**: Allows manually specifying custom paths (e.g., full Windows paths like `F:\videos\test.mp4`).
* **Trigger Button (`ipywidgets.Button`)**: Clicking it starts the real-time inference loop.
* **Output Panel (`ipywidgets.Output`)**: Standard output redirection to print logging messages and status updates directly below the widgets.

### 2.2 Inference & OpenCV Playback Logic
When the play button is clicked:
1. Validate that the specified video file path exists.
2. Initialize `cv2.VideoCapture(video_path)`.
3. Create a named, resizable window: `cv2.namedWindow("YOLOE Real-time Detection", cv2.WINDOW_NORMAL)`.
4. Run a loop to read frames:
   * Execute YOLOE inference: `results = model.predict(frame, conf=0.25, verbose=False)`.
   * Render segmentation masks and bounding boxes: `annotated_frame = results[0].plot()`.
   * Show in window: `cv2.imshow("YOLOE Real-time Detection", annotated_frame)`.
   * Handle user exit input: check `cv2.waitKey(1) & 0xFF == ord('q')` or key code `27` (Esc). If matched, break the loop.
5. Post-loop cleanups: release the video capture object (`cap.release()`) and destroy all OpenCV windows (`cv2.destroyAllWindows()`) to ensure no resource leaks or Jupyter kernel freezes.

---

## 3. Detailed Proposed Changes

### [MODIFY] [yoloe_video_lab.ipynb](file:///f:/test/yoloe_lab/yoloe_video_lab.ipynb)
We will insert a new cell at the end of the notebook (after cell 8, which is the 1-line predict method) containing the implementation of this interactive widget and real-time playback logic.

---

## 4. Verification Plan

### Manual Verification
1. Run the newly added cell in the Jupyter Notebook environment.
2. Verify that the dropdown list shows `sample_video.mp4` and `yoloe_detected_video.mp4`.
3. Select `sample_video.mp4` and click the play button.
4. Verify that an external window titled "YOLOE Real-time Detection" pops up, displaying the video with real-time detection boxes and segmentation masks.
5. Press the 'q' key and verify that the window closes immediately and the Jupyter cell execution completes cleanly.
