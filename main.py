
import cv2
import mediapipe as mp
import time


# 1. Initialize MediaPipe
 

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode

 
# 2. Configure hand detection
 

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

 
# 3. Draw hand connections
 

connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

 
# 4. Function to count fingers
 

def count_fingers(landmarks, hand_label):

    count = 0

    # Thumb detection
    if hand_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            count += 1
    else:
        if landmarks[4].x > landmarks[3].x:
            count += 1

    # Other four fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):

        if landmarks[tip].y < landmarks[pip].y:
            count += 1

    return count


 
# 5. Start webcam
 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

   
# 6. Start hand landmarker
 

with HandLandmarker.create_from_options(options) as landmarker:

    start_time = time.monotonic()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot read webcam")
            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Calculate timestamp in milliseconds
        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        # Detect hand landmarks
        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        total_fingers = 0

        # --------------------------------
        # 7. Process detected hands
        # --------------------------------

        if results.hand_landmarks:

            for i, hand_landmarks in enumerate(
                results.hand_landmarks
            ):

                # Get hand type
                hand_label = results.handedness[i][0].category_name

                # Count fingers
                finger_count = count_fingers(
                    hand_landmarks,
                    hand_label
                )

                total_fingers += finger_count

                # Draw landmarks
                height, width, _ = frame.shape

                for point in hand_landmarks:

                    cx = int(point.x * width)
                    cy = int(point.y * height)

                    cv2.circle(
                        frame,
                        (cx, cy),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw connections
                for connection in connections:

                    start = hand_landmarks[
                        connection.start
                    ]

                    end = hand_landmarks[
                        connection.end
                    ]

                    start_point = (
                        int(start.x * width),
                        int(start.y * height)
                    )

                    end_point = (
                        int(end.x * width),
                        int(end.y * height)
                    )

                    cv2.line(
                        frame,
                        start_point,
                        end_point,
                        (255, 0, 255),
                        2
                    )

     
        # 8. Display finger count
    

        cv2.rectangle(
            frame,
            (10, 10),
            (380, 85),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "Finger Count: " + str(total_fingers),
            (25, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Show webcam
        cv2.imshow(
            "MediaPipe Finger Counter",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

 
# 9. Release resources
 

cap.release()
cv2.destroyAllWindows()

print("Finger Counter Closed")
