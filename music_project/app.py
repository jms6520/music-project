import cv2
import mediapipe as mp
import mido
import time

# Initialize MediaPipe Hand Tracker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open MIDI Output Port
try:
    outport = mido.open_output(mido.get_output_names()[0])  # Select the first available MIDI output port
    print(f"Connected to MIDI output: {mido.get_output_names()[0]}")
except IndexError:
    print("No MIDI output device found. Please connect one and try again.")
    exit()

# Function to send MIDI clock ticks
def send_midi_clock(outport, tempo):
    # Calculate the interval between MIDI clock ticks based on tempo (in microseconds)
    microseconds_per_tick = 60000000 / (24 * tempo)  # 24 ticks per beat (MIDI Clock standard)
    outport.send(mido.Message('clock'))  # Send a single clock tick message
    return microseconds_per_tick



def list_available_cameras():
    # Get the number of available cameras
    num_cameras = 0
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            num_cameras += 1
            cap.release()
        else:
            break

    # List the available cameras
    print(f"Found {num_cameras} available cameras:")
    for i in range(num_cameras):
        cap = cv2.VideoCapture(i)
        print(f"Camera {i}: {cap.getBackendName()}")
        cap.release()


def select_camera():
    # Ask the user to select a camera
    camera_id = int(input(f"Select a camera: "))
    return camera_id



# Initialize webcam
try:
    list_available_cameras()
    camera_id = select_camera()
    cap = cv2.VideoCapture(camera_id)  # Select the second available camera
except e:
    print("No camera found. Please connect one and try again.")
    exit()

# Set default tempo
tempo = 120  # Default tempo in BPM

print("Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Get thumb and index finger landmarks
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the distance between thumb and index finger
            distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

            # Adjust tempo based on distance
            # Larger distance = slower tempo, smaller distance = faster tempo
            tempo = max(60, min(180, 120 + (distance * 200)))

            # Send MIDI Clock Tempo message to adjust Logic's tempo
            send_midi_clock(outport, tempo)

    # Display the current tempo on the video feed
    cv2.putText(frame, f"Tempo: {int(tempo)} BPM", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow('Hand Gesture Tempo Control', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
outport.close()

