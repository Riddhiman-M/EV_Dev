import cv2

# Address of sample video to be fetched
path = "C:/Users/Riddhiman Moulick/IIT Kharagpur/pythonProject/Electric_Vehicle/Resource/Traffic - 27260.mp4"
vid = cv2.VideoCapture(path)

ret, frame1 = vid.read()

while vid.isOpened():

    # Retrieving frames from the video
    ret, frame2 = vid.read()

    # Checking if the video has come to an end
    if not ret:
        break

    frame = frame2.copy()

    # Creating a mask to separate out the foreground from the vehicles
    # In 2 continuous frames the only parts that remain same are all the stationary objects
    # Hence by using absdiff, we subtract the 2 frames, keeping only the moving vehicles in the output
    Mask = cv2.absdiff(frame1, frame2)
    Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(Mask, 50, 255, cv2.THRESH_BINARY) # _ used when we don't need the other
                                                                # output parameters
    frame1 = frame2

    # Extracting Contours from the Mask
    conts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in conts:

        # Checking Contour Area so that small irregularities in contours are overlooked
        if cv2.contourArea(c) < 800:
            continue

        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Foreground Mask", Mask)
    cv2.imshow("Original Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vid.release()
