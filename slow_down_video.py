import cv2
def slow_down(video_path):

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # result = cv2.VideoWriter('slowed_down_video.avi', fourcc, 10.0, (600, 600), False) # better to use os.join here
    cap = cv2.VideoCapture(video_path)
    frame_time = 240
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.resize(frame, (600, 600))
            # result.write(frame)
            cv2.imshow('frame', frame)
        if cv2.waitKey(frame_time) & 0xFF == ord('q'):
            break
        
    cap.release()
    # result.release()
    cv2.destroyAllWindows()

slow_down('.//NN_Output//NN_Stream_18.avi')