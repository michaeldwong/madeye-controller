
import cv2
from PIL import Image


def main():
    # To properly know which camera to use, make sure to do `ls /dev | grep video`.
    # THe input to VideoCapture is the index corresponding to the connected camera
    cap = cv2.VideoCapture(0)

    ctr = 0
    while True:
        ret, frame = cap.read()
#        print(type(frame))
        print(type(frame))
        cv2.imshow('Video stream ', frame)
        img = Image.fromarray(frame)
#        img.save(f'images/frame{ctr}.jpg')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ctr += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



