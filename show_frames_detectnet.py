import cv2
from PIL import Image

import sys
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput,  loadImage, cudaResize, cudaAllocMapped, cudaFromNumpy, Log
import argparse
import time




coco_classes = [
"unlabeled",
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"street sign",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"hat",
"backpack",
"umbrella",
"shoe",
"eye glasses",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"plate",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"potted plant",
"bed",
"mirror",
"dining table",
"window",
"desk",
"toilet",
"door",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"blender",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
]


def main():


    parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                     formatter_class=argparse.RawTextHelpFormatter, 
                                     epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
    parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

        
    # load the object detection network
    net = detectNet(args.network, sys.argv, args.threshold)


    # to properly know which camera to use, make sure to do `ls /dev | grep video`.
    # the input to videocapture is the index corresponding to the connected camera
    cap = cv2.VideoCapture(0)

    import time
    ctr = 0
    t0 = time.time()
    for i in range(0, 10000):
        ret, frame = cap.read()
        print(type(frame))
        img = Image.fromarray(frame)
#        img.save(f'images/frame{ctr}.jpg')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ctr += 1
        cuda_mem = cudaFromNumpy(frame)
        detections = net.Detect(cuda_mem, overlay=args.overlay)
        print(len(detections), ' detected')
        for idx,d in enumerate(detections):
            print(d)
            print(dir(d))
            cv2.rectangle(frame,(int(d.Left),int(d.Top)),(int(d.Right),int(d.Bottom)),(0,255,0),2)
            cv2.putText(frame,coco_classes[d.ClassID],(int(d.Right)+10,int(d.Bottom)),0,0.3,(0,255,0))

#            roi=frame[int(d.Top):int(d.Bottom),int(d.Left):int(d.Right)]
#            cv2.imwrite(str(idx) + '.jpg', roi)

        cv2.imshow('Video stream ', frame)
    t1 = time.time()
    total = t1 - t0
    print('total ', total)
    print((ctr / total) , 'fps')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



