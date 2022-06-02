from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='image_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    #cap = cv2.VideoCapture(args.video_source)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    #start_time = datetime.datetime.now()
    #num_frames = 0
    #im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    #cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
    image_np = cv2.imread(args.image_source)
    
    path = args.image_source.split('/')[-1]
    print(path)
    dir = "/content/runs/" +path
    text_file =  "/content/runs/" + path.split('.jpg')[0] +".txt"
    im_height,im_width,im_channels  = image_np.shape
    
    boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

      #draw bounding boxes on frame
    detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np,text_file)
    
    cv2.imwrite(dir,image_np)

    
    
