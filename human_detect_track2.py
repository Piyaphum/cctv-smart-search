'''
Version2 (Modified with Video Recording)
Tracking + reidentification + Export Video
'''
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import time
import os
import sys 
from run import Reid
from importlib import import_module

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.reid = Reid()
        self.path_to_ckpt = path_to_ckpt
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()    
        self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        # print("Elapsed Time:", end_time-start_time) # ปิด print เพื่อไม่ให้ลกหน้าจอ

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()  

    def find(self, img, boxes_cur, boxes_prev, box):
        print('## Find called (Re-ID)')
        cv2.imwrite('./temporaryImg.jpg',img)

        past_ppl = './past_ppl'
        if not os.path.exists(past_ppl):
            os.makedirs(past_ppl)

        folders = os.listdir(past_ppl)

        for folder in folders:
            files = os.listdir(past_ppl + '/' + folder)
            same = 0
            diff = 0
            for f in files:
                ret = self.reid.compare('./temporaryImg.jpg'  ,    './past_ppl/' + folder + '/' + f)
                
                if(ret == True):
                    same += 1
                else:
                    diff += 1
                
            total_compare = same + diff
            if total_compare == 0:
                p = 0
            else:
                p = 100 * float(same) / float(total_compare) 

            # Threshold ในการตัดสินว่าใช่คนเดิมไหม (แนะนำ 70-80)
            if( p > 70 ):
                person_no = len(files) + 1
                cv2.imwrite(past_ppl + '/' + folder + '/' + str(person_no) + '.jpg',img)   
                boxes_cur[ int(folder) ][0] = box   
                boxes_prev[ int(folder) ] = -1 
                return
        
        # ถ้าหาไม่เจอเลย สร้าง ID ใหม่
        l = len(folders)
        os.makedirs(past_ppl + '/' + str( l )  )
        cv2.imwrite(past_ppl + '/' + str( l ) + '/1.jpg',img)
        boxes_cur.append( [box] )
        
        return


def iou(box1, box2):
    xa = max( box1[1] , box2[1] )
    ya = max( box1[0] , box2[0] )
    xb = min( box1[3] , box2[3] )
    yb = min( box1[2] , box2[2] )
    
    interArea = max(0, xb - xa ) * max(0, yb - ya )

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1] )
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1] )
 
    denom = float(box1Area + box2Area - interArea)
    if denom == 0: return 0
    
    iou = float(interArea) / denom
    return iou
 

if __name__ == "__main__":
    # Path ของโมเดล Detect คน
    model_path = './model/frozen_inference_graph.pb'

    past_ppl = './past_ppl'
    if not os.path.exists(past_ppl):
        os.makedirs(past_ppl)

    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.8
    iou_threshold = 0.6
    
    # รับชื่อไฟล์วิดีโอ
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = './video.avi'
        
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        sys.exit()
    
    # ==========================================
    # [ADDED] ส่วนตั้งค่าการบันทึกวิดีโอ (Video Writer)
    # ==========================================
    # ดึง FPS จากวิดีโอต้นฉบับ
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # ค่า Default
    
    # กำหนดขนาด Output ให้ตรงกับที่ resize ด้านล่าง (1280x720)
    output_width = 1280
    output_height = 720
    
    output_filename = 'output_result.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # ใช้ Codec XVID (ไฟล์ .avi)
    out_writer = cv2.VideoWriter(output_filename, fourcc, fps, (output_width, output_height))
    
    print(f"🎬 Recording to: {output_filename}")
    # ==========================================

    k = 25
    boxes_prev = []
    framenum = 1
    start_time =  time.time()

    while True: 
        r, img = cap.read()
        
        if not r or img is None:
            print("Video ended.")
            break
            
        # Resize ภาพเป็น 1280x720 (ต้องตรงกับ VideoWriter)
        img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)
        boxes_cur = []
        for l in range(len(boxes_prev)):
            if( len(boxes_prev[l]) < k ):
                boxes_cur.append(  [-1] + boxes_prev[l]  )
            else:
                boxes_cur.append(  [-1] + boxes_prev[l][0:k-1]  )
                
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                
                cropped_img = img[ box[0]:box[2] , box[1]:box[3] ]
                if cropped_img.size == 0: continue

                maxthreshold = -1
                maxindex = 101     
                
                # Tracking Loop
                for j in range( len(boxes_prev) ):
                    if( boxes_prev[j] == -1 ): continue
                    for kk in range( len(boxes_prev[j]) ):
                        if(boxes_prev[j][kk] == -1): continue
                        r = iou( boxes_prev[j][kk] ,box)
                        if(  r > maxthreshold  and  r > iou_threshold):
                            maxthreshold = r
                            maxindex = j            
                        
                display_text = "Identifying..."
                box_color = (0, 0, 255) # Red

                if( maxthreshold != -1 ):
                    # Tracked successfully
                    boxes_cur[ maxindex ][0] = box
                    boxes_prev[ maxindex ] = -1
                    
                    person_folder = past_ppl + '/' + str(maxindex)
                    if not os.path.exists(person_folder): os.makedirs(person_folder)
                    person_no = len( os.listdir( person_folder ) ) + 1
                    cv2.imwrite(person_folder + '/' + str(person_no) + '.jpg',cropped_img) 
                    
                    display_text = "ID: " + str(maxindex)
                    box_color = (255, 0, 0) # Blue

                else:
                    # Re-ID needed
                    odapi.find(img, boxes_cur,boxes_prev, box )
                    display_text = "Re-ID Process"
                    box_color = (0, 255, 0) # Green

                # Draw Rectangle & Text
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]), box_color, 2)
                cv2.putText(img, display_text, (box[1], box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                    
        framenum += 1  
        boxes_prev =  boxes_cur

        # แสดงผลหน้าจอ
        cv2.imshow("preview", img)
        
        # [ADDED] บันทึกภาพลงไฟล์วิดีโอ
        out_writer.write(img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out_writer.release() # [ADDED] ปิดไฟล์วิดีโอ
    cv2.destroyAllWindows()
    print(f"✅ Video saved successfully as {output_filename}")