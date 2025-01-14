from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import os

def load_dino():
    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")
    print("Model loaded.")
    return model

def generate_annotations(model, image, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD):
    
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    return boxes, logits, phrases

def iou(box1, box2):
    '''
    Helper functon of nms. Calculates intersection of union of 2 boxes (given as [topleft_x, topleft_y, bottomright_x, bottomright_y]).

    :param box1: Coordinates of first box
    :param box2: Coordinates of second box
    '''
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
	# compute the area of intersection rectangle
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
	# compute the intersection over union by taking the intersection
    
    iou = interArea / float(box1Area + box2Area - interArea)
	# return the intersection over union value
    return iou


def nms(NMS_THRESHOLD, boxes, logits):
    '''
    Process bounding boxes after they have been generating to suppress boxes that detect the same object
    :param threshold: Maximum intersection over union for box to be kept
    :param results: a tensor of tensors containing bounding box coordinates, bounding box class, and bounding box confidence
    '''

    
    boxes = boxes.tolist()
    logits = logits.tolist()
    


    for i in range(len(boxes)):
        boxes[i].append(logits[i]) # [[top_x, top_y, btm_x, btm_y, score], . . ., [top_x, top_y, btm_x, btm_y, score]]

    boxes_after_nms = []
    logits_after_nms = []
    boxes = sorted(boxes, key=lambda x: x[-1], reverse=True) #sort boxes in descending order based on confidence

    while boxes:
        chosen_box = boxes.pop(0)
        boxes = [bbox for bbox in boxes if iou(chosen_box[:-1], bbox[:-1]) < NMS_THRESHOLD]
        boxes_after_nms.append(chosen_box[:-1])
        logits_after_nms.append(chosen_box[-1])
    return torch.Tensor(boxes_after_nms), torch.Tensor(logits_after_nms)

def overlay_and_write(image_source, boxes_after_nms, logits_after_nms, phrases, filenum, path, filename):
    annotated_frame = annotate(image_source=image_source, boxes=boxes_after_nms, logits=logits_after_nms, phrases=phrases)
    cv2.imwrite(os.path.join(path, f'{filename}.png'), annotated_frame)

def generate_textfile(boxes_after_nms, logits_after_nms, filenum, path, filename):
    
    annotation_file = os.path.join(path, f'{filename}.txt')
    # Write the contents of boxes_after_nms and logits_after_nms to the file
    with open(annotation_file, "w") as f:
       
        for box, logit in zip(boxes_after_nms, logits_after_nms):
            box = box.tolist()
            logit = logit.tolist()
            #box.append(logit)
            f.write(f'0 {box[0]} {box[1]} {box[2]} {box[3]}' +  '\n')

    f.close()
def main():
    model = load_dino()

    INPUT_DIR = '<enter images directory>'
    TEXT_PROMPT = "<enter object to detect>"
    BOX_THRESHOLD = 0.2
    TEXT_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.7
    OUTPUT_DIRECTORY1 = '<output images directory>'
    OUTPUT_DIRECTORY2 = '<output text directory>'

    PARENT_DIRECTORY = '<parent directory>'
    os.mkdir(os.path.join(PARENT_DIRECTORY, OUTPUT_DIRECTORY1))
    os.mkdir(os.path.join(PARENT_DIRECTORY, OUTPUT_DIRECTORY2))
    filenum = 0
    for file in os.listdir(INPUT_DIR): 
        #read file
        filename = file.split('.')[0]
        file = os.path.join(INPUT_DIR, file)
        print("filenum:", filenum)
        image_source, image = load_image(file)
        boxes, logits, phrases = generate_annotations(model, image, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD)
        boxes_after_nms, logits_after_nms = nms(NMS_THRESHOLD, boxes, logits)
        overlay_and_write(image_source, boxes_after_nms, logits_after_nms, phrases, filenum, PARENT_DIRECTORY + OUTPUT_DIRECTORY1, filename)
        generate_textfile(boxes_after_nms, logits_after_nms, filenum, PARENT_DIRECTORY + OUTPUT_DIRECTORY2, filename)

        filenum += 1
        

if __name__ == '__main__':
    main()