import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm


class PicoDet():
    def __init__(self,
                 model_pb_path,
                 prob_threshold=0.2,
                 iou_threshold=0.3):
        # BGR
        self.classes = []
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.mean = [0.485, 0.456, 0.406]
        # np.array(
        #     [103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = [0.229, 0.224, 0.225]
        # np.array(
        #     [57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)
        inputs_name = [a.name for a in self.net.get_inputs()]
        inputs_shape = {
            k: v.shape
            for k, v in zip(inputs_name, self.net.get_inputs())
        }
        self.input_shape = inputs_shape['image'][2:]

    def preprocess(self, image, keep_ratio=False):
        input_h, input_w = self.input_shape
        scale = min(input_h / image.shape[0], input_w / image.shape[1])
        ox = (-scale * image.shape[1] + input_w + scale  - 1) * 0.5
        oy = (-scale * image.shape[0] + input_h + scale  - 1) * 0.5
        M = np.array([
            [scale, 0, ox],
            [0, scale, oy]
        ], dtype=np.float32)
        IM = cv2.invertAffineTransform(M)

        image_prep = cv2.warpAffine(image, M, (input_w, input_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        image_prep = (image_prep[..., ::-1] / 255.0 - self.mean) / self.std

        image_prep = image_prep.transpose(2, 0, 1)[None].astype(np.float32)
        return image_prep, M, IM

    def postprocess(self, raw_output, IM, prob_threshold= 0.25):
        def nms(boxes, threshold=0.5):

            keep = []
            remove_flags = [False] * len(boxes)
            for i in range(len(boxes)):

                if remove_flags[i]:
                    continue

                ib = boxes[i]
                keep.append(ib)
                for j in range(len(boxes)):
                    if remove_flags[j]:
                        continue

                    jb = boxes[j]

                    # class mismatch or image_id mismatch
                    if ib[6] != jb[6] or ib[5] != jb[5]:
                        continue

                    cleft,  ctop    = max(ib[:2], jb[:2])
                    cright, cbottom = min(ib[2:4], jb[2:4])
                    cross = max(0, cright - cleft) * max(0, cbottom - ctop)
                    union = max(0, ib[2] - ib[0]) * max(0, ib[3] - ib[1]) + max(0, jb[2] - jb[0]) * max(0, jb[3] - jb[1]) - cross
                    iou = cross / union
                    if iou >= threshold:
                        remove_flags[j] = True
            return keep
        bboxes = raw_output[0]                      # b num_box     4(xyxy)
        scores = raw_output[1].transpose(0,2,1)     # b num_box num_class
        confidences = np.max(scores, axis=2)
        labels = np.argmax(scores, axis=2)
        boxes = []
        for bathc_id, box_id in zip(*np.where(confidences > self.prob_threshold)):
            item = bboxes[bathc_id, box_id]
            label = labels[bathc_id, box_id]
            confidence = confidences[bathc_id, box_id]
            boxes.append([*item, confidence, bathc_id, label])
        boxes = np.array(boxes)
        if(len(boxes) == 0):
            return boxes
        lr = boxes[:, [0, 2]]
        tb = boxes[:, [1, 3]]
        boxes[:, [0, 2]] = lr * IM[0, 0] + IM[0, 2]
        boxes[:, [1, 3]] = tb * IM[1, 1] + IM[1, 2]

        # left, top, right, bottom, confidence, image_id, label
        boxes = sorted(boxes.tolist(), key=lambda x:x[4], reverse=True)
        print(len(boxes))
        return nms(boxes)

    def detect(self, srcimg):
        image_prep, M, IM = self.preprocess(srcimg)

        inputs_dict = {
            'image': image_prep,
        }
        inputs_name = [a.name for a in self.net.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}

        outs = self.net.run(None, net_inputs)
        boxes = self.postprocess(outs, IM)
        for obj in boxes:
            left, top, right, bottom = map(int, obj[:4])
            confidence = obj[4]
            label = int(obj[6])
            cv2.rectangle(srcimg, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(srcimg, f"{label}: {confidence:.2f}", (left, top+20), 0, 1, (0, 0, 255), 2, 16)
        cv2.imwrite("result.jpg",srcimg)
        return srcimg

    def detect_folder(self, img_fold, result_path):
        img_fold = Path(img_fold)
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        img_name_list = filter(
            lambda x: str(x).endswith(".png") or str(x).endswith(".jpg"),
            img_fold.iterdir(), )
        img_name_list = list(img_name_list)
        print(f"find {len(img_name_list)} images")

        for img_path in tqdm(img_name_list):
            img = cv2.imread(str(img_path))

            srcimg = net.detect(img)
            save_path = str(result_path / img_path.name.replace(".png", ".jpg"))
            cv2.imwrite(save_path, srcimg)
if __name__ == '__main__':
    import sys
    net = PicoDet(
        sys.argv[1])
    img = cv2.imread(sys.argv[2])
    net.detect(img)
    