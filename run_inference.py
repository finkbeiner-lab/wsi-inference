import runpod
import sys
sys.path.insert(0, '../')
import torchvision
import torch
import numpy as np
import cv2
import pandas as pd
from skimage.measure import regionprops
from skimage.color import rgb2hed, hed2rgb
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from collections import OrderedDict
from torch import Tensor
import warnings
from typing import List, Dict, Optional, Union
import base64
import io
from PIL import Image

class NumpyArrayDataset(Dataset):
    def __init__(self, array_dict):
        """
        Args:
            arrays (list of np.ndarray): List of NumPy arrays.
            keys (list): List of keys corresponding to each array.
        """
        self.arrays = array_dict

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            dict: Contains the 'array' and 'key' for the item at index idx.
        """
        array = self.arrays[idx][1]
        key = self.arrays[idx][0]
        # Define the torchvision image transforms
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        input_tensor = transform(array)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        image_float_np = array.astype(np.float32) / 255.0
        return {'tensor': input_tensor, 'array':image_float_np, 'key': key}

class LitMaskRCNN(L.LightningModule):
    def __init__(self, optim_config,backbone,rpn,roi_heads,transform):
        super().__init__()
        #self.model_config = _default_mrcnn_config(num_classes=1 + train_config['num_classes']).config
        #self.model = model
        self.optim_config = optim_config
        #self.model = build_default(model_config, im_size=1024)
        self.loss_names = 'objectness rpn_box_reg classifier box_reg mask'.split()
        self.loss_weights = [1., 4., 1., 4., 1.,]
        self.loss_weights = OrderedDict([(f'loss_{name}', weight) for name, weight in zip(self.loss_names, self.loss_weights)])
        self.backbone=backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform
        # used only on torchscript mode
        self._has_warned = False
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.avg_segmentation_overlap = 0.0
        self.val_acc = 0.0

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        #type hint
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        #TODO Why Another Transform Here?
        images, targets = self.transform(images, targets)
        

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    print(target_idx)
                    print(target["boxes"])
                    pdb.set_trace()
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # Image is passed through backbone model
        features = self.backbone(images.tensors)
        #self.visualize_feature_maps(images, features, show=False)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # Features - odict_keys(['0', '1', '2', '3', 'pool'])
        # targets - dict_keys(['boxes', 'labels', 'masks', 'image_id', 'area'])
        # images - torch.Size([3, 3, 1024, 1024])
        # proposals - torch.Size([2000, 4])
        proposals, proposal_losses = self.rpn(images, features, targets)
        #self.visualize_rpn_proposals(images, proposals, False)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        #if len(detections)!= 0:
            #self.visualize_roi_detections(images, detections, 20,False)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
        
       
    def get_loss_fn(self, weights, default=0.):
        def compute_loss_fn(losses):
            item = lambda k: (k, losses[k].item())
            metrics = OrderedDict(list(map(item, [k for k in weights.keys() if k in losses.keys()] + [k for k in losses.keys() if k not in weights.keys()])))
            loss = sum(map(lambda k: losses[k] * (weights[k] if weights is not None and k in weights.keys() else default), losses.keys()))
            return loss, metrics
        return compute_loss_fn
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        #print(batch)
        opt = self.optimizers()
        images, targets = batch 
        #images = [image for image in images]
        #targets = [dict([(k, v) for k, v in target.items()]) for target in targets]
        opt.zero_grad()
        loss_fn = self.get_loss_fn(self.loss_weights)
        loss, metrics = loss_fn(self.forward(images, targets))
        
        #loss.backward()
        self.manual_backward(loss)
        opt.step()
        #log_metrics.append(dict(epoch=epoch, loss=loss.item(), metrics=metrics))
        print_logs = "batch no : {batch_no}, total loss : {loss},  classifier :{classifier}, mask: {mask} ==================="
        print(print_logs.format( batch_no=batch_idx, loss=loss.item(),  classifier=metrics['loss_classifier'], mask=metrics['loss_mask']))
        self.log("loss", loss.item())
        self.log("metrics-loss_classifier", metrics['loss_classifier'])
        self.log("metrics-loss_mask", metrics['loss_mask'])
      
        #yield log_metrics
    
    #def backward(self, loss):
    #    loss.backward()
    
    def configure_optimizers(self):
        optimizer = self.optim_config['cls']([dict(params=list(self.parameters()))], **self.optim_config['defaults'])
        return optimizer

    
    def get_outputs(self, outputs, threshold):
        mask_list = []
        label_list = []
        class_names = ['Cored', 'Diffuse', 'Coarse-Grained', 'CAA']
        for j in range(len(outputs)):
            scores = outputs[j]['scores'].tolist()
            thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
            scores = [scores[x] for x in thresholded_preds_inidices]
            # get the masks
            masks = (outputs[j]['masks']>0.5).squeeze()
            # print("masks", masks)
            # discard masks for objects which are below threshold
            masks = [masks[x] for x in thresholded_preds_inidices]
            # get the bounding boxes, in (x1, y1), (x2, y2) format
            boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[j]['boxes'].tolist()]
            # discard bounding boxes below threshold value
            boxes = [boxes[x] for x in thresholded_preds_inidices]
            # get the classes labels
            labels = outputs[j]['labels'].tolist()
            labels = [labels[x] for x in thresholded_preds_inidices]
            mask_list.append(masks)
            label_list.append(labels)
        return mask_list, label_list

    def match_label(self, pred_label, gt_label):
        if pred_label==gt_label:
            return 1
        else:
            return 0
    
    def actual_label_target(self, gt_label):
        return gt_label
    
    
    def compute_iou(self, mask1, mask2):
        intersection = torch.logical_and(mask1, mask2).sum().item()
        union = torch.logical_or(mask1, mask2).sum().item()
        iou = (2*intersection) / union if union != 0 else 0
        return iou
    
    
    def evaluate_metrics(self, target,masks, labels):
        f1_score_list=[]
        matched_label_list=[]
        mean_f1_score = -1
        mean_matched_label=-1
        actual_label_list = []
        pred_label_list = []
        for i in range(len(target)):
            target_labels = self.actual_label_target(target[i]['labels'])
            for l in range(len(target_labels)):
                for j in range(len(masks)):
                    for k in range(len(masks[j])):
                        target_mask = target[i]['masks'][l]
                        target_mask = torch.where(target_mask > 0, torch.tensor(1), torch.tensor(0))
                        if target_mask.shape==masks[j][k].shape:
                            f1_score = self.compute_iou(masks[j][k],target_mask)
                            if f1_score>0:
                                f1_score_list.append(f1_score)
                                matched_label = self.match_label(labels[j][k],target_labels[l])
                                matched_label_list.append(matched_label)
                            #else:
                            #    matched_label_list.append(0)
            if len(f1_score_list)>0:
                mean_f1_score=np.nansum(f1_score_list)/len(f1_score_list)
            if len(matched_label_list)>0:
                mean_matched_label = sum(matched_label_list)/len(matched_label_list)
            #print(f1_score_list, matched_label_list)
            return mean_f1_score, mean_matched_label

    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        images, targets = batch 
        #images = [image for image in batch[0]]
        #targets = [dict([(k, v) for k, v in target.items()]) for target in batch[1]]
        #loss_fn = self.get_loss_fn(self.loss_weights)
        outputs = self.forward(images, targets)
        #print(outputs)
        masks, labels = self.get_outputs(outputs, 0.50)
        f1_mean, labels_matched =  self.evaluate_metrics(targets, masks, labels)
        self.avg_segmentation_overlap = f1_mean
        self.val_acc = labels_matched
        if (f1_mean>=0) or (labels_matched>=0):
            print(" Validation f1 mean score:", f1_mean, " perc labels matched", labels_matched)
        self.log('avg_seg_overlap',f1_mean)
        self.log('val_acc', labels_matched)
        return f1_mean, labels_matched
        #outputs1 = [x for x in outputs if len(x["labels"])!=0]
        #if len(outputs1)>0:
        #    print(outputs1[0].keys())
        #loss, metrics = loss_fn(outputs1[0])
        #loss, metrics = loss_fn(self.forward(images, targets))
        #print_logs = "batch no : {batch_no}, total loss : {loss},  classifier :{classifier}, mask: {mask} ==================="
        #print(print_logs.format( batch_no=batch_idx, loss=loss.item(),  classifier=metrics['loss_classifier'], mask=metrics['loss_mask']))
        
    
    def test_step(self, batch, batch_idx):
        # this is the validation loop
        images, targets = batch 
        #images = [image for image in batch[0]]
        #targets = [dict([(k, v) for k, v in target.items()]) for target in batch[1]]
        #loss_fn = self.get_loss_fn(self.loss_weights)
        #loss, metrics = loss_fn(self.forward(images, targets))
        outputs = self.forward(images)
        masks, labels = self.get_outputs(outputs, 0.25)
        f1_mean, labels_matched =  self.evaluate_metrics(targets, masks, labels)
        return f1_mean, labels_matched

class ExplainPredictions():
    # TODO fix the visualization flags
    def __init__(self, model, x, y, image_buffer, detection_threshold):
        self.model = model
        self.x = x
        self.y = y
        self.image_buffer = image_buffer
        self.detection_threshold = detection_threshold
        self.class_names = ['Cored', 'Diffuse', 'Coarse-Grained', 'CAA']
        self.class_to_colors = {'Cored': (255, 0, 0), 'Diffuse' : (0, 0, 255), 'Coarse-Grained': (0,255,0), 'CAA':(225, 255, 0)}
        #self.result_save_dir= os.path.join( "/home/mahirwar/Desktop/Monika/npsad_data/vivek/reports/New-Minerva-Data-output", self.model_input_path.split("/")[-1])
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        self.column_names = ["image_name", "region", "region_mask", "label", 
                            "confidence", "brown_pixels", "centroid", 
                            "eccentricity", "area", "equivalent_diameter","mask_present"]

    
    def prepare_input(self, image):
    
        image_float_np = np.float32(image) / 255
        # define the torchvision image transforms
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        input_tensor = transform(image)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        # Add a batch dimension:
        input_tensor = input_tensor.unsqueeze(0)

        return input_tensor, image_float_np

    def draw_segmentation_map(self, image, masks, boxes, labels):
        alpha = 1
        beta = 0.6
        gamma = 0
        result_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        masks = [mask.squeeze() for mask in masks]

        for i, mask in enumerate(masks):
            color = self.class_to_colors[labels[i]]
            red_map, green_map, blue_map = [np.zeros_like(mask, dtype=np.uint8) for _ in range(3)]        
            red_map[mask == 1], green_map[mask == 1], blue_map[mask == 1] = color
            result_masks[mask == 1] = 255
        return result_masks

    def get_outputs_nms(self, input_tensor,image,img_name,  score_threshold = 0.5, iou_threshold = 0.5):
        #start=timer()
        with torch.no_grad():
            # forward pass of the image through the model
            outputs = self.model(input_tensor)
        #print(timer()-start)
        r= []
        for j in range(len(outputs)):
            boxes = outputs[j]['boxes']
            labels = outputs[j]['labels']
            scores = outputs[j]['scores']
            masks = outputs[j]['masks']
            # Apply score threshold
            keep = scores > score_threshold
            boxes, labels, scores, masks = boxes[keep], labels[keep], scores[keep], masks[keep]
            keep = torchvision.ops.nms(boxes, scores, iou_threshold)
            boxes, labels, scores, masks = boxes[keep], labels[keep], scores[keep], masks[keep]
            scores = list(scores.detach().cpu().numpy())
            masks = list((masks>0.5).detach().cpu().numpy())
            boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in boxes.detach().cpu()]
            labels = list(labels.detach().cpu().numpy())
            labels = [self.class_names[i-1] for i in labels]
            
            result_masks = self.draw_segmentation_map(image[j], masks, boxes, labels)
            total_brown_pixels = self.get_brown_pixel_cnt(image[j], img_name[j])
            df = self.quantify_plaques(pd.DataFrame(), img_name[j], result_masks, boxes, labels, scores, total_brown_pixels)
            #print(df)
            r.append(df)
        return pd.concat(r, ignore_index=True)
    
    def quantify_plaques(self, df, img_name, result_masks, boxes, labels, scores, total_brown_pixels):
        '''This function takes masks image and generates attributes like plaque count, area, and eccentricity'''
        plaque_counts = {
            "Cored": 0,
            "Coarse-Grained": 0,
            "Diffuse": 0,
            "CAA": 0
        }
        img_x, img_y = img_name.split("_")[0], img_name.split("_")[2]
        for i, label in enumerate(labels):
            if len(boxes) == 0:
                continue
            
            x1, x2 = boxes[i][0][1], boxes[i][1][1]
            y1, y2 = boxes[i][0][0], boxes[i][1][0]
            
            cropped_img_mask = result_masks[x1:x2, y1:y2]
            
            _, bw_img = cv2.threshold(cropped_img_mask, 0, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)
            regions = regionprops(closing)
            mask_present = 1 if 0 in np.unique(closing) else 0
            
            #qupath_coord_x = self.x +img_x +  (y1 + y2)//2
            
            qupath_coord_x1 = self.x +img_x +  y1
            qupath_coord_x2 = self.x +img_x +  y2
            qupath_coord_y1 = self.y + img_y + x1
            qupath_coord_y2 = self.y + img_y + x2
            
            #qupath_coord_y = self.y + img_y + (x1 + x2)//2

            for props in regions:
                plaque_counts[label] += 1
                data_record = { 'image_name': img_name,  'label': label, 'confidence': scores[i], 'brown_pixels': total_brown_pixels, 'core': plaque_counts["Cored"], 
                    'coarse_grained': plaque_counts["Coarse-Grained"], 'diffuse': plaque_counts["Diffuse"], 'caa': plaque_counts["CAA"], 'centroid': props.centroid, 
                    'eccentricity': props.eccentricity,  'area': props.area,  'equivalent_diameter': props.equivalent_diameter, 'mask_present': mask_present,
                    'qupath_coord_x1':qupath_coord_x1,'qupath_coord_x2':qupath_coord_x2,  'qupath_coord_y1':qupath_coord_y1,
                    'qupath_coord_y2':qupath_coord_y2}
                df = pd.concat([df, pd.DataFrame.from_records([data_record])], ignore_index=True)

        return df
    
    
    def get_brown_pixel_cnt(self, img, img_name):
        # Separate the stains from the IHC image
        ihc_hed = rgb2hed(img)

        # Create an RGB image for the DAB stain (brown color)
        null_channel = np.zeros_like(ihc_hed[:, :, 0])
        ihc_d = hed2rgb(np.stack((null_channel, null_channel, ihc_hed[:, :, 2]), axis=-1)).astype('float32')

        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(ihc_d, cv2.COLOR_RGB2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Count brown pixels (pixels with intensity below 0.35)
        brown_pixel_count = np.sum(gray_blurred < 0.35)
        
        return brown_pixel_count

    def create_dataloader(self, arrays, batch_size=1, shuffle=False, num_workers=0):
        dataset = NumpyArrayDataset(arrays)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader

    def process_tile(self, batch):
        img_name, input_tensor, image = batch["key"],batch["tensor"],batch["array"]
        df = self.get_outputs_nms(input_tensor,image,img_name, score_threshold = 0.6, iou_threshold = 0.5)
        return df

    
    def generate_results_mpp(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(device)
        
        image = np.array(self.image_buffer)
        
        # Add error checking for image
        if image.size == 0:
            raise ValueError("Image buffer is empty")
        
        if len(image.shape) != 3:
            raise ValueError(f"Invalid image shape: {image.shape}. Expected 3 dimensions.")
        
        height, width, channels = image.shape 
        if height != 3072 or width != 3072:
            raise ValueError(f"Invalid image dimensions: {height}x{width}. Expected 3072x3072.")
        
        # Tile size
        tile_size = 1024
        # Loop to create the tiles
        
        tiles = []
        tile_names = []
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Crop the image to create a tile
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                tile_names.append(str(x)+"_x_"+str(y)+"_y")
            
        numpy_arrays = [(tile_names[i],tiles[i]) for i in range(len(tiles))]
        
        dataloader = self.create_dataloader(numpy_arrays, batch_size=9, shuffle=False, num_workers=0)
        results =[]
        for batch in dataloader:
            df = self.process_tile(batch)
            results.append(df)
        if len(results)>0:
            final_df = pd.concat(results, ignore_index=True)
        else:
            # Return an empty DataFrame if no results
            print("-------Empty DataFrame")
            final_df = pd.DataFrame(columns=self.column_names)
        return final_df

import base64
import io
from PIL import Image

def process_image(job):
    try:
        job_input = job["input"]
        x = job_input["x"]
        y = job_input["y"]
        image_buffer = job_input["Image_buffer"]
        
        # Add error checking for Image_buffer
        if not image_buffer:
            raise ValueError("image_buffer is empty")
        
        # Decode base64 image_buffer
        decoded_image = base64.b64decode(image_buffer)
        image = Image.open(io.BytesIO(decoded_image))
        image_buffer = np.array(image)
        
        # Add error checking for decoded image
        if image_buffer.size == 0:
            raise ValueError("Decoded Image buffer is empty")
        
        if len(image_buffer.shape) != 3:
            raise ValueError(f"Invalid image shape: {image_buffer.shape}. Expected 3 dimensions.")
        
        height, width, channels = image_buffer.shape 
        if height != 3072 or width != 3072:
            raise ValueError(f"Invalid image dimensions: {height}x{width}. Expected 3072x3072.")
        
        model_name = "/workspace/Projects/Amyb_plaque_detection/models/yp2mf3i8_epoch=108-step=872.ckpt"
        #model_name = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/runpod_mrcnn_models/yp2mf3i8_epoch=108-step=872.ckpt"
        model = LitMaskRCNN.load_from_checkpoint(model_name)
        
        explain = ExplainPredictions(model, x, y, image_buffer, detection_threshold=0.6)
        final_df = explain.generate_results_mpp()
        
        # Convert DataFrame to dictionary for JSON serialization
        result = final_df.to_json(orient='records')
        
        return result
    except Exception as e:
        import traceback
        error_message = f"Error in process_image: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_message}
        if height != 3072 or width != 3072:
            raise ValueError(f"Invalid image dimensions: {height}x{width}. Expected 3072x3072.")

        model_name = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/runpod_mrcnn_models/yp2mf3i8_epoch=108-step=872.ckpt" 
        model = LitMaskRCNN.load_from_checkpoint(model_name)
        
        explain = ExplainPredictions(model, x, y, image_buffer, detection_threshold=0.6)
        final_df = explain.generate_results_mpp()
        
        # Convert DataFrame to dictionary for JSON serialization
        result = final_df.to_json(orient='records')
        
        return result
    except Exception as e:
        import traceback
        error_message = f"Error in process_image: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_message}

runpod.serverless.start({"handler": process_image})
