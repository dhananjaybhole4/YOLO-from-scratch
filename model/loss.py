import torch
import torch.nn as nn

from util.util import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S = 25, B = 2, C = 11):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.mse = nn.MSELoss(reduction = "sum")

    def forward(self, predictions, target):
        # reshaping the prediction to S by S with (C+2B) features
        predictions = predictions.reshape(-1, self.S, self.S, self.C + 5*self.B)

        iou_b1 = intersection_over_union(predictions[..., 12:16], target[..., 12:16])
        iou_b2 = intersection_over_union(predictions[..., 17:21], target[..., 12:16])
        
        # this method is used to get which bbox has highestt intersection over union
        ious = torch.cat((iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)), dim = 0)

        # choosen_bbox is a mask which get value 0 if bbox1 have higher iou, gets value 1 if bbox2 have higher iou
        max_iou, choosen_bbox = torch.max(ious, dim = 0)

        exist_obj = target[..., 11:12]

        ## -------------------- ##
        ## Box coordinate loss  ##  
        ## -------------------- ##
 
        # exist_obj make sure that bbox is there only if object is there and following line gets best bbox from the two predicted bbox from the model    
        bbox_predictions = exist_obj*(predictions[..., 12:16]*(1 - choosen_bbox) + predictions[..., 17:21]*(choosen_bbox))
        bbox_targets = exist_obj*target[..., 12:16]

        # updated the width and height with their square root as per the loss fn in Yolov1 paper
        bbox_predictions[..., 2:4] = torch.sign(bbox_predictions[..., 2:4]*torch.sqrt(torch.abs(bbox_predictions[...,2:4])))
        bbox_targets[..., 2:4] = torch.sqrt(bbox_targets[..., 2:4])

        # loss considering bbox paramters where object exist
        box_loss = self.lambda_coord*self.mse(bbox_predictions[..., 0:4], bbox_targets[..., 0:4])

        ## ---------------------------------------- ##
        ##  object loss if there is object present  ##
        ## ---------------------------------------- ##

        # obj presence prediction of choosen bbox which ahs highest iou with target bbox where obj exist
        obj_predictions = exist_obj*((1 - choosen_bbox)*predictions[..., 11:12] + choosen_bbox*predictions[..., 16:17])
        obj_targets = exist_obj*(target[..., 11:12])

        # loss of probability showing object present where object exist
        object_loss = self.mse(obj_predictions, obj_targets)
    
        ## --------------------------------------------- ##
        ##  no object loss if there is no object present ##
        ## --------------------------------------------- ##

        # loss of probability showing object present where object does not exist
        no_object_loss = self.lambda_noobj*self.mse((1 - exist_obj)*predictions[..., 11:12], (1 - exist_obj)*(target[..., 11:12]))

        no_object_loss += self.lambda_noobj*self.mse((1 - exist_obj)*predictions[..., 16:17], (1 - exist_obj)*(target[..., 11:12]))
        
        ## ---------- ##
        ## class loss ##
        ## ---------- ##

        class_loss = self.mse(exist_obj*predictions[..., :11], exist_obj*target[..., :11])


        loss = (box_loss              # loss due to error in bbox parameters
                + object_loss         # loss due to object present probability when object exist
                + no_object_loss      # loss due to object present probability when object doesnt exist
                + class_loss)         # loss due to class probability 
        print(f" box loss = {box_loss} | object loss = {object_loss} | no object loss = {no_object_loss} | class_loss = {class_loss}")
        return loss