import torch.nn as nn
import torch
import numpy as np
from PIL import Image 
from torchvision import transforms 
import torch.nn.functional as F 
from torch.autograd import Variable
import cv2 
from tqdm import tqdm
anchors = torch.tensor([
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],])
COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]
img = Image.open("dog.jpg")
img = img.resize((416,416))
print(img.size)
convert_tensor = transforms.ToTensor()
cimg = convert_tensor(img)

config = [
     [32, 3, 1],
     [64, 3, 2],
     ['Residual',1],
     [128, 3, 2],
     ['Residual',2],
     [256, 3, 2],
     ['Residual',8],
     [512, 3, 2],
     ['Residual',8],
     [1024, 3, 2],
     ['Residual',4],
     [512, 1, 1],
     [1024, 3, 1],
     ['S'],
     [256, 1, 1],
     ['U'],
     [256, 1, 1],
     [512, 3, 1],
     ['S'],
     [128, 1, 1],
     ['U'],
     [128, 1, 1],
     [256, 3, 1],
     ['S']
]


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, anchors_per_scale):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            ConvoBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            ConvoBlock(2*in_channels, (num_classes + 5) * 3, use_batch_normalization=False, kernel_size=1),
        )
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale

    def forward(self, x):
        return (
            self.pred(x)
                .reshape(x.shape[0], self.anchors_per_scale, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2)
        )
    
class ConvoBlock(nn.Module):
    def __init__ (self, input_channels, output_channels, use_batch_normalization = True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels,output_channels, bias = not use_batch_normalization, **kwargs)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_normalization
    def forward(self, x):
        if self.use_batch_norm:
            return self.leaky_relu(self.batch_norm(self.conv(x)))
        else:
            return self.leaky_relu(self.conv(x))

class ResidualLayerBlock(nn.Module):
    def __init__(self, channels, use_residual_layer = True, number_of_repeat = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(number_of_repeat):
            self.layers += [
                nn.Sequential(
                    ConvoBlock(channels, channels // 2, kernel_size=1),
                    ConvoBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]
        self.use_residual = use_residual_layer
        self.repeats = number_of_repeat
    def forward(self,x):
            for layer in self.layers:
                if self.use_residual:
                     x = layer(x) + x
                else:
                     x = layer(x)
            return x


class YOLO_Model(nn.Module):
     def __init__(self, input_channels = 3, number_of_classes = 20):
          super().__init__()
          self.numClasses = number_of_classes
          self.inChannels = input_channels
          self.layers = self.createLayers()
     def forward(self,x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualLayerBlock) and layer.repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs
     def createLayers(self):
          layers = nn.ModuleList()
          in_channels = self.inChannels

          for module in config:
               if module[0] == 'Residual':
                    # print(f'Creating Residual Block with num_repeats = {module[1]} for module = {module} in config')
                    num_repeats = module[1]
                    layers.append(ResidualLayerBlock(in_channels, number_of_repeat=num_repeats))
               elif module[0] == 'S':
                    layers += [
                        ResidualLayerBlock(in_channels, use_residual_layer=False, number_of_repeat=1),
                        ConvoBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.numClasses, anchors_per_scale= 3),
                    ]
                    in_channels = in_channels // 2
               elif module[0] == "U":
                    layers.append(
                        nn.Upsample(scale_factor=2),
                    )
                    in_channels = in_channels * 3
               else:
                    layers.append(ConvoBlock(in_channels, module[0], kernel_size = module[1], stride = module[2], padding = 1 if module[1] == 3 else 0))
                    in_channels = module[0]
               
          return layers
     def load_CNN_weights(self, ptr, block):

        conv_layer = block.conv
        if block.use_batch_norm:
            # Load BN bias, weights, running mean and running variance
            bn_layer = block.batch_norm
            num_b = bn_layer.bias.numel()  # Number of biases
            # Bias
            bn_b = torch.from_numpy(self.weights[ptr : ptr + num_b]).view_as(
                bn_layer.bias
            )
            bn_layer.bias.data.copy_(bn_b)
            ptr += num_b
            # Weight
            bn_w = torch.from_numpy(self.weights[ptr : ptr + num_b]).view_as(
                bn_layer.weight
            )
            bn_layer.weight.data.copy_(bn_w)
            ptr += num_b
            # Running Mean
            bn_rm = torch.from_numpy(self.weights[ptr : ptr + num_b]).view_as(
                bn_layer.running_mean
            )
            bn_layer.running_mean.data.copy_(bn_rm)
            ptr += num_b
            # Running Var
            bn_rv = torch.from_numpy(self.weights[ptr : ptr + num_b]).view_as(
                bn_layer.running_var
            )
            bn_layer.running_var.data.copy_(bn_rv)
            ptr += num_b
        else:
            # Load conv. bias
            num_b = conv_layer.bias.numel()

            conv_b = torch.from_numpy(self.weights[ptr : ptr + num_b]).view_as(
                conv_layer.bias
            )
            conv_layer.bias.data.copy_(conv_b)
            ptr += num_b
            # Load conv. weights
        num_w = conv_layer.weight.numel()
        conv_w = torch.from_numpy(self.weights[ptr : ptr + num_w]).view_as(
            conv_layer.weight
        )
        conv_layer.weight.data.copy_(conv_w)
        ptr += num_w
        return ptr

     def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(
                f, dtype=np.int32, count=5
            )  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            self.weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, ConvoBlock):
                ptr = self.load_CNN_weights(ptr, layer)

            elif isinstance(layer, ResidualLayerBlock):
                for i in range(layer.repeats):
                    ptr = self.load_CNN_weights(ptr, layer.layers[i][0])
                    ptr = self.load_CNN_weights(ptr, layer.layers[i][1])

            elif isinstance(layer, ScalePrediction):
                # print("Starting scale prediction route")
                cnn_block = layer.pred[0]
                last_block = layer.pred[1]
                ptr = self.load_CNN_weights(ptr, cnn_block)
                ptr = self.load_CNN_weights(ptr, last_block)


if __name__ == "__main__":
    num_classes = 20
    img_size = img.size[0]
    model = YOLO_Model(number_of_classes= num_classes)
    model.load_darknet_weights(weights_path="yolov3.weights")
    x = cimg.unsqueeze(0)
    out = model(x)
    predictions = out[0]
    scores = torch.sigmoid(predictions[..., 0:1])
    best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

    cell_indices = (
        torch.arange(13)
        .repeat(predictions.shape[0], 3, 13, 1)
        .unsqueeze(-1)
    )
    box_predictions = predictions[..., 1:5]
    x = 1 / 13 * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / 13 * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / 13 * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(1, 3 * 13 * 13, 6)
    converted_bboxes = converted_bboxes.tolist()
    for pred in converted_bboxes:
        for preds in pred:
            class_pred = preds[0]
            confidence = preds[1]
            if confidence > 0.4:
                print(COCO_LABELS[int(class_pred)])




    

