import os 
import cv2
import numpy as np
import argparse
import json
import pickle

import torch
import torchvision.transforms as transforms
from PIL import Image
from collections import OrderedDict

# import sys
# sys.path.append("../")

from models.vggface_models.resnet import resnet50
from models.face_parser import FaceParser, read_img
from models.iresnet import iresnet50, iresnet100
from models.AttributeNet import AttributeNet
from interpretability.grad_cam import GradCAM

from tqdm import tqdm

transforms = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

anonymized_hair = ["Receding Hairline"]
anonymized_eyebrows = ["Bushy Eyebrows", "Arched Eyebrows"]
anonymized_eye = ["Brown Eyes"] # Narrow_Eyes is only support on CelebA attribute, due to the machine hasn't repaired, it would be a replace and we will release soon.
anonymized_nose = ["Big Nose", "Pointy Nose"]
anonymized_lips = ["Big Lips"]
anonymized_attribute = anonymized_hair + anonymized_eyebrows + anonymized_eye + anonymized_nose + anonymized_lips


def parse_args():
    parser = argparse.ArgumentParser(description='Train paramters.')
    # general
    parser.add_argument('--dataset-root', type=str,
                        default='image',
                        help='')
    parser.add_argument('--data-list', type=str,
                        default='label.txt',
                        help='')
    parser.add_argument('--num-classes', type=int,
                        default = 8631,
                        # default=10177,
                        help='')
    parser.add_argument('--backbone', type=str,
                        default='resnet50',
                        choices=['resnet50','resnet100'],
                        help='')
    parser.add_argument('--mode', type=str,
                        default='arcface',
                        choices=['arcface','vggface'],
                        help='')
    parser.add_argument('--pre-trained', type=str,
                        default = "pretrained/ArcFace-r50-8631.pth",
                        # default = "../CelebA/pretrained/ArcFace-r50-10177.pth",
                        help='')
    parser.add_argument('--seg-pre-trained', type=str,
                        default = "pretrained/FaceParser.ckpt",
                        help='')
    parser.add_argument('--attr-pre-trained', type=str,
                        default = "pretrained/Face-Attributes2.pth",
                        help='')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        choices=['cuda','cpu'],
                        help='')
    parser.add_argument('--cls-device', type=str, default="cuda:0",
                        help='GPU device')
    parser.add_argument('--seg-device', type=str, default="cuda:0",
                        help='GPU device')
    parser.add_argument('--attr-device', type=str, default="cuda:0",
                        help='GPU device')
    parser.add_argument('--save-dir', type=str, default="results/VGGFace2",
                        help='GPU device')
    args = parser.parse_args()
    return args

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def get_last_conv_name(net):
    """
    Get the name of last convolutional layer
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            layer_name = name
    return layer_name

def define_backbone(mode, net_type, num_classes):
    """
    define the model
    """
    if mode == "arcface":
        if net_type == "resnet50":
            backbone = iresnet50(num_classes)
        elif net_type == "resnet100":
            backbone = iresnet100(num_classes)
    elif mode == "vggface":
        backbone = resnet50(num_classes=num_classes)
    return backbone

def Path_Image_Preprocessing(image_path, mode="arcface"):
    """Preprocessing method

    Args:
        image_path (_type_): Image path
        mode (str, optional): You can choose arcface, vggface and attribute. Defaults to "arcface".

    Returns:
        _type_: _description_
    """
    if mode == "arcface":
        data = Image.open(image_path)
        data = transforms(data)
        data = torch.unsqueeze(data,0)
    elif mode == "vggface" or mode == "attribute":
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        image = cv2.imread(image_path)
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        data = torch.tensor([image])
    return data

def load_face_recognition_model(args):
    """
    Face recognition model
    """
    recognition_net = define_backbone(args.mode, args.backbone, args.num_classes)

    if args.mode == "arcface" and args.pre_trained is not None and os.path.exists(args.pre_trained):
        # No related
        model_dict = recognition_net.state_dict()
        pretrained_param = torch.load(args.pre_trained, map_location=torch.device('cpu'))
        try:
            pretrained_param = pretrained_param.state_dict()
        except:
            pass

        new_state_dict = OrderedDict()
        for k, v in pretrained_param.items():
            if k in model_dict:
                new_state_dict[k] = v
                print("Load parameter {}".format(k))
            elif k[7:] in model_dict:
                new_state_dict[k[7:]] = v
                print("Load parameter {}".format(k[7:]))

        model_dict.update(new_state_dict)
        recognition_net.load_state_dict(model_dict)
        print("Success load pre-trained face model {}".format(args.pre_trained))

    elif args.mode == "vggface" and args.pre_trained is not None and os.path.exists(args.pre_trained):
        with open(args.pre_trained, 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        recognition_net.load_state_dict(weights, strict=True)

    # Set to device
    recognition_net.to(args.cls_device)
    recognition_net.eval()
    return recognition_net

def load_segmentation_model(args):
    """
    Segmentation Network
    """
    segmentation_net = FaceParser(num_classes=9, model_path=args.seg_pre_trained)
    segmentation_net.to(args.seg_device)
    segmentation_net.eval()

    return segmentation_net

def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def visualize_heatmap(image, mask):
    '''
    Save the heatmap of ones
    '''
    masks = norm_image(mask).astype(np.uint8)
    # mask->heatmap
    heatmap = cv2.applyColorMap(masks, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))    # same shape

    # merge heatmap to original image
    cam = 0.4*heatmap + 0.6*np.float32(image)
    return cam

def get_edge(image):
    """
    Get the edge of the image
    """
    kernel = np.ones((5,5),np.uint8)  
    erosion = cv2.erode(image,kernel,iterations = 2)
    edge = image - erosion
    return edge

def draw_edge(image, edge, color):
    """
    Given edge and draw into image.
    """
    # edge = cv2.resize(edge, (image.shape[1],image.shape[0]))    # same shape
    edge = edge / edge.max()    # noramlization
    edge = edge[:,:,np.newaxis] # (112, 112, 1)
    # edge = (edge>0.3).astype(np.uint8)
    color = np.array(color)
    color = color[np.newaxis, np.newaxis, :]    # (1,1,3)
    colored_edge = color * edge

    image_w_edge = image * (1-edge) + colored_edge

    return image_w_edge

def draw_image(image, mouth_edge, eyebrows_edge, eyes_edge, hair_edge, nose_edge, skin_edge):
    """
    Put the edge on the image.
    edge range 0-255
    """
    # image = cv2.resize(image, (mouth_edge.shape[1],mouth_edge.shape[0]))    # same shape
    image = draw_edge(image, mouth_edge, [60, 158, 226])
    image = draw_edge(image, eyebrows_edge, [226, 224, 86])
    image = draw_edge(image, eyes_edge, [226, 130, 106])
    image = draw_edge(image, hair_edge, [226, 116, 174])
    image = draw_edge(image, nose_edge, [102, 226, 132])
    image = draw_edge(image, skin_edge, [45, 230, 254])
    return image

def visualization_by_edge(image_path, salience_map, mouth, hair, eyes, eyebrows, nose, skin):
    """
    One type of visualization
    """
    # get the edge
    mouth_edge = get_edge(mouth*255)
    hair_edge = get_edge(hair*255)
    eyes_edge = get_edge(eyes*255)
    eyebrows_edge = get_edge(eyebrows*255)
    nose_edge = get_edge(nose*255)
    skin_edge = get_edge(skin*255)
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, (mouth_edge.shape[1],mouth_edge.shape[0]))    # same shape
    
    saliency_map_image = visualize_heatmap(image, salience_map)
    
    image = draw_image(saliency_map_image, mouth_edge, eyebrows_edge, eyes_edge, hair_edge, nose_edge, skin_edge)

    return image

def GradCAM_visualization(image_path, salience_map):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (salience_map.shape[1], salience_map.shape[0]))    # same shape
    
    saliency_map_image = visualize_heatmap(image, salience_map)
    return saliency_map_image


def calculate_score(salience_map, segmentation):
    """
    The average response values of different parts were calculated
    """
    score = (segmentation * salience_map).sum() / (segmentation.sum() + 1e-10)

    return score

def hierarchy_visualization(salience_map, image_path,
    mouth, hair, eyes, eyebrows, nose, skin,
    mouth_score, hair_score, eyes_score, eyebrows_score, nose_score, skin_score):
    heatmap = np.zeros_like(salience_map)
    heatmap = heatmap + mouth * mouth_score
    heatmap = heatmap + eyebrows * eyebrows_score
    heatmap = heatmap + eyes * eyes_score
    heatmap = heatmap + hair * hair_score
    heatmap = heatmap + nose * nose_score
    heatmap = heatmap + skin * skin_score

    image = cv2.imread(image_path)
    image = cv2.resize(image, (heatmap.shape[1],heatmap.shape[0]))

    # mask->heatmap
    heatmap = norm_image(heatmap).astype(np.uint8)
    heatmap = 255 - heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_AUTUMN)

    heatmap = np.float32(heatmap)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))    # same shape

    # merge heatmap to original image
    cam_hiera = 0.4*heatmap + 0.6*np.float32(image)

    return cam_hiera

def sort_face_part_score(face_part_score):
    tmp = sorted(face_part_score.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    face_part_score = {}
    part_sort = []
    for i in range(len(tmp)):
        face_part_score[tmp[i][0]] = tmp[i][1]
        part_sort.append(tmp[i][0])
    return face_part_score, part_sort

def compute_activated_attribute(face_part_score, part_sort, selected_attr_):
    index = np.argwhere(selected_attr_ == True)
    activated_attr_name = np.array(anonymized_attribute)[index.reshape(index.shape[0])]

    avtivated_attribute = check_activated(face_part_score, part_sort, activated_attr_name)
    return avtivated_attribute

def check_activated(face_part_score, part_sort, activated_attr_name):
    avtivated_attribute = {}
    for part in part_sort:
        part_list = return_list(part)
        for activated_attr_name_ in activated_attr_name:
            if activated_attr_name_ in part_list:
                avtivated_attribute[activated_attr_name_] = face_part_score[part]
    return avtivated_attribute

def return_list(part):
    if part == "hair":
        return anonymized_hair
    elif part == "eyebrows":
        return anonymized_eyebrows
    elif part == "eyes":
        return anonymized_eye
    elif part == "nose":
        return anonymized_nose
    elif part == "lips":
        return anonymized_lips


def main(args):
    # save path
    mkdir(args.save_dir)
    img_save_dir = os.path.join(args.save_dir, "Img")
    mkdir(img_save_dir)

    # different device
    cls_device = torch.device(args.cls_device)
    seg_device = torch.device(args.seg_device)
    attr_device = torch.device(args.attr_device)

    # face recognition model
    recognition_net = load_face_recognition_model(args)

    # segmentation network
    segmentation_net = load_segmentation_model(args)

    # attribute network
    attribute_net = AttributeNet(pretrained = args.attr_pre_trained)
    attribute_net.to(attr_device)
    attribute_net.set_idx_list(attribute = anonymized_attribute)

    # Grad-CAM
    layer_name = get_last_conv_name(recognition_net)
    cam = GradCAM(recognition_net, layer_name)

    with open(args.data_list, "r") as f:
        image_paths = f.readlines()

    # json file
    Json_file = {}

    for image_path in tqdm(image_paths):
        sub_json = {}
        #### For Recognition and Grad-CAM
        # ID
        gt_id = int(image_path.split(" ")[-1].strip())

        related_image_path = image_path.split(" ")[0]
        image_path = os.path.join(args.dataset_root, related_image_path)

        people_name = related_image_path.split("/")[0]
        sub_img_save_dir = os.path.join(img_save_dir, people_name)
        mkdir(sub_img_save_dir)

        sub_json["ID"] = gt_id
        sub_json["ImagePath"] = image_path

        image = Path_Image_Preprocessing(image_path, args.mode)
        image_attr = Path_Image_Preprocessing(image_path, "attribute")
        salience_map, id, scores = cam(image.to(cls_device))  # cam salience_map shape(112,112)

        ### For Attribute
        predicted_attr_results = attribute_net(image_attr.to(attr_device))
        selected_attr_ = (predicted_attr_results[0] > 0.5).cpu().numpy()
        attribute_score = {}
        for i in range(len(anonymized_attribute)):
            attribute_score[anonymized_attribute[i]] = predicted_attr_results[0][i].cpu().item()
            
        #### For Segmentation
        image = read_img(image_path)
        parsed_face = segmentation_net(image.to(seg_device))

        # predicted segmentation images
        seg_image = parsed_face.cpu().numpy()[0]
        seg_image = (seg_image>0.5).astype(np.uint8)
        
        # segmentation
        mouth = seg_image[1]; eyebrows = seg_image[2]; eyes=seg_image[3];
        hair = seg_image[4]; nose = seg_image[5]; skin = seg_image[6]

        # edge_visualization = visualization_by_edge(image_path, salience_map, mouth, hair, eyes, eyebrows, nose, skin)
        ## GradCAM visualization
        salience_map = cv2.resize(salience_map, (seg_image.shape[2], seg_image.shape[1]))   # (512, 512)
        saliency_map_image = GradCAM_visualization(image_path, salience_map)
        
        # calculate score
        mouth_score = calculate_score(salience_map, mouth)
        eyebrows_score = calculate_score(salience_map, eyebrows)
        eyes_score = calculate_score(salience_map, eyes)
        hair_score = calculate_score(salience_map, hair)
        nose_score = calculate_score(salience_map, nose)
        skin_score = calculate_score(salience_map, skin)

        # Visualization
        PartScoreImage = hierarchy_visualization(salience_map, image_path,
            mouth, hair, eyes, eyebrows, nose, skin,
            mouth_score, hair_score, eyes_score, eyebrows_score, nose_score, skin_score)

        face_part_score = {}
        face_part_score["lips"] = mouth_score
        face_part_score["eyebrows"] = eyebrows_score
        face_part_score["eyes"] = eyes_score
        face_part_score["hair"] = hair_score
        face_part_score["nose"] = nose_score
        # face_part_score["skin"] = skin_score

        face_part_score, part_sort = sort_face_part_score(face_part_score)
        avtivated_attribute = compute_activated_attribute(face_part_score, part_sort, selected_attr_)
        
        sub_json["ScoreSort"] = face_part_score
        sub_json["AttributePredictedScore"] = attribute_score
        sub_json["AvtivatedAttribute"] = avtivated_attribute

        saliency_map_image_path = os.path.join(
            sub_img_save_dir, related_image_path.replace(".jpg", "-gradcam.jpg")
        )
        PartScoreImage_path = os.path.join(
            sub_img_save_dir, related_image_path.replace(".jpg", "-part-score.jpg")
        )

        sub_json["GradCAMPath"] = saliency_map_image_path
        sub_json["FacePartScorePath"] = PartScoreImage_path

        cv2.imwrite(saliency_map_image_path, saliency_map_image)
        cv2.imwrite(PartScoreImage_path, PartScoreImage)

        Json_file[related_image_path] = sub_json


    with open(os.path.join(args.save_dir, "Record.json"), "w") as f:
        f.write(json.dumps(Json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
    
    return 

if __name__ == "__main__":
    args = parse_args()
    main(args)