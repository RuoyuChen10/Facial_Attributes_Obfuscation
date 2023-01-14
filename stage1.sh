# ArcFace
python -m utils.score \
    --mode arcface \
    --dataset-root ./align_arc \
    --data-list ./images/label.txt \
    --num-classes 8631 \
    --pre-trained ./pretrained/ArcFace-8631.pth \
    --cls-device cuda:0 \
    --seg-device cuda:0 \
    --save-dir results/arc_mode

# VGGFace
python -m utils.score \
    --mode vggface \
    --dataset-root ./align_vgg \
    --data-list ./images/label.txt \
    --num-classes 8631 \
    --pre-trained ./pretrained/resnet50_scratch_weight.pkl \
    --cls-device cuda:0 \
    --seg-device cuda:0 \
    --save-dir results/vgg_mode