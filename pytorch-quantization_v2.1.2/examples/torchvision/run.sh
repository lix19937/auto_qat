python3  ./nv_classification_flow.py   \
--data-dir=./deps/image1000test200  \
--num-calib-batch=0 \
--num-finetune-epochs=0 \
--calibrator=histogram \
--out-dir=./tmp \
--ckpt-path=./deps/resnet18-5c106cde.pth \
--model-name=resnet18 \
--percentile=99.9

# --pretrained  &  --ckpt-path  mutual exclusion


 