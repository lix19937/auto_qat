## model graph with q-dq setting but no calib and no load ckpt 

* print(model)   

```  
ResNet(
  (conv1): QuantConv2d(
    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
    (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
  )
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): QuantConv2d(
        64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
    )
    (1): BasicBlock(
      (conv1): QuantConv2d(
        64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): QuantConv2d(
        64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): QuantConv2d(
          64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
          (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
          (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
        )
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
    )
    (1): BasicBlock(
      (conv1): QuantConv2d(
        128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): QuantConv2d(
        128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): QuantConv2d(
          128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
          (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
          (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
        )
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
    )
    (1): BasicBlock(
      (conv1): QuantConv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): QuantConv2d(
        256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): QuantConv2d(
          256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
          (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
          (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
        )
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
    )
    (1): BasicBlock(
      (conv1): QuantConv2d(
        512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): QuantLinear(
    in_features=512, out_features=1000, bias=True
    (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
    (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
  )
)
```

## input & weight  
```
(_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=dynamic calibrator=HistogramCalibrator scale=1.0 quant)
(_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=dynamic calibrator=MaxCalibrator scale=1.0 quant)
``` 
input use **HistogramCalibrator**  
weight use **MaxCalibrator** 

## after calib  

```
ResNet(
  (conv1): QuantConv2d(
    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=2.6400 calibrator=HistogramCalibrator scale=1.0 quant)
    (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0000, 1.0165](64) calibrator=MaxCalibrator scale=1.0 quant)
  )
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): QuantConv2d(
        64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=4.2149 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.1761, 0.7993](64) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=2.5036 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0919, 0.4879](64) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=4.2149 calibrator=HistogramCalibrator scale=1.0 quant)
    )
    (1): BasicBlock(
      (conv1): QuantConv2d(
        64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=4.4724 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.1268, 0.6491](64) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=2.2714 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.1042, 0.3806](64) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=4.4724 calibrator=HistogramCalibrator scale=1.0 quant)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): QuantConv2d(
        64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=5.2465 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.1239, 0.3405](128) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=2.3208 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.1058, 0.4272](128) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): QuantConv2d(
          64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
          (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=5.2465 calibrator=HistogramCalibrator scale=1.0 quant)
          (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0416, 0.7802](128) calibrator=MaxCalibrator scale=1.0 quant)
        )
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=2.8600 calibrator=HistogramCalibrator scale=1.0 quant)
    )
    (1): BasicBlock(
      (conv1): QuantConv2d(
        128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=3.8342 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.1004, 0.4387](128) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=2.5667 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0746, 0.3557](128) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=3.8342 calibrator=HistogramCalibrator scale=1.0 quant)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): QuantConv2d(
        128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=4.5324 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0895, 0.3920](256) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=2.6470 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0806, 0.3335](256) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): QuantConv2d(
          128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
          (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=4.5324 calibrator=HistogramCalibrator scale=1.0 quant)
          (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0505, 0.2677](256) calibrator=MaxCalibrator scale=1.0 quant)
        )
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=1.4978 calibrator=HistogramCalibrator scale=1.0 quant)
    )
    (1): BasicBlock(
      (conv1): QuantConv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=4.1628 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0727, 0.2960](256) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=2.3853 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0392, 0.3295](256) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=4.1628 calibrator=HistogramCalibrator scale=1.0 quant)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): QuantConv2d(
        256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=3.5338 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0637, 0.3826](512) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=2.0715 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0559, 0.3487](512) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): QuantConv2d(
          256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
          (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=3.5338 calibrator=HistogramCalibrator scale=1.0 quant)
          (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0322, 0.7472](512) calibrator=MaxCalibrator scale=1.0 quant)
        )
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=2.7359 calibrator=HistogramCalibrator scale=1.0 quant)
    )
    (1): BasicBlock(
      (conv1): QuantConv2d(
        512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=4.0201 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0537, 0.2658](512) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): QuantConv2d(
        512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=1.9455 calibrator=HistogramCalibrator scale=1.0 quant)
        (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0485, 0.2718](512) calibrator=MaxCalibrator scale=1.0 quant)
      )
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (residual_quantizer): TensorQuantizer(8bit fake per-tensor amax=4.0201 calibrator=HistogramCalibrator scale=1.0 quant)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): QuantLinear(
    in_features=512, out_features=1000, bias=True
    (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=8.5265 calibrator=HistogramCalibrator scale=1.0 quant)
    (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.2172, 0.7152](1000) calibrator=MaxCalibrator scale=1.0 quant)
  )
)
```