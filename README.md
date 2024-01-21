## auto qat  

**auto qat** is a tool, which automatically insert qdq nodes in torch nn.module, calibrate models, and finetune.   


## Architecture  
* lib       `impl of auto quantize`      
* __models  `models for unit testing, include structure definition of networks`   

## Requirements   
```

```

## Build and Installation 
```  
cd pytorch-quantization_v2.1.2

python3 setup.py build 

# you can try install follow !!!

pip3 install loguru
pip3 install onnx==1.8.1
pip3 install onnx-simplifier==0.4.10


```
## Run demo   
```
cd auto_qat 
python3  ./qat_resnet50.py
```

## Guide and Pipeline     
* auto qdq insert  
* calib  
* load pretrained weights 
* finetune  
* export quant onnx  
* qat vs ptq benchmark  


## Known Issues   


## Reference   
[1] https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization    
[2] https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html  
[3] https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8   
[4] https://google.github.io/styleguide/pyguide.html   
[5] https://github.com/zh-google-styleguide/zh-google-styleguide/tree/master/google-python-styleguide   
[6] https://elinux.org/index.php?title=TensorRT/CommonErrorFix   
[7] https://developer.nvidia.com/zh-cn/blog/tensorrt-measuring-performance-cn/   
[8] https://developer.nvidia.com/blog/exploring-tensorrt-engines-with-trex/   

## Copyright and License    

```
auto qat is provided under the [Apache-2.0 license](LICENSE).  
```
