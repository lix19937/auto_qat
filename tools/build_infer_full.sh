#/**************************************************************
# * @Copyright: 2021-2022 Copyright SAIC 
# * @Author: lijinwen 
# * @Date: 2022-01-10 13:49:25 
# * @Last Modified by: lijinwen
# * @Last Modified time: 2022-01-13 11:22:47
# **************************************************************/
#!/bin/bash

# ./build_infer_full.sh x86_64 best resnet.onnx
# ./build_infer_full.sh aarch64 best resnet.onnx


###############################################################################
if [ -z "$#" ];then
  echo -e "\nUsage: ./build_infer.sh x86_64 [int8 fp16 fp32 best] onnx"
  "or ./build_infer.sh aarch64 [int8 fp16 fp32 best] onnx\n"
  exit 1
fi

os_aarch=${1}
type=${2}
onnx=${3}
basename=${onnx:0:-5}
outpath=`pwd`/${basename}/${os_aarch}/${type}

if [ ${type} = "fp32" ] ; then 
  type="noTF32"
fi
echo ${type}

echo ${outpath}

chmod 777 -R `pwd`

if [ ! -d ${outpath} ]; then
  mkdir -p ${outpath}
  chmod 777 -R ${outpath}
fi 

outname=`date +%Y%m%d-%H%M%S`

# --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT \

/usr/src/tensorrt/bin/trtexec --onnx=./${onnx} --dumpProfile    \
--workspace=90120 --verbose  --separateProfileRun \
--${type}   \
--saveEngine=${outpath}/${outname}.plan   \
--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT \
--useCudaGraph   \
--exportOutput=$outpath/$outname.json \
--profilingVerbosity=detailed \
--exportProfile=${outpath}/${outname}.profile.json \
--exportLayerInfo=${outpath}/${outname}.layerinfo.json \
  2>&1 | tee  ${outpath}/${outname}_build.log  

# outname=20230411-065119

/usr/src/tensorrt/bin/trtexec --onnx=./${onnx}     \
--verbose  --dumpProfile --separateProfileRun --${type}   \
--loadEngine=${outpath}/${outname}.plan   \
--useCudaGraph   \
  2>&1 | tee  ${outpath}/${outname}_infer.log  

chmod 777 -R `pwd`

echo ${type}" ===> dy done"

