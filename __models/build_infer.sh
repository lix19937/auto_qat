#/**************************************************************
# * @Copyright: 2021-2022 Copyright SAIC 
# * @Author: lijinwen 
# * @Date: 2022-01-10 13:49:25 
# * @Last Modified by: lijinwen
# * @Last Modified time: 2022-01-13 11:22:47
# **************************************************************/
#!/bin/bash

# ./build_infer.sh  x86_64  best  new_erfnet_encoder.onnx


###############################################################################
if [ -z "$#" ];then
  echo -e "\nUsage: ./build_infer.sh x86_64 [int8 fp16 fp32 best] onnx"
  "or ./build_infer.sh aarch64 [int8 fp16 fp32 best] onnx\n"
  exit 1
fi

if [ "$1" = "x86_64" ] ;then
  # docker 
  deps=`pwd`
else
  deps=/home/pp-cem/nfs/ljw/155/cuda_ops/build
fi

export LD_LIBRARY_PATH=${deps}

os_aarch=${1}
type=${2}
onnx=${3}
basename=${onnx:0:-5}
outpath=`pwd`/${basename}/${os_aarch}/${type}

if [ ${type} = "fp32" ] ; then 
  type="noTF32"
fi
echo ${type}
echo ${deps}
echo ${outpath}

chmod 777 -R `pwd`

if [ ! -d ${outpath} ]; then
  mkdir -p ${outpath}
  chmod 777 -R ${outpath}
fi 

outname='test'

#--loadInputs=${inpairs} \

/usr/src/tensorrt/bin/trtexec --onnx=./${onnx} --dumpProfile    \
--workspace=90120 --verbose  --separateProfileRun \
--${type}   \
--saveEngine=${outpath}/${outname}.plan   \
--useCudaGraph   \
--profilingVerbosity=detailed \
--exportProfile=${outpath}/${outname}.profile.json \
--exportLayerInfo=${outpath}/${outname}.layerinfo.json \
  2>&1 | tee  ${outpath}/${outname}_build.log  


chmod 777 -R `pwd`


echo ${type}" ===> dy done"

