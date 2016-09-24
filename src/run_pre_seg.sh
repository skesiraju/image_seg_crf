#!/bin/bash

data_dir='/home/santosh/Downloads/VOCdevkit/VOC2008/'

ppm_dir=${data_dir}'PPMImages/'
patch_dir=${data_dir}'patches/'

seg_src=$PWD/segment/segment

sigmas=(0.5 0.75 1.25)
ks=(500 500 700)
ms=(50 200 1200)

for i in $(seq 0 2); do
        
    out_dir=${patch_dir}'s_'${sigmas[$i]}'_k_'${ks[$i]}'_m_'${ms[$i]}
    mkdir -p ${out_dir}

done


for in_ppm in `ls ${ppm_dir}*`; do

    tmp_b=`basename ${in_ppm}`
    base=`echo $tmp_b | cut -d'.' -f1`
    
    for i in $(seq 0 2); do
        
        out_dir=${patch_dir}'s_'${sigmas[$i]}'_k_'${ks[$i]}'_m_'${ms[$i]}
        out_ppm=${out_dir}/${base}.ppm
        
        # echo "${seg_src} ${sigmas[$i]} ${ks[$i]} ${ms[$i]} ${in_ppm} ${out_ppm}"

        if [ ! -e ${out_ppm} ]; then
            echo $in_ppm
            ${seg_src} ${sigmas[$i]} ${ks[$i]} ${ms[$i]} ${in_ppm} ${out_ppm}
        fi
    
    done    

done

echo 'Pre-segmentation done.'
