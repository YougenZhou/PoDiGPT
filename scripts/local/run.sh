#!/bin/bash

if [ $# != 1 ]; then
    echo "usage: bash $0 config_file"
    exit -1
fi

config_file=$1
source ${config_file}

#if [ ${log_dir:-"" != ""} ]; then
#    rm ${log_dir}/*.*
#fi

export PYTHONPATH=$(dirname "$0")/../..:${PYTHONPATH:-}

${run_script} ${config_file}