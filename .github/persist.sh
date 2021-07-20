#!/bin/bash
if [[ -d "${PERSIST_DIR}/ccache-cachedir" ]]; then
    echo "The Ccache directory exists at ${PERSIST_DIR}/ccache-cachedir"
else
    echo "Creating Ccache directory at ${PERSIST_DIR}/ccache-cachedir"
    mkdir -p ${PERSIST_DIR}/ccache-cachedir
    envsubst > ${PERSIST_DIR}/ccache-config <<EOF
    max_size = 20G
    cache_dir = ${PERSIST_DIR}/ccache-cachedir
EOF
fi
