#!/bin/bash
if [ -z "$1" ]; then
        comment="Fix bugs"
else
        comment=$1
fi
git add . && git commit -m "$comment" && git push
