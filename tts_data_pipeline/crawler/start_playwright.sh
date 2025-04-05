#!/bin/env bash

docker run -p 3000:3000 --rm --init -d --workdir /home/pwuser --user pwuser mcr.microsoft.com/playwright:v1.51.1-noble \
    /bin/sh -c "npx -y playwright@1.51.0 run-server --port 3000 --host 0.0.0.0"