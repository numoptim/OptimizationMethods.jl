#!/bin/bash

julia --project make.jl
cd build
python3 -m http.server
