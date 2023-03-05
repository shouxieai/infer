#!/bin/bash

find ./src -regex '.*\.\(cpp\|hpp\|cc\|cxx\|cu\|cuh\|h\)' | xargs clang-format -i