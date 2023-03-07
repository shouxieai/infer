#!/bin/bash

find ./src -regex '.*\.\(cpp\|hpp\|cc\|cxx\|cu\|cuh\|h\)' | xargs clang-format -i \
    -style="{BasedOnStyle: google, IndentWidth: 2, ColumnLimit: 100}"