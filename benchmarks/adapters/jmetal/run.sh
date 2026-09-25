#!/usr/bin/env bash
# Runs the jMetal adapter (build it first with build.sh):
#   run.sh <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#   run.sh values <problem> <size>
#   run.sh --version
# One thread (rule 4.3, see docs/benchmarks/libraries/jmetal.md): jMetal evaluates sequentially
# (Bench.java); the serial garbage collector collects on the calling thread; and -Xbatch
# (-XX:-BackgroundCompilation) makes the calling thread wait while the JIT compiles a method, so
# the JIT's compiler threads never run beside it.
exec "$HOME/opt/jdk/bin/java" -XX:+UseSerialGC -Xbatch \
    -cp "$HOME/bench-targets/jmetal:$HOME/opt/jmetal/*" Bench "$@"
