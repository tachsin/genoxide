#!/usr/bin/env bash
# Runs the Jenetics adapter (build it first with build.sh):
#   run.sh <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#   run.sh values <problem> <size>
#   run.sh --version
# One thread (rule 4.3, see docs/benchmarks/libraries/jenetics.md): the engine runs on the calling
# thread (Bench.java); the serial garbage collector collects on that thread; and -Xbatch
# (-XX:-BackgroundCompilation) makes the calling thread wait while the JIT compiles a method, so
# the JIT's compiler threads never run beside it.
LIB="$HOME/opt/jenetics"
exec "$HOME/opt/jdk/bin/java" -XX:+UseSerialGC -Xbatch \
    -cp "$HOME/bench-targets/jenetics:$LIB/jenetics-9.1.0.jar:$LIB/jenetics.ext-9.1.0.jar" Bench "$@"
