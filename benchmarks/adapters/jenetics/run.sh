#!/usr/bin/env bash
# Runs the Jenetics adapter (build it first with build.sh):
#   run.sh <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#   run.sh --version
# The serial garbage collector keeps the JVM's own work on one thread; the engine runs on the
# calling thread (see Bench.java). Only the JIT compiler threads run in the background.
LIB="$HOME/opt/jenetics"
exec "$HOME/opt/jdk/bin/java" -XX:+UseSerialGC -cp "$HOME/bench-targets/jenetics:$LIB/jenetics-9.1.0.jar:$LIB/jenetics.ext-9.1.0.jar" Bench "$@"
