#!/usr/bin/env bash
# Runs the jMetal adapter (build it first with build.sh):
#   run.sh <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#   run.sh --version
# The serial garbage collector keeps the JVM's own work on one thread; jMetal evaluates
# sequentially (see Bench.java). Only the JIT compiler threads run in the background.
exec "$HOME/opt/jdk/bin/java" -XX:+UseSerialGC -cp "$HOME/bench-targets/jmetal:$HOME/opt/jmetal/*" Bench "$@"
