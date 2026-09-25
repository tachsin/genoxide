#!/usr/bin/env bash
# Builds the jMetal adapter: fetches the pinned JDK (Eclipse Temurin 25) into ~/opt/jdk and the
# pinned jMetal jars and their runtime dependencies from Maven Central into ~/opt/jmetal, and
# compiles Bench.java into ~/bench-targets/jmetal. (Test and charting dependencies of jMetal, such
# as JUnit, XChart and Weka, aren't needed.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JMETAL_VERSION=7.5
JDK_URL="https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.4.1%2B1/OpenJDK25U-jdk_x64_linux_hotspot_25.0.4.1_1.tar.gz"
JDK_SHA256=dbb698396d478e7fa2b1e50f4103324b2a99b90569ee27c33f2261f9215cf41e
JDK="$HOME/opt/jdk"
LIB="$HOME/opt/jmetal"
OUT="$HOME/bench-targets/jmetal"

if [ ! -x "$JDK/bin/javac" ]; then
    mkdir -p "$HOME/opt"
    curl -fsSL -o "$HOME/opt/jdk.tar.gz" "$JDK_URL"
    echo "$JDK_SHA256  $HOME/opt/jdk.tar.gz" | sha256sum -c --quiet
    mkdir -p "$JDK"
    tar -xzf "$HOME/opt/jdk.tar.gz" -C "$JDK" --strip-components=1
    rm "$HOME/opt/jdk.tar.gz"
fi

mkdir -p "$LIB"
fetch() { # group path, artifact, version
    local jar="$2-$3.jar"
    [ -s "$LIB/$jar" ] || curl -fsSL -o "$LIB/$jar" "https://repo1.maven.org/maven2/$1/$2/$3/$jar"
}
fetch org/uma/jmetal jmetal-core "$JMETAL_VERSION"
fetch org/uma/jmetal jmetal-algorithm "$JMETAL_VERSION"
fetch org/uma/jmetal jmetal-component "$JMETAL_VERSION"
fetch org/uma/jmetal jmetal-problem "$JMETAL_VERSION"
# the versions of jMetal 7.5's parent pom
fetch org/apache/commons commons-lang3 3.18.0
fetch org/apache/commons commons-math3 3.6.1
fetch org/apache/commons commons-collections4 4.4
fetch commons-io commons-io 2.16.1
fetch com/github/mbuzdalov non-dominated-sorting-implementations 0.2.1

rm -rf "$OUT"
mkdir -p "$OUT"
"$JDK/bin/javac" -nowarn -d "$OUT" -cp "$LIB/*" "$HERE/Bench.java"
