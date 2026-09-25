#!/usr/bin/env bash
# Builds the Jenetics adapter: fetches the pinned JDK (Eclipse Temurin 25) into ~/opt/jdk and the
# pinned Jenetics jars from Maven Central into ~/opt/jenetics, and compiles Bench.java into
# ~/bench-targets/jenetics. Jenetics 9 needs Java 25.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JENETICS_VERSION=9.1.0
JDK_URL="https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.4.1%2B1/OpenJDK25U-jdk_x64_linux_hotspot_25.0.4.1_1.tar.gz"
JDK_SHA256=dbb698396d478e7fa2b1e50f4103324b2a99b90569ee27c33f2261f9215cf41e
JDK="$HOME/opt/jdk"
LIB="$HOME/opt/jenetics"
OUT="$HOME/bench-targets/jenetics"

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
fetch io/jenetics jenetics "$JENETICS_VERSION"
fetch io/jenetics jenetics.ext "$JENETICS_VERSION"

rm -rf "$OUT"
mkdir -p "$OUT"
"$JDK/bin/javac" -nowarn -d "$OUT" -cp "$LIB/jenetics-$JENETICS_VERSION.jar:$LIB/jenetics.ext-$JENETICS_VERSION.jar" "$HERE/Bench.java"
