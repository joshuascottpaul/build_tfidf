#!/bin/bash
set -e
VERSION=${1:-"v0.1.0"}
PLATFORMS=("linux-amd64" "linux-arm64" "darwin-amd64" "darwin-arm64")
echo "Packaging build_tfidf $VERSION"
mkdir -p dist
for platform in "${PLATFORMS[@]}"; do
    platform_dir="dist/build_tfidf-$platform"
    mkdir -p "$platform_dir"
    cp -r build_tfidf "$platform_dir/"
    cp requirements*.txt "$platform_dir/" 2>/dev/null || true
    cp README.md LICENSE "$platform_dir/" 2>/dev/null || true
    cat > "$platform_dir/install.sh" << 'INSTALL'
#!/bin/bash
set -e
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$INSTALL_DIR"
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi
pip3 install -r requirements.txt --user
pip3 install -e . --user
echo "✓ Installed build_tfidf"
INSTALL
    chmod +x "$platform_dir/install.sh"
    cd dist
    tar -czf "build_tfidf-$VERSION-$platform.tar.gz" "build_tfidf-$platform"
    rm -rf "build_tfidf-$platform"
    cd ..
done
echo "✓ All packages created in dist/"
