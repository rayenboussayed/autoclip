#!/bin/sh
# Install autoclip: no Python, no venv, no ffmpeg needed.
#   curl -fsSL https://github.com/rayenboussayed/autoclip/releases/latest/download/install.sh | sh
set -eu

REPO="rayenboussayed/autoclip"
VERSION="${AUTOCLIP_VERSION:-latest}"
HOME_DIR="${AUTOCLIP_HOME:-$HOME/.autoclip}"
BIN_DIR="${AUTOCLIP_BIN_DIR:-$HOME/.local/bin}"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }
need curl; need tar; need uname; need awk

if [ -n "${AUTOCLIP_TARGET:-}" ]; then
  target="$AUTOCLIP_TARGET"
else
  case "$(uname -s)" in
    Linux) os=linux ;; Darwin) os=macos ;;
    *) die "unsupported OS: $(uname -s) (on Windows use install.ps1)" ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64) arch=x64 ;; aarch64|arm64) arch=arm64 ;;
    *) die "unsupported CPU: $(uname -m)" ;;
  esac
  target="$os-$arch"
fi
asset="autoclip-$target.tar.gz"

if [ -n "${AUTOCLIP_BASE_URL:-}" ]; then base="$AUTOCLIP_BASE_URL"
elif [ "$VERSION" = latest ]; then base="https://github.com/$REPO/releases/latest/download"
else base="https://github.com/$REPO/releases/download/v${VERSION#v}"; fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM
say "Downloading $asset ..."
curl -fsSL --retry 3 -o "$tmp/$asset" "$base/$asset" || die "download failed: $base/$asset"
curl -fsSL --retry 3 -o "$tmp/SHA256SUMS" "$base/SHA256SUMS" || die "could not download SHA256SUMS"

expected="$(awk -v f="$asset" '$2==f || $2=="*"f {print $1}' "$tmp/SHA256SUMS")"
[ -n "$expected" ] || die "no checksum listed for $asset"
if command -v sha256sum >/dev/null 2>&1; then actual="$(sha256sum "$tmp/$asset" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then actual="$(shasum -a 256 "$tmp/$asset" | awk '{print $1}')"
else die "need sha256sum or shasum to verify the download"; fi
[ "$expected" = "$actual" ] || die "checksum mismatch for $asset (expected $expected, got $actual)"

mkdir -p "$HOME_DIR" "$BIN_DIR"
rm -rf "$HOME_DIR/app.new" && mkdir "$HOME_DIR/app.new"
tar -xzf "$tmp/$asset" -C "$HOME_DIR/app.new"
rm -rf "$HOME_DIR/app" && mv "$HOME_DIR/app.new/autoclip" "$HOME_DIR/app" && rmdir "$HOME_DIR/app.new"
ln -sf "$HOME_DIR/app/autoclip" "$BIN_DIR/autoclip"
if [ "$(uname -s)" = Darwin ]; then xattr -dr com.apple.quarantine "$HOME_DIR/app" 2>/dev/null || true; fi

say "Installed: $BIN_DIR/autoclip"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "Add to PATH:  export PATH=\"$BIN_DIR:\$PATH\"" ;; esac
"$BIN_DIR/autoclip" --version || true