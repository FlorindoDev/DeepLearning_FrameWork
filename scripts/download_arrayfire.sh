#!/usr/bin/env bash
set -euo pipefail

version=3.10.0
url="https://arrayfire.gateway.scarf.sh/linux/$version/ArrayFire.sh"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
download_dir="$script_dir/../deps/downloads"
installer="$download_dir/ArrayFire-v$version-Linux-x64.sh"

if [[ "${1:-}" == "--help" ]]; then
    echo "Scarica l'installer ufficiale ArrayFire Linux x64 in deps/downloads."
    echo "L'installazione va avviata manualmente dopo il download."
    exit 0
fi

mkdir -p -- "$download_dir"
if [[ ! -f "$installer" ]]; then
    command -v curl >/dev/null || { echo "Errore: installa curl per scaricare ArrayFire." >&2; exit 1; }
    echo "Download ArrayFire $version per Linux x64..."
    curl --fail --location --retry 3 --output "$installer.part" "$url"
    mv -- "$installer.part" "$installer"
fi

echo "Installer disponibile: $installer"
echo "Per installare ArrayFire in /opt/arrayfire, esegui:"
printf 'sudo bash "%s" --include-subdir --prefix=/opt\n' "$installer"
echo 'Per una cartella diversa, specifica AF_DIR quando esegui make build.'
