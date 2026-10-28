#!/usr/bin/env bash
# ==============================================================================
# Helper script to sync datasets or checkpoints between AWS EC2 and Amazon S3
# Usage:
#   bash cloud/sync_s3.sh download s3://my-bucket/roman-data/ /data/roman/
#   bash cloud/sync_s3.sh upload /data/checkpoints/ s3://my-bucket/roman-checkpoints/
# ==============================================================================

set -e

ACTION="${1:-}"
SOURCE="${2:-}"
DEST="${3:-}"

if [ -z "$ACTION" ] || [ -z "$SOURCE" ] || [ -z "$DEST" ]; then
    echo "Usage: $0 [download|upload] <source> <destination>"
    echo ""
    echo "Examples:"
    echo "  Download data from S3:"
    echo "    $0 download s3://my-bucket/roman/synthetic/ /data/roman/"
    echo ""
    echo "  Upload trained checkpoints to S3:"
    echo "    $0 upload ./checkpoints/ s3://my-bucket/sam2-checkpoints/"
    exit 1
fi

if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI ('aws') is not installed. Install with: pip install awscli"
    exit 1
fi

echo "=============================================================================="
echo " AWS S3 Sync: $ACTION"
echo " Source:      $SOURCE"
echo " Destination: $DEST"
echo "=============================================================================="

if [ "$ACTION" == "download" ]; then
    mkdir -p "$DEST"
    aws s3 sync "$SOURCE" "$DEST" --exclude "*.DS_Store"
    echo "[Sync Complete] Downloaded to $DEST"
elif [ "$ACTION" == "upload" ]; then
    aws s3 sync "$SOURCE" "$DEST" --exclude "*.DS_Store"
    echo "[Sync Complete] Uploaded to $DEST"
else
    echo "Error: Unknown action '$ACTION'. Use 'download' or 'upload'."
    exit 1
fi
