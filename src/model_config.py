"""One model/cache contract shared by inference and optional RunPod prefetch."""

import os

MODEL_ID_DEFAULT = "jingheya/lotus-depth-d-v2-0-disparity"
MODEL_REVISION_DEFAULT = "10b3e82cbb999274f69982d518d5abfdbe5f7fe8"
MODEL_ID = os.environ.get("MODEL_ID", MODEL_ID_DEFAULT)
MODEL_REVISION = os.environ.get(
    "MODEL_REVISION", MODEL_REVISION_DEFAULT if MODEL_ID == MODEL_ID_DEFAULT else "main"
)
