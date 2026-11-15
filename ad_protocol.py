# -*- coding: utf-8 -*-
"""兼容入口：协议实现统一放在 models/common/protocol.py。

保留本文件是为了让 itransformer_improved.py 等旧脚本继续
``from ad_protocol import ...``。
"""
from models.common.protocol import *  # noqa: F401,F403
