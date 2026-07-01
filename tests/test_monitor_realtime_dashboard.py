from eihead.monitoring.web import _render_lightweight_index


def test_lightweight_monitor_only_shows_live_runtime_status_fields() -> None:
    body = _render_lightweight_index(123.0)

    assert "实时运行状态" in body
    assert "系统健康" in body
    assert "视觉实时流" in body
    assert "脖子/云台" in body
    assert "语音会话" in body
    assert "agent 连接" in body
    assert "ASR 识别内容" in body
    assert "TTS 回复内容" in body
    assert "发送给 gateway" in body
    assert "gateway 返回" in body
    assert "每 2.5 秒自动刷新" in body

    removed_legacy_labels = [
        "运行证据",
        "OpenClaw WS",
        "回声门控",
        "本地文本 TTS",
        "最近本地发声",
        "Scheduler",
        "性能优化",
        "ASR 到首文本",
        "ASR 到首音频",
        "首文本到首音频",
        "最大音频间隔",
        "唤醒次数",
        "丢弃片段",
    ]
    for label in removed_legacy_labels:
        assert label not in body
