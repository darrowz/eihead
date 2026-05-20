from __future__ import annotations

from eihead.eivoice_runtime import FakeWebSocketTransport, InMemoryVoiceStreamTransport


class ManualClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_inmemory_transport_buffers_outbound_and_inbound_events() -> None:
    transport = InMemoryVoiceStreamTransport(clock=ManualClock(100.0), outbound_capacity=2, inbound_capacity=2)
    outbound = {"contentType": "TEXT", "content": {"eventType": "TEXT", "text": "hello"}}
    inbound = {"contentType": "ASR", "content": {"eventType": "ASR", "text": "你好鸿嘉"}}

    transport.mark_connected()

    assert transport.send_event(outbound) is True
    assert transport.pop_outbound_event() == outbound
    assert transport.pop_outbound_event() is None

    transport.push_inbound_event(inbound)

    assert transport.receive_event() == inbound
    assert transport.receive_event() is None

    status = transport.status()
    assert status["connection"]["state"] == "connected"
    assert status["queues"]["outbound_queue"]["depth"] == 0
    assert status["queues"]["inbound_queue"]["depth"] == 0
    assert status["queues"]["ws_send_queue"]["capacity"] == 2
    assert status["queues"]["opus_decode_queue"]["capacity"] == 2


def test_transport_heartbeat_ping_pong_updates_timing_status() -> None:
    clock = ManualClock(10.0)
    transport = InMemoryVoiceStreamTransport(
        clock=clock,
        heartbeat_interval_s=5.0,
        pong_timeout_s=3.0,
    )
    transport.mark_connected()

    assert transport.heartbeat_due() is False

    clock.advance(5.0)
    ping_event = transport.send_ping(uid="user-1", mid="mid-1")

    assert ping_event["content"]["eventType"] == "PING"
    assert ping_event["uid"] == "user-1"
    assert transport.status()["heartbeat"]["awaiting_pong"] is True
    assert transport.status()["heartbeat"]["last_ping_at"] == 15.0

    clock.advance(1.25)
    transport.record_pong({"contentType": "PONG", "content": {"eventType": "PONG"}})

    status = transport.status()
    assert status["heartbeat"]["awaiting_pong"] is False
    assert status["heartbeat"]["last_pong_at"] == 16.25
    assert status["heartbeat"]["latency_ms"] == 1250


def test_transport_schedules_reconnect_and_records_timeout_error_when_pong_is_missing() -> None:
    clock = ManualClock(20.0)
    transport = InMemoryVoiceStreamTransport(
        clock=clock,
        heartbeat_interval_s=5.0,
        pong_timeout_s=2.0,
        reconnect_base_delay_s=1.0,
        reconnect_max_delay_s=4.0,
    )
    transport.mark_connected()

    clock.advance(5.0)
    transport.send_ping()
    clock.advance(2.1)

    assert transport.check_heartbeat() == "heartbeat_timeout"

    status = transport.status()
    assert status["connection"]["state"] == "reconnect_wait"
    assert status["reconnect"]["attempt"] == 1
    assert status["reconnect"]["backoff_s"] == 1.0
    assert status["reconnect"]["ready"] is False
    assert status["last_error"]["kind"] == "TimeoutError"
    assert status["last_error"]["context"] == "heartbeat"

    clock.advance(1.0)

    assert transport.ready_to_reconnect() is True

    transport.begin_reconnect()
    transport.mark_connected()

    status = transport.status()
    assert status["connection"]["state"] == "connected"
    assert status["reconnect"]["attempt"] == 0
    assert status["reconnect"]["next_retry_at"] is None
    assert status["last_error"] is None
    assert status["recent_errors"][0]["context"] == "heartbeat"


def test_transport_record_error_and_backoff_cap_are_visible_in_status() -> None:
    clock = ManualClock(50.0)
    transport = InMemoryVoiceStreamTransport(
        clock=clock,
        reconnect_base_delay_s=1.0,
        reconnect_max_delay_s=4.0,
    )

    transport.record_error(RuntimeError("first failure"), context="send")
    transport.schedule_reconnect("first")
    clock.advance(1.0)
    transport.begin_reconnect()
    transport.schedule_reconnect("second")
    clock.advance(2.0)
    transport.begin_reconnect()
    transport.schedule_reconnect("third")

    status = transport.status()
    assert status["last_error"]["message"] == "first failure"
    assert status["last_error"]["context"] == "send"
    assert status["reconnect"]["attempt"] == 3
    assert status["reconnect"]["backoff_s"] == 4.0
    assert status["reconnect"]["reason"] == "third"
    assert status["recent_errors"][0]["message"] == "first failure"


def test_fake_websocket_transport_exposes_runner_friendly_aliases() -> None:
    transport = FakeWebSocketTransport(clock=ManualClock(5.0))
    outbound = {"contentType": "TEXT", "content": {"eventType": "TEXT", "text": "runner"}}
    inbound = {"contentType": "COMPLETE", "content": {"eventType": "COMPLETE"}}

    transport.open()
    transport.send_event(outbound)

    assert transport.recv_from_client() == outbound

    transport.deliver_from_server(inbound)

    assert transport.receive_event() == inbound
    assert transport.status()["transport"] == "fake_websocket"


def test_transport_status_is_consumable_by_runtime_monitor_panel() -> None:
    from eihead.monitoring.eivoice_runtime import build_eivoice_runtime_panel

    transport = InMemoryVoiceStreamTransport(clock=ManualClock(30.0), outbound_capacity=3, inbound_capacity=4)
    transport.mark_connected()
    transport.send_event({"contentType": "TEXT", "content": {"eventType": "TEXT", "text": "hello"}})
    transport.push_inbound_event({"contentType": "ASR", "content": {"eventType": "ASR", "text": "nihao"}})

    panel = build_eivoice_runtime_panel(transport.status())

    assert panel["state"] == "connected"
    assert panel["conversationState"] == "connected"
    assert panel["queues"]["ws_send_queue"]["capacity"] == 3
    assert panel["queues"]["opus_decode_queue"]["capacity"] == 4
