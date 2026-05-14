# eihead Implementation Plan

目标：先把 honjia 的真实硬件头部运行时从 `eibrain` 中剥离为 `eihead`，
同时建立 `eiprotocol` 的最小统一协议。第一阶段不引入新的安全权限层，
避免在语音、视觉、云台已经打通的链路上叠加额外变量。

## Scope

本轮实施只做四件事：

- 建立 `eiprotocol` 的事件、观测、动作、能力注册、执行结果协议。
- 建立 `/dev-project/eihead`，承接 honjia 的摄像头、麦克风、音箱、云台、Hailo、Web 监控。
- 让 `eibrain` 通过协议调用 `eihead`，不再直接依赖 honjia 硬件实现。
- 把执行结果写回 `eimemory` / `eitraining`，形成可追踪的反馈闭环。

当前拆分状态：`/dev-project/eibrain` 仍是 source repo；
`/dev-project/eiprotocol` 是 exported shared protocol repo；
`/dev-project/eihead` 是 exported head repo。下一批 transport
binding 在 `POST /events` scaffold 之后进入 runtime event routing：HTTP
JSON 每次请求只传递一个 eiprotocol envelope，先完成 action dispatch、
recent event journal/diagnostics 和明确的 `not_processed`/`not_wired` 结果；
SSE/WebSocket/MQTT、二进制音频/视频 chunk 和真正实时 transport streaming
仍是后续工作。

Eye 方向单独明确：正式目标是 `/dev/video0` + `/dev/hailo0` 的 realtime stream detection，也就是从
摄像头/Hailo 连续流产生 `RealtimeVisionObservation`、运行状态和监控检测框。
静态图片检测只作为兼容/测试占位，不作为部署目标或验收主线。
本轮 native eye 的代码边界进一步收敛为 `eihead/eye/gstreamer.py` 负责
`/dev/video0` appsink 实时帧，`eihead/eye/hailo_metadata.py` 负责
`/dev/hailo0` Hailo ROI metadata 检测框和分数解析，`eihead/eye/adapters.py`
只做组合、truthfulness 和 monitor/status 契约适配。

语音边界并未退回 legacy body：`eihead/ear` 与 `eihead/mouth` 已作为
native eihead 边界接管实时链路。`eihead/ear/__init__.py` 与
`eihead/ear/realtime.py`、`eihead/mouth/__init__.py` 与
`eihead/mouth/playback.py` 及 `eihead/monitoring/voice.py` 属于 native
可导出文件，`/api/voice/realtime` 与 `/api/audio/realtime` 是当前的
web 语音状态入口。

当前语音链条已进入 scheduler-backed functional stage：round 生命周期、
scheduler 状态和 interrupt 可见性会进入 Web 监控，但仍处于
functional-not-complete。当前闭环语音诊断是 functional offline/quasi-streaming
diagnostics，不是 hardware-verified real streaming。真实流式 LLM/TTS 尚未接入，
监控只能展示真实来源可见状态，缺失阶段必须显示 `not_wired/unknown`，
不能把未接入的 streaming LLM/TTS 阶段显示成完成。
同一 truthfulness 规则也适用于 event routing handler：非法 envelope
必须走清晰 JSON error / `not_processed` 路径；未接线的 handler 必须返回
明确的 `not_wired` / `not_processed` 状态和原因，不能返回空数据或伪正常结果。

暂不做：

- 不重构安全与权限层。
- 不重构 `eimemory` 内部存储模型。
- 不把大脑策略、人格、LLM 路由迁移到 `eihead`。
- 不把 `eibrain.cognition.realtime` 认知调度所有权迁移到 `eihead`；
  standalone export 只临时携带最小 scheduler 兼容目录，直到
  eibrain/eihead protocol split 完成。
- 不在第一版引入复杂消息总线；先用 HTTP JSON 加状态文件，后续再评估 WebSocket/MQTT。

## Current Extraction Boundary

当前 `eibrain` 中应迁往 `eihead` 的主要边界：

- `apps/body_runtime/*` -> `eihead/apps/head_runtime/*`
- `eibrain/body/*` -> `eihead/eihead/body/*`
- `config/eibrain.honjia*.yaml` -> `eihead/config/eihead.honjia*.yaml`
- `deploy/systemd/eibrain-monitor.service` -> `eihead/deploy/systemd/eihead-monitor.service`
- `deploy/systemd/eibrain-vision-hailo.service` -> `eihead/deploy/systemd/eihead-vision-hailo.service`

`eibrain` 中保留兼容 wrapper，直到 honjia 的 systemd 和监控全部切换完成。

## Phase 0: Baseline Freeze

目标：冻结当前可用链路，后面迁移时知道坏在迁移还是原本就坏。

实施项：

- 在 honxin `/dev-project/eibrain` 确认最新主仓库状态。
- 在 honjia 记录当前服务、端口、设备节点、Web 监控状态。
- 记录语音链路响应指标：唤醒、VAD、ASR、LLM、TTS、播放。
- 记录视觉链路指标：摄像头取帧、Hailo 检测、检测框、云台动作。
- 给当前 eibrain 打 baseline tag 或至少记录 commit。

验收标准：

- honjia `18080` 能看到语音、视觉、云台、服务健康指标。
- 可完成一轮语音对话。
- 摄像头取帧和 Hailo 检测结果可见。
- 云台保持当前可控状态，不因准备工作退化。

## Phase 1: eiprotocol MVP

目标：先统一数据形状，不急着改变传输方式。

当前已落地一个顶层 `eiprotocol` MVP 包，并通过
`docs/eiprotocol-v0.1-mvp.md` 固化 JoyInside 参考文档中进入第一版的
Envelope、CapabilityManifest、AudioTurn、RealtimeVisionObservation、
HeadAction、ExecutionOutcome、UserFeedback 和基础校验范围。第一版只保留
`policy` 元数据，不引入安全权限层运行依赖。

`/dev-project/eiprotocol` 是当前 exported shared protocol repo。后续
`eihead`, `eibrain`, `eimemory`, `eiskills`, `eidocs` 都应依赖这个共享协议，
而不是继续扩展各自的本地镜像。

核心协议：

- `Envelope`: `specVersion`, `id`, `type`, `name`, `source`, `target`,
  `traceId`, `time`, `sequence`, `requestId`, `sessionId`, `roundId`,
  `content`, `policy`
- `CapabilityManifest`: 设备、模型、后端、健康、限制、版本。
- `DeviceStatus`: 摄像头、麦克风、音箱、云台、Hailo、I2C、服务状态。
- `AudioTurn`: 唤醒词、ASR 文本、置信度、音频时长、分段时间。
- `RealtimeVisionObservation`: realtime stream frame 摘要、检测框、类别、分数、延迟、帧时间。
- `HeadAction`: `speak`, `stop_speech`, `move_head`, `set_attention`, `capture_frame`
- `ExecutionOutcome`: 动作、执行者、成功/失败、延迟、错误、观测结果。
- `UserFeedback`: 用户显式纠正、满意/不满意、偏好信号。

验收标准：

- 协议包有单元测试和 JSON round-trip 测试。
- 每个消息都有 `traceId`，能串起一次语音或视觉交互。
- 现有 eibrain 可通过兼容 adapter 使用这些模型，不要求一次性替换所有字典。

## Phase 2: eihead Repository Scaffold

目标：建立新仓库和运行框架，但先不拆断旧链路。

目录建议：

```text
eihead
├── apps/head_runtime
├── eihead/body
├── eihead/protocol
├── eihead/monitoring
├── eihead/services
├── config
├── deploy/systemd
├── scripts
└── tests
```

实施项：

- 在 honxin `/dev-project/eihead` 初始化仓库。
- 迁入 `apps/body_runtime`，重命名为 `apps/head_runtime`。
- 迁入 `eibrain/body` 的硬件驱动、器官状态、语音、视觉、云台代码。
- 保留原 eibrain 路径 wrapper，转发到 `eihead` 或通过 HTTP client 调用。
- 建立 `eihead status`、`eihead verify-hardware`、`eihead serve` 三个入口。

验收标准：

- 在开发机和 honxin 上能跑单元测试。
- 在 honjia 上能独立启动 `eihead` 的 monitor/runtime，不依赖 eibrain 内部 import。
- 旧 `eibrain-monitor.service` 仍可用，避免一次迁移把现场链路打断。

## Phase 3: Capability Registration

目标：`eihead` 启动后主动告诉 `eibrain` 自己有哪些能力。

传输方式：

- 第一版：`POST /api/head/capabilities` 到 eibrain。
- 同时在 honjia 写本地状态文件，供 Web 监控和故障排查读取。

能力内容：

- 设备：`/dev/video0`, `/dev/hailo0`, `/dev/i2c-1`, 麦克风输入、音箱输出。
- 模型：ASR、TTS、Hailo HEF、视觉后处理、embedding。
- 健康：online/offline/degraded、错误、最近成功时间、延迟。
- 限制：云台水平角度范围、速率限制、可用动作。

验收标准：

- eibrain Web/API 能看到 honjia 的 `CapabilityManifest`。
- honjia Web `18080` 能显示同一份能力和健康数据。
- 替换设备或模型配置时，不需要改 eibrain 认知代码。

## Phase 4: Runtime Protocol Bridge

目标：让 eibrain 与 eihead 通过协议交互。

下一批 transport binding 先只接受 event transport MVP：
HTTP JSON `POST /events`，请求体为 `eiprotocol` envelope。实时 transport
streaming、WebSocket/SSE/MQTT、二进制音频/视频 chunk、replay/resume 和
backpressure 不作为本批验收要求。本批只路由每个 HTTP request 中的一
个 JSON envelope，并把 route/dispatch/diagnostic 结果留在 runtime
recent event journal 中。

eihead -> eibrain：

- `AudioTurn`: 语音识别结果进入对话链。
- `RealtimeVisionObservation`: realtime stream detection 的检测框、分数、画面摘要进入视觉链。
- `DeviceStatus`: 设备状态进入监控和健康评估。
- `ExecutionOutcome`: 语音播放、云台动作、视觉处理结果写回。

eibrain -> eihead：

- `SpeakAction`: TTS/播放。
- `MoveHeadAction`: 水平云台动作。
- `StopSpeechAction`: 打断播放。
- `AttentionIntent`: 注视、跟随、休眠、唤醒等状态。
- `CaptureFrameAction`: 诊断取帧。

runtime routing:

- action request event 先进入 `handle_action` bridge，再分发到已有的
  mouth、neck、attention、diagnostic capture handler。
- observation、outcome、feedback event 写入 recent event journal/diagnostics，
  供 Web monitor 和 runtime API 查看。
- invalid envelope 返回 JSON error，并产生明确的 `not_processed` 结果；
  unknown/unwired event 返回 `not_wired` 或 `not_processed` 和原因。

验收标准：

- `eihead` 与 `eibrain` 至少能通过 HTTP JSON `POST /events` 交换
  capability、observation、action、outcome envelope。
- action request event 能通过 `handle_action` bridge 到达当前 action
  执行路径。
- observation、outcome、feedback 能被记录为 recent event journal/diagnostics。
- invalid envelope 走清晰 JSON error / `not_processed` 路径。
- monitor/API 能 inspect recent events。
- 语音对话仍能完成，且 trace 能看到 ASR -> LLM -> TTS -> 播放。
- 视觉检测框和分数仍能显示在 honjia Web 监控。
- 云台动作可以从 eibrain 通过 action 下发到 eihead。
- 缺失或尚未接线的 event handler 返回明确 `not_wired` 或
  `not_processed` 状态，包含原因，不返回 blank/fake-normal payload。

## Phase 5: Web Monitoring Split

目标：Web 监控从“看似正常但没数据”变成真实数据面板。

面板分层：

- `Runtime`: 服务心跳、刷新频率、平均延迟、错误。
- `Ear`: 输入设备、VAD、ASR、最近文本、分段耗时。
- `Mouth`: TTS provider、合成耗时、播放状态、错误。
- `Eye`: realtime stream detection 状态、`/dev/video0` 摄像头、`/dev/hailo0` Hailo、GStreamer pipeline、parser error count、FPS、检测框、分数、最近帧。
- `Neck`: 当前水平角、目标角、动作频率、抖动抑制状态。
- `Protocol`: 最近 `traceId`、capability、observation、action、outcome。
- `Voice scheduler`: round、scheduler、interrupt 状态；真实流式 LLM/TTS
  未接入前必须显示 `not_wired/unknown/degraded`。

验收标准：

- 每个面板字段都有真实来源；没有来源的字段显示 `unknown` 或 `not wired`，不能假正常。
- 最近一轮对话和最近一轮 realtime eye stream detection 可通过 `traceId` 串起来。
- 刷新后布局和数据都保持稳定。

## Phase 6: Memory And Training Feedback

目标：把“做了什么、结果如何、下次怎么改”沉淀下来。

写入 eimemory：

- 重要对话回合。
- 用户身份/偏好/纠正。
- 视觉识别到的稳定人物或场景事件。
- 执行动作结果和失败原因。

写入 eitraining：

- 可复盘的完整 trace。
- ASR 误识别样本。
- LLM 低质量回复样本。
- 云台追踪抖动/丢失目标样本。
- 用户明确反馈的好/坏案例。

验收标准：

- 最近一轮对话可在 eimemory 查到写入记录。
- 最近一次动作 outcome 可在 eimemory 或 eitraining 查到。
- 反馈不直接污染长期人格记忆，需经过 `MultimodalMemoryPolicy` 分类。

## Phase 7: Deployment Cutover

目标：把 honjia 的 systemd 从 `eibrain-*` 平滑切到 `eihead-*`。

步骤：

1. honxin `/dev-project/eihead` 确认为主仓库。
2. 同步 `eihead` 到 honjia 标准路径，例如 `/opt/eihead/current`。
3. 新增并启动 `eihead-monitor.service`、`eihead-runtime.service`、`eihead-vision-hailo.service`。
4. 保留旧 `eibrain-monitor.service` wrapper 一段时间，指向新服务或反向代理。
5. Web 端确认 `18080` 指向 `eihead` monitor。
6. eibrain 配置改为远程调用 `eihead`，不再加载 honjia 本地硬件 driver。
7. 连续测试语音、realtime eye stream detection、云台、Web、记忆写回。
8. 再移除旧服务或标记 deprecated。

验收标准：

- honjia 重启后 `eihead` 自动常驻。
- honxin eibrain 重启后能重新发现 honjia capability。
- GitHub、honxin `/dev-project`、honjia 部署路径三方版本可追踪。

## Recommended Work Order

推荐先做这三件，因为风险最低且后续全部依赖它们：

1. `eiprotocol` MVP：消息模型、JSON schema、`traceId`。
2. `eihead` scaffold：复制而不是大改，先让独立仓库能跑。
3. capability + monitor：先让 honjia 的真实设备能力和 realtime eye stream detection 状态稳定显示，
   且硬件/插件/metadata parser 缺失时必须显示 `not_wired`、`unknown` 或 degraded，不能假正常。

完成这三步后，再迁移语音、realtime eye stream detection、云台 action bridge。这样每一步都有回滚点，
不会把已经打通的现场链路重新搅乱。
