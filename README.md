# BiliSpeech2Text

## 简介
BiliSpeech2Text 把 B 站视频里的语音转成文本：解析视频链接（支持 b23.tv 短链）→ 下载 DASH 音频轨 → 转码切段 → 调用语音识别模型 → 输出文字稿。

语音识别后端为**小米 MiMo ASR**（`mimo-v2.5-asr`），走 OpenAI 兼容的 `/v1/chat/completions` 协议，音频以 data URL（Base64）形式随请求发送，无需本地 GPU。

## 功能
- 支持 av 号、BV 号和 b23.tv 短链
- 只下载音频轨（比下载整段视频快得多），并缓存复用
- 自动转码为单声道 16 kHz mp3（MiMo 只接受 mp3 / wav）
- 按 70 秒切段，切点尽量落在静音处；相邻两段保留 2 秒重叠，用文本比对把接缝合起来，避免吞词或重复
- 逐段调用 MiMo ASR，失败自动重试；检测到复读会重试并做去重
- 识别完成后调用大模型（默认 `mimo-v2.6-flash`）把文字稿整理成便于阅读的段落，只分段不改写
- 同时输出纯文本 `.txt`、分段整理稿 `.md` 和带时间戳分段的 `.json`
- 视频属于合集时默认只处理当前一集，`--collection` 可交互选择分集

## 使用方法
1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
2. 准备 ffmpeg：系统 PATH 里有 `ffmpeg` 即可；否则安装 `imageio-ffmpeg`，或在 `config.json` 的 `audio.ffmpeg` 写明可执行文件路径。
3. 配置密钥：把 `.env.example` 复制成 `.env`，填入真实值（`.env` 已在 `.gitignore` 里，不会被提交）：
   ```ini
   MIMO_API_KEY=sk-...
   # 可选：覆盖 config.json 中的同名配置
   # MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
   # MIMO_MODEL=mimo-v2.5-asr
   # MIMO_LANGUAGE=zh
   # 可选：B 站登录态
   BILI_SESSDATA=
   BILI_JCT=
   BILI_DEDEUSERID=
   ```
   - 不写 `.env` 时也可以直接用 shell 环境变量，已经存在的环境变量优先级更高。
   - 模型、服务地址、语种等非密钥配置仍在 `config.json`：`asr.base_url` / `asr.model` / `asr.language`（`auto` / `zh` / `en`）。
   - 官方直连地址为 `https://api.xiaomimimo.com/v1`；Token Plan 套餐用 `https://token-plan-cn.xiaomimimo.com/v1`。
4. 运行
   ```bash
   python main.py https://b23.tv/IhbEdSu
   ```

### 常用参数
```bash
python main.py <链接> \
  --language zh \          # 识别语种
  --chunk-seconds 70 \     # 每段音频长度
  --collection \           # 视频属于合集时交互选择分集
  --model mimo-v2.5-asr \  # 覆盖识别模型
  --no-organize \          # 不做分段整理
  --organize-only \        # 跳过识别，只整理已有 txt
  --organize-model mimo-v2.6-flash  # 覆盖整理用的模型
```

## 分段整理
识别完成后会再调一次大模型，把一整段没有换行的文字稿整理成 200 字左右的段落，写到 `outputs/{bvid}.md`。

- **只改结构**：提示词禁止改写、增删、总结；每次返回都会做校验——去掉空白和标点后必须和原文逐字一致，不一致就重试，两次都不过就退回本地按句分段（并在控制台提示）
- **分块处理**：`organize.max_input_chars`（默认 6000）超过就按句子边界切块，每块单独整理后拼回
- **省 token**：默认关掉思考模式（`organize.thinking=false`），纯排版任务更快更省
- 想换更强的模型：`organize.model` 改成 `mimo-v2.6-pro`

## 输出说明
以 `bilibili_video/{视频标题}/` 为根目录：
- `conv/{bvid}.m4a`：下载的原始音频
- `conv/{bvid}.mp3`：转码后的识别输入
- `conv/{bvid}_chunks/*.mp3`：切段音频
- `outputs/{bvid}.txt`：最终文字稿（已按重叠区合并，无换行）
- `outputs/{bvid}.md`：大模型整理过分段的版本，便于阅读
- `outputs/{bvid}.json`：每段原始识别结果与起止时间，便于核对

## 注意事项
- MiMo ASR 单次请求的音频 Base64 后不得超过 10 MB；默认 70 秒一段（约 0.4 MB），既远低于上限，也能减少长音频偶发的“复读”退化。
- 已存在的音频、切段、文字稿和整理稿都会自动复用；想重新识别就删掉对应的 `outputs/{bvid}.txt`，想重新整理就删掉 `outputs/{bvid}.md`。
- 密钥只放在 `.env` 里（已被忽略）；`config.json` 不再保存 API Key 和 cookies，可以安全提交。
