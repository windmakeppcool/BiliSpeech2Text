"""BiliSpeech2Text

把 B 站视频里的语音转成文本。

语音识别后端：小米 MiMo ASR（模型 ``mimo-v2.5-asr``），走 OpenAI 兼容的
``/v1/chat/completions`` 协议；音频以 data URL + Base64 传给模型。
"""

from __future__ import annotations

import argparse
import base64
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import requests

BASE_DIR = 'bilibili_video'
BV_PATTERN = re.compile(r'BV[0-9A-Za-z]{10}')
AV_PATTERN = re.compile(r'av(\d+)', re.I)
# Base64 之后的字符串上限 10 MB（官方限制），按裸音频 ~7 MB 控制更稳
MAX_AUDIO_BASE64_BYTES = 10 * 1024 * 1024

DEFAULT_ASR = {
    'provider': 'mimo',
    'base_url': 'https://token-plan-cn.xiaomimimo.com/v1',
    'model': 'mimo-v2.5-asr',
    'language': 'zh',          # auto / zh / en
    'timeout': 180,
    'retries': 3,
}
DEFAULT_AUDIO = {
    'chunk_seconds': 70,       # 每段送识别的音频长度：太长容易触发复读
    'overlap_seconds': 2.0,    # 相邻两段的重叠时长，用文本比对把接缝合起来
    'sample_rate': 16000,
    'bitrate': '48k',
    'ffmpeg': '',              # 留空则自动查找
    'silence_db': -25,         # 切点搜索用的静音阈值（有 BGM 时 -35dB 太严）
    'silence_dur': 0.3,        # 静音最短持续时间（秒）
}
DEFAULT_ORGANIZE = {
    'enabled': True,
    'model': 'mimo-v2.6-flash',
    'base_url': '',            # 留空复用 asr.base_url
    'api_key': '',             # 留空复用 MIMO_API_KEY
    'temperature': 0.3,
    'max_input_chars': 6000,   # 单次送整理的字符数，超了就分块
    'retries': 2,
    'thinking': False,         # 纯排版任务，关掉思考更快更省
}


# --------------------------------------------------------------------------- #
# 复读检测与去重
# --------------------------------------------------------------------------- #
def dedupe_repeats(text: str, min_len: int = 50, max_rounds: int = 10) -> str:
    """删掉 ASR 退化产生的“复读”：同一段话被连续重复输出。

    只处理长度 ≥ ``min_len`` 的重复片段，正常口语里几乎不会出现这么长的
    逐字重复，所以误伤概率很低。
    """
    if len(text) < min_len * 2:
        return text
    for _ in range(max_rounds):
        found = None
        for i in range(len(text) - min_len):
            probe = text[i:i + min_len]
            j = text.find(probe, i + min_len)
            if j < 0:
                continue
            n = min_len
            while j + n < len(text) and i + n < j and text[i + n] == text[j + n]:
                n += 1
            found = (j, j + n)
            break
        if not found:
            break
        text = text[:found[0]] + text[found[1]:]
    return text


def is_degenerate(text: str, min_len: int = 50) -> bool:
    """判断一段识别结果是否明显退化（复读）。"""
    n = len(text)
    if n < 300:
        return False
    unique_ratio = len({text[i:i + 20] for i in range(n - 20)}) / max(1, n - 20)
    if unique_ratio < 0.9:
        return True
    return text.find(text[0:min_len], min_len) >= 0


# --------------------------------------------------------------------------- #
# 整理分段（调用大模型，只改结构不改内容）
# --------------------------------------------------------------------------- #
def content_key(text: str) -> str:
    """去掉空白和标点后的文本，用来校验整理前后内容是否一致。"""
    return ''.join(ch for ch in text
                   if not ch.isspace() and not unicodedata.category(ch).startswith('P'))


def split_blocks(text: str, limit: int) -> List[str]:
    """按句子边界把长文本切成不超过 limit 字符的若干块。"""
    if len(text) <= limit:
        return [text]
    blocks: List[str] = []
    buf = ''
    for sentence in re.split(r'(?<=[。！？!?])', text):
        if not sentence:
            continue
        if buf and len(buf) + len(sentence) > limit:
            blocks.append(buf)
            buf = ''
        buf += sentence
    if buf:
        blocks.append(buf)
    return blocks


def local_paragraphs(text: str, target: int = 260) -> str:
    """不调模型时的兜底分段：按句子攒够 target 字左右就换段。"""
    paragraphs: List[str] = []
    buf = ''
    for sentence in re.split(r'(?<=[。！？!?])', text):
        if not sentence:
            continue
        buf += sentence
        if len(buf) >= target and sentence[-1] in '。！？!?':
            paragraphs.append(buf.strip())
            buf = ''
    if buf.strip():
        paragraphs.append(buf.strip())
    return '\n\n'.join(paragraphs)


def _normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """去掉标点和空白，同时记录每个字符在原文里的下标。"""
    chars: List[str] = []
    index: List[int] = []
    for i, ch in enumerate(text):
        if ch.isspace() or unicodedata.category(ch).startswith('P'):
            continue
        chars.append(ch)
        index.append(i)
    return ''.join(chars), index


def merge_overlap(prev: str, new: str, window: int = 60, min_match: int = 6) -> str:
    """把带重叠的两段识别结果拼起来：在重叠区里找最长公共片段，从这里接上。

    找不到足够长的公共片段时，退化成直接拼接。
    """
    if not prev:
        return new
    if not new:
        return prev
    tail = prev[-window:]
    head = new[:window]
    tail_norm, tail_map = _normalize_with_map(tail)
    head_norm, head_map = _normalize_with_map(head)
    matcher = difflib.SequenceMatcher(None, tail_norm, head_norm, autojunk=False)
    a, b, size = matcher.find_longest_match(0, len(tail_norm), 0, len(head_norm))
    if size < min_match:
        return prev + new
    cut_prev = len(prev) - len(tail) + tail_map[a + size - 1] + 1
    cut_new = head_map[b + size - 1] + 1
    return prev[:cut_prev] + new[cut_new:]



# --------------------------------------------------------------------------- #
# 配置与工具
# --------------------------------------------------------------------------- #
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')


def load_env(path: str = ENV_FILE) -> None:
    """读取项目目录下的 .env（KEY=VALUE，支持 # 注释和成对引号）。

    已经存在的环境变量优先，不会被 .env 覆盖，方便临时用 shell 变量顶替。
    """
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


def load_config(path: str = 'config.json') -> Dict:
    """读取 config.json；缺字段时用默认值补齐。

    密钥类变量（MiMo API Key、B 站 cookies）优先从项目目录的 .env 读取，
    其次才用 config.json 里的值。
    """
    load_env()

    cfg: Dict = {}
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f'加载配置文件失败: {e}')

    bili = cfg.setdefault('bilibili', {})
    bili.setdefault('cookies', {})
    bili.setdefault('headers', {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'),
        'Referer': 'https://www.bilibili.com',
    })
    env_cookies = {
        'SESSDATA': os.environ.get('BILI_SESSDATA', ''),
        'bili_jct': os.environ.get('BILI_JCT', ''),
        'DedeUserID': os.environ.get('BILI_DEDEUSERID', ''),
    }
    bili['cookies'] = {**bili['cookies'], **{k: v for k, v in env_cookies.items() if v}}
    user_agent = os.environ.get('BILI_USER_AGENT', '')
    if user_agent:
        bili['headers']['User-Agent'] = user_agent

    asr = DEFAULT_ASR.copy()
    asr.update(cfg.get('asr') or {})
    for env_key, field in (('MIMO_BASE_URL', 'base_url'),
                           ('MIMO_MODEL', 'model'),
                           ('MIMO_LANGUAGE', 'language')):
        if os.environ.get(env_key):
            asr[field] = os.environ[env_key]
    asr['api_key'] = os.environ.get('MIMO_API_KEY') or asr.get('api_key', '')
    cfg['asr'] = asr

    audio = DEFAULT_AUDIO.copy()
    audio.update(cfg.get('audio') or {})
    cfg['audio'] = audio

    organize = DEFAULT_ORGANIZE.copy()
    organize.update(cfg.get('organize') or {})
    cfg['organize'] = organize
    return cfg


def safe_name(text: str, limit: int = 80) -> str:
    """把标题清理成合法的目录名。"""
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', ' ', text or '').strip()
    name = re.sub(r'\s+', ' ', name)
    return (name[:limit] or 'untitled').strip('. ')


class FFmpeg:
    """定位并调用 ffmpeg 可执行文件。"""

    def __init__(self, explicit: str = ''):
        self.exe = self._resolve(explicit)

    @staticmethod
    def _resolve(explicit: str) -> str:
        for cand in (explicit, os.environ.get('FFMPEG_BINARY', '')):
            if cand and os.path.isfile(cand):
                return cand
        try:  # 可选依赖：imageio-ffmpeg 自带 ffmpeg 二进制
            import imageio_ffmpeg  # type: ignore

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            pass
        found = shutil.which('ffmpeg')
        if found:
            return found
        # 兜底：项目内 .tools 目录下的静态构建
        for root, _dirs, files in os.walk('.tools'):
            for name in files:
                if name.lower().startswith('ffmpeg') and name.lower().endswith('.exe'):
                    return os.path.join(root, name)
        raise RuntimeError('未找到 ffmpeg，请安装 ffmpeg 或在 config.json 里填写 audio.ffmpeg')

    def run(self, args: List[str]) -> None:
        cmd = [self.exe, '-hide_banner', '-loglevel', 'error', '-y', *args]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if proc.returncode != 0:
            raise RuntimeError(f'ffmpeg 执行失败: {proc.stderr.strip()[:800]}')


# --------------------------------------------------------------------------- #
# 语音识别：小米 MiMo ASR（OpenAI 兼容）
# --------------------------------------------------------------------------- #
class MiMoASR:
    """MiMo-V2.5-ASR 客户端。

    请求体与 OpenAI Chat Completions 一致，音频放在 user 消息的
    ``input_audio`` 内容块里（data URL），另加 ``asr_options.language``。
    """

    def __init__(self, api_key: str, base_url: str, model: str = 'mimo-v2.5-asr',
                 language: str = 'zh', timeout: int = 180, retries: int = 3):
        if not api_key:
            raise RuntimeError('缺少 MiMo API Key：请在项目根目录的 .env 里设置 MIMO_API_KEY（同名环境变量也可以）')
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.language = language
        self.timeout = timeout
        self.retries = retries

    @staticmethod
    def _encode(audio_path: str) -> str:
        fmt = os.path.splitext(audio_path)[1].lower().lstrip('.')
        if fmt not in ('mp3', 'wav'):
            raise ValueError(f'MiMo ASR 只接受 mp3 / wav，收到 {fmt or "未知格式"}')
        with open(audio_path, 'rb') as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode('ascii')
        if len(b64) > MAX_AUDIO_BASE64_BYTES:
            raise ValueError(f'单段音频 Base64 超过 10 MB（{len(b64) / 1048576:.1f} MB），请调小 audio.chunk_seconds')
        mime = 'audio/mpeg' if fmt == 'mp3' else 'audio/wav'
        return f'data:{mime};base64,{b64}'

    def transcribe(self, audio_path: str) -> str:
        payload = {
            'model': self.model,
            'messages': [{
                'role': 'user',
                'content': [{
                    'type': 'input_audio',
                    'input_audio': {'data': self._encode(audio_path)},
                }],
            }],
            'asr_options': {'language': self.language},
            'stream': False,
        }
        url = f'{self.base_url}/chat/completions'
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers={'api-key': self.api_key, 'Content-Type': 'application/json'},
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choice = (data.get('choices') or [{}])[0]
                    return (choice.get('message') or {}).get('content', '') or ''
                # 4xx（除 429）没有必要重试
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise RuntimeError(f'MiMo ASR 返回 {resp.status_code}: {resp.text[:500]}')
                last_err = RuntimeError(f'MiMo ASR 返回 {resp.status_code}: {resp.text[:300]}')
            except Exception as e:  # noqa: BLE001
                last_err = e
            if attempt < self.retries:
                wait = 2 ** attempt
                print(f'  识别失败（{last_err}），{wait}s 后重试 {attempt}/{self.retries - 1}')
                time.sleep(wait)
        raise RuntimeError(f'MiMo ASR 多次重试仍失败: {last_err}')


# --------------------------------------------------------------------------- #
# 文字整理：只做分段排版，不改内容
# --------------------------------------------------------------------------- #
class TextOrganizer:
    """调用大模型把连续的文字稿整理成便于阅读的段落。

    合同：只允许插入段落分隔和微调标点，字词必须逐字保留。每次返回都会用
    :func:`content_key` 校验（去掉空白和标点后应完全一致），不通过就重试，
    仍不通过则退回本地按句分段。
    """

    SYSTEM_PROMPT = (
        '你是严谨的中文文字编辑。你的任务只有一个：给文章分段。'
        '不允许改写、润色、增删、总结或解释任何内容。'
    )
    USER_TEMPLATE = (
        '下面是一段视频的语音转写文字，共 {chars} 字。请把它整理成便于阅读的段落。\n\n'
        '硬性要求：\n'
        '1. 逐字保留原文，一个字都不要增加、删除或替换；只有标点可以按语义微调。\n'
        '2. 段落之间用空行分隔，段落内部不要插入换行。\n'
        '3. 分成大约 {count} 段，每段 150~350 字，避免出现超长段落。\n'
        '4. 不要输出标题、序号、小标题、注释、总结或任何解释性文字。\n'
        '5. 直接输出整理后的正文，不要用代码块包裹。\n\n'
        '原文：\n<<<\n{text}\n>>>'
    )

    def __init__(self, config: Dict):
        org = config['organize']
        self.base_url = (org.get('base_url') or config['asr']['base_url']).rstrip('/')
        self.api_key = org.get('api_key') or config['asr']['api_key']
        self.model = org.get('model', 'mimo-v2.6-flash')
        self.temperature = org.get('temperature', 0.3)
        self.max_input_chars = int(org.get('max_input_chars', 6000))
        self.retries = max(1, int(org.get('retries', 2)))
        self.thinking = bool(org.get('thinking', False))

    def organize(self, text: str) -> Tuple[str, bool]:
        """返回 (整理后的文本, 是否全部由模型整理成功)。"""
        blocks = split_blocks(text, self.max_input_chars)
        results: List[str] = []
        all_ok = True
        for i, block in enumerate(blocks, 1):
            prefix = f'  [{i}/{len(blocks)}] {len(block)} 字' if len(blocks) > 1 else '  '
            print(f'{prefix} 整理分段 ...', end='', flush=True)
            organized, ok = self._organize_block(block)
            all_ok = all_ok and ok
            paragraphs = organized.count('\n\n') + 1
            print(f' {len(organized)} 字 / {paragraphs} 段' + ('' if ok else '（未通过校验，退回本地分段）'))
            results.append(organized)
        return '\n\n'.join(results), all_ok

    def _organize_block(self, block: str) -> Tuple[str, bool]:
        expected = content_key(block)
        for attempt in range(1, self.retries + 1):
            try:
                out = self._request(block)
            except Exception as e:  # noqa: BLE001
                print(f' 整理请求失败（{e}）', end='', flush=True)
                out = ''
            if out and content_key(out) == expected:
                return out, True
            if attempt < self.retries:
                print(' 内容校验未通过，重试 ...', end='', flush=True)
                time.sleep(2)
        return local_paragraphs(block), False

    def _request(self, block: str) -> str:
        target = max(2, round(len(block) / 250))
        body = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self.SYSTEM_PROMPT},
                {'role': 'user', 'content': self.USER_TEMPLATE.format(
                    chars=len(block), count=target, text=block)},
            ],
            'temperature': self.temperature,
            'stream': False,
        }
        if not self.thinking:
            body['thinking'] = {'type': 'disabled'}
        resp = requests.post(
            f'{self.base_url}/chat/completions',
            headers={'api-key': self.api_key, 'Content-Type': 'application/json'},
            json=body,
            timeout=300,
        )
        if resp.status_code != 200:
            raise RuntimeError(f'{resp.status_code} {resp.text[:200]}')
        data = resp.json()
        content = ((data.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
        content = content.strip()
        content = re.sub(r'^```[a-zA-Z]*\n', '', content)
        content = re.sub(r'\n```$', '', content)
        return content.strip()


# --------------------------------------------------------------------------- #
# B 站：信息解析与音频下载
# --------------------------------------------------------------------------- #
class BiliSpeech2Text:
    INITIAL_STATE_PATTERN = re.compile(r'window\.__INITIAL_STATE__=([\s\S]+?);\(function')

    def __init__(self, config: Dict):
        self.config = config
        self.headers = config['bilibili']['headers']
        self.cookies = config['bilibili']['cookies']
        self.audio_cfg = config['audio']
        self.ffmpeg = FFmpeg(self.audio_cfg.get('ffmpeg', ''))
        self.asr = MiMoASR(
            api_key=config['asr']['api_key'],
            base_url=config['asr']['base_url'],
            model=config['asr']['model'],
            language=config['asr']['language'],
            timeout=config['asr'].get('timeout', 180),
            retries=config['asr'].get('retries', 3),
        )
        self.organizer = TextOrganizer(config) if config['organize'].get('enabled') else None

    # ---------------- 网络 ----------------
    def _get(self, url: str, retries: int = 3, **kwargs) -> requests.Response:
        kwargs.setdefault('timeout', 60)
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, headers=self.headers, cookies=self.cookies, **kwargs)
                resp.raise_for_status()
                return resp
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < retries:
                    time.sleep(2 * attempt)
        raise RuntimeError(f'请求失败: {url} -> {last_err}')

    def resolve_url(self, url: str) -> str:
        """b23.tv 短链 → 完整地址。"""
        if re.search(r'b23\.tv|bili2233\.cn', url):
            resp = self._get(url, allow_redirects=True)
            return resp.url
        return url

    def extract_video_info(self, url: str, whole_collection: bool = False) -> List[Dict]:
        """返回待处理的视频列表：[{title, bvid, aid, cid, duration}]

        链接里已经指明某个 BV/av 号时，默认只处理这一个视频；只有显式传
        ``whole_collection=True`` 才会进入合集选择流程。
        """
        url = self.resolve_url(url)
        bv_match = BV_PATTERN.search(url)
        av_match = AV_PATTERN.search(url)
        if bv_match:
            api = f'https://api.bilibili.com/x/web-interface/view?bvid={bv_match.group(0)}'
        elif av_match:
            api = f'https://api.bilibili.com/x/web-interface/view?aid={av_match.group(1)}'
        else:
            raise RuntimeError(f'无法从链接中解析 BV/av 号: {url}')

        data = self._get(api).json()
        if data.get('code') != 0:
            raise RuntimeError(f'获取视频信息失败: {data.get("message")}')
        v = data['data']

        ugc = v.get('ugc_season')
        if ugc and whole_collection:
            return self._select_from_collection(v)

        if ugc:
            print(f'该视频属于合集《{ugc.get("title", "")}》，本次只处理当前这一集')

        pages = v.get('pages') or [{}]
        return [{
            'title': safe_name(v.get('title', '')),
            'bvid': v.get('bvid'),
            'aid': v.get('aid'),
            'cid': v.get('cid') or pages[0].get('cid'),
            'duration': v.get('duration'),
        }]

    def _select_from_collection(self, video_data: Dict) -> List[Dict]:
        season = video_data['ugc_season']
        sections = season.get('sections', [])
        print('\n' + '=' * 50)
        print(f'检测到视频合集：{season.get("title", "未知标题")}')
        print(f'合集总集数：{sum(len(s.get("episodes", [])) for s in sections)}')
        print('=' * 50)

        videos: List[Dict] = []
        for section in sections:
            for ep in section.get('episodes', []):
                if not ep.get('bvid'):
                    continue
                title = ep.get('title', '未知标题')
                if ep.get('long_title'):
                    title = f'{title} - {ep["long_title"]}'
                videos.append({
                    'title': safe_name(f'{season.get("title", "")}-{title}'),
                    'bvid': ep['bvid'],
                    'aid': ep.get('aid'),
                    'cid': ep.get('cid'),
                    'duration': ep.get('duration'),
                })
                print(f'[{len(videos)}] [{ep["bvid"]}] {title}')
        print('=' * 50)

        choice = input('选择：1=全部，2=按序号/BV号（逗号分隔），3=区间（如 1,10）\n> ').strip()
        if choice == '1':
            return videos
        if choice == '3':
            start, end = (int(x) for x in choice.replace('，', ',').replace(' ', '').split(','))
            return videos[start - 1:end]
        picked = [x.strip() for x in choice.replace('，', ',').split(',') if x.strip()]
        return [v for idx, v in enumerate(videos, 1) if str(idx) in picked or v['bvid'] in picked]

    # ---------------- 音频 ----------------
    def _playurl(self, bvid: str, cid: str) -> Dict:
        api = ('https://api.bilibili.com/x/player/playurl'
               f'?bvid={bvid}&cid={cid}&fnval=16&fnver=0&fourk=1')
        data = self._get(api).json()
        if data.get('code') != 0:
            raise RuntimeError(f'获取播放地址失败: {data.get("message")}')
        return data['data']

    def download_audio(self, video: Dict) -> str:
        """下载 DASH 里的音频轨（m4a），已存在则直接复用。"""
        out_dir = os.path.join(BASE_DIR, video['title'], 'conv')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{video["bvid"]}.m4a')
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f'音频已存在：{out_path}')
            return out_path

        play = self._playurl(video['bvid'], video['cid'])
        audios = (play.get('dash') or {}).get('audio') or []
        if not audios:
            raise RuntimeError('该视频没有可用的音频流')
        best = max(audios, key=lambda a: a.get('bandwidth', 0))
        url = best.get('baseUrl') or best.get('base_url')
        last_err: Optional[Exception] = None
        for cand in [url, *best.get('backupUrl', [])][:3]:
            try:
                with self._get(cand, stream=True) as resp, open(out_path, 'wb') as f:
                    for chunk in resp.iter_content(1 << 16):
                        f.write(chunk)
                print(f'音频下载完成：{out_path}（{os.path.getsize(out_path) / 1048576:.1f} MB）')
                return out_path
            except Exception as e:  # noqa: BLE001
                last_err = e
        if os.path.exists(out_path):
            os.remove(out_path)
        raise RuntimeError(f'音频下载失败: {last_err}')

    def to_mp3(self, audio_path: str) -> str:
        """m4a → 单声道 16k mp3（MiMo 只吃 mp3/wav）。"""
        mp3_path = os.path.splitext(audio_path)[0] + '.mp3'
        if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
            print(f'音频已转码：{mp3_path}')
            return mp3_path
        print('转码为 mp3 ...')
        self.ffmpeg.run([
            '-i', audio_path,
            '-vn', '-ac', '1', '-ar', str(self.audio_cfg['sample_rate']),
            '-c:a', 'libmp3lame', '-b:a', self.audio_cfg['bitrate'],
            mp3_path,
        ])
        return mp3_path

    def split_audio(self, mp3_path: str, bvid: str) -> List[Dict]:
        """按 chunk_seconds 切段，尽量切在静音处，并让相邻两段带一点重叠。

        返回 [{index, path, start, end}]，其中 start/end 是该段实际覆盖的音频区间。
        """
        chunk = float(self.audio_cfg['chunk_seconds'])
        overlap = float(self.audio_cfg.get('overlap_seconds', 0.0))
        out_dir = os.path.join(os.path.dirname(mp3_path), f'{bvid}_chunks')
        os.makedirs(out_dir, exist_ok=True)

        duration = float(self._probe_duration(mp3_path))
        cuts = self._cut_points(mp3_path, duration, chunk)

        bounds = [0.0, *cuts, duration]
        parts: List[Dict] = []
        for index, (start, end) in enumerate(zip(bounds, bounds[1:])):
            audio_start = max(0.0, start - overlap if index else start)
            audio_end = min(duration, end + overlap if index < len(bounds) - 2 else end)
            part = os.path.join(out_dir, f'{index:03d}.mp3')
            self.ffmpeg.run(['-ss', f'{audio_start:.3f}', '-t', f'{audio_end - audio_start:.3f}',
                             '-i', mp3_path, '-c', 'copy', part])
            parts.append({'index': index, 'path': part, 'start': audio_start, 'end': audio_end})
        return parts

    def _cut_points(self, path: str, duration: float, chunk: float) -> List[float]:
        """在理想分界点附近找一段静音，用它的中点作为实际切点。"""
        silences = self._silence_midpoints(path)
        cuts: List[float] = []
        target = chunk
        while target < duration - 5:
            window = [p for p in silences if abs(p - target) <= max(20.0, chunk * 0.35)]
            cut = min(window, key=lambda p: abs(p - target)) if window else target
            if not cuts or cut - cuts[-1] >= 20:
                cuts.append(cut)
            target += chunk
        return cuts

    def _silence_midpoints(self, path: str) -> List[float]:
        noise = self.audio_cfg.get('silence_db', -25)
        dur = self.audio_cfg.get('silence_dur', 0.3)
        proc = subprocess.run(
            [self.ffmpeg.exe, '-hide_banner', '-i', path,
             '-af', f'silencedetect=noise={noise}dB:d={dur}', '-f', 'null', '-'],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
        )
        starts = [float(x) for x in re.findall(r'silence_start:\s*([\d.]+)', proc.stderr)]
        ends = [float(x) for x in re.findall(r'silence_end:\s*([\d.]+)', proc.stderr)]
        return [(s + e) / 2 for s, e in zip(starts, ends)]

    def _probe_duration(self, path: str) -> str:
        proc = subprocess.run(
            [self.ffmpeg.exe, '-hide_banner', '-i', path],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
        )
        match = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', proc.stderr)
        if not match:
            raise RuntimeError('无法读取音频时长')
        h, m, s = match.groups()
        return str(int(h) * 3600 + int(m) * 60 + float(s))

    # ---------------- 主流程 ----------------
    def transcribe_video(self, video: Dict, organize_only: bool = False) -> List[str]:
        output_dir = os.path.join(BASE_DIR, video['title'], 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, f'{video["bvid"]}.txt')
        json_path = os.path.join(output_dir, f'{video["bvid"]}.json')
        md_path = os.path.join(output_dir, f'{video["bvid"]}.md')

        if organize_only:
            if not os.path.exists(txt_path):
                raise RuntimeError(f'找不到文字稿，无法整理：{txt_path}')
            return [p for p in (txt_path, self.organize_text(txt_path, md_path)) if p]

        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            print(f'文本已存在，跳过识别：{txt_path}')
            return [p for p in (txt_path, self.organize_text(txt_path, md_path)) if p]

        m4a = self.download_audio(video)
        mp3 = self.to_mp3(m4a)
        parts = self.split_audio(mp3, video['bvid'])

        print(f'开始识别，共 {len(parts)} 段，模型 {self.asr.model}')
        segments = []
        merged = ''
        for part in parts:
            print(f'  [{part["index"] + 1}/{len(parts)}] {part["start"]:.0f}s - {part["end"]:.0f}s ...',
                  end='', flush=True)
            text = self.transcribe_guarded(part['path'])
            print(f' {len(text)} 字')
            merged = merge_overlap(merged, text)
            segments.append({
                'index': part['index'],
                'start': round(part['start'], 2),
                'end': round(part['end'], 2),
                'text': text,
            })

        full_text = dedupe_repeats(merged)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'title': video['title'],
                'bvid': video['bvid'],
                'model': self.asr.model,
                'language': self.asr.language,
                'segments': segments,
            }, f, ensure_ascii=False, indent=2)

        print(f'文本转换完成：{txt_path}')
        return [p for p in (txt_path, self.organize_text(txt_path, md_path)) if p]

    def organize_text(self, txt_path: str, md_path: str) -> Optional[str]:
        """把文字稿交给大模型分段，输出便于阅读的 .md。"""
        if not self.organizer:
            return None
        if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
            print(f'整理稿已存在，跳过：{md_path}')
            return md_path

        text = open(txt_path, 'r', encoding='utf-8').read().strip()
        if not text:
            return None
        print(f'用 {self.organizer.model} 整理分段 ...')
        organized, ok = self.organizer.organize(text)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(organized.strip() + '\n')
        note = '' if ok else '（部分段落未通过校验，已用本地分段兜底）'
        print(f'整理完成：{md_path}{note}')
        return md_path

    def transcribe_guarded(self, audio_path: str) -> str:
        """识别一段音频；结果复读时重试一次，仍复读则做去重。"""
        best = ''
        for attempt in (1, 2):
            text = self.asr.transcribe(audio_path).strip()
            if not is_degenerate(text):
                return dedupe_repeats(text)
            print(' 检测到复读，重试一次 ...', end='', flush=True)
            cleaned = dedupe_repeats(text)
            if len(cleaned) > len(best):
                best = cleaned
        return best

    def process_video(self, url: str, whole_collection: bool = False,
                      organize_only: bool = False) -> List[str]:
        videos = self.extract_video_info(url, whole_collection=whole_collection)
        results: List[str] = []
        for i, video in enumerate(videos, 1):
            print(f'\n[{i}/{len(videos)}] 处理：{video["title"]}（{video["bvid"]}）')
            results.extend(self.transcribe_video(video, organize_only=organize_only))
        return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='B 站视频语音转文本（小米 MiMo ASR）')
    parser.add_argument('url', help='B 站视频链接，支持 b23.tv 短链')
    parser.add_argument('--config', default='config.json', help='配置文件路径')
    parser.add_argument('--model', help='覆盖 asr.model，默认 mimo-v2.5-asr')
    parser.add_argument('--language', choices=['auto', 'zh', 'en'], help='识别语种，默认 zh')
    parser.add_argument('--chunk-seconds', type=int, help='每段音频秒数，默认 70')
    parser.add_argument('--collection', action='store_true', help='视频属于合集时，交互式选择要处理的分集')
    parser.add_argument('--no-organize', action='store_true', help='不调用大模型做分段整理')
    parser.add_argument('--organize-only', action='store_true', help='跳过识别，只用已有文字稿做分段整理')
    parser.add_argument('--organize-model', help='覆盖 organize.model，默认 mimo-v2.6-flash')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.model:
        config['asr']['model'] = args.model
    if args.language:
        config['asr']['language'] = args.language
    if args.chunk_seconds:
        config['audio']['chunk_seconds'] = args.chunk_seconds
    if args.no_organize:
        config['organize']['enabled'] = False
    if args.organize_model:
        config['organize']['model'] = args.organize_model

    app = BiliSpeech2Text(config)
    if args.organize_only and not app.organizer:
        print('已禁用整理（--no-organize），无法使用 --organize-only')
        return 2
    paths = app.process_video(args.url, whole_collection=args.collection,
                              organize_only=args.organize_only)
    for path in paths:
        print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
