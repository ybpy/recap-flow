"""Portable, offline comparison of paired critic curves and two-camera recordings."""

from collections import Counter
import html
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from openpi_client.e0.artifacts import write_json


def video(attempt, file):
    with np.load(attempt["trajectory"], allow_pickle=False) as archive:
        left, right = archive["image"], archive["wrist_image"]
    if left.shape != right.shape or len(left) != attempt["policy_steps"] or left.dtype != np.uint8:
        raise ValueError("unaligned camera frames")
    _, h, w, c = left.shape
    if c != 3:
        raise ValueError("expected RGB")
    args = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{2*w}x{h}",
        "-r",
        "10",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(file),
    ]
    subprocess.run(args, input=np.concatenate([left, right], axis=2).tobytes(), check=True)


def gallery(dest, names, attempts, frames, summary):
    eligible = sorted(next(iter(frames.values())))
    fixed = []
    for task in range(10):
        for success in (True, False):
            fixed.extend(
                u
                for u in [v for v in eligible if attempts[v]["task_id"] == task and attempts[v]["success"] == success][
                    :2
                ]
            )
    scores = []
    for uid in eligible:
        votes = np.stack([frames[n][uid]["positive"] for n in names]).sum(axis=0)
        discord = votes * (len(names) - votes) / (len(names) * (len(names) - 1) / 2)
        k = min(50, len(discord))
        windows = np.convolve(discord, np.ones(k) / k, mode="valid")
        start = int(np.argmax(windows))
        scores.append((float(windows[start]), uid, start))
    scores.sort(key=lambda row: (-row[0], row[1]))
    divergent = scores[:10]
    links = {}
    for i, uid in enumerate(dict.fromkeys(fixed + [r[1] for r in divergent])):
        row = attempts[uid]
        stem = f"trace-{i:03d}"
        video(row, dest / (stem + ".mp4"))
        decisions = json.loads(Path(row["decisions_path"]).read_text())
        if decisions["episode_uid"] != uid or decisions["policy_id"] != row["policy_id"]:
            raise ValueError("decision sidecar identity mismatch")
        marks = [[d["start_step"], d["end_step"]] for d in decisions["decisions"]]
        curves = {}
        for n in names:
            with np.load(dest / n / frames[n][uid]["path"], allow_pickle=False) as a:
                curves[n] = {
                    k: (a[k].astype(int).tolist() if k == "positive" else a[k].tolist())
                    for k in ("target", "value", "window_reward", "value_delta", "advantage", "positive", "threshold")
                }
        payload = dict(
            uid=uid,
            n=row["policy_steps"],
            prompt=row["prompt"],
            success=row["success"],
            decisions=marks,
            data=curves,
            video=stem + ".mp4",
        )
        page = VIEW.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
        (dest / (stem + ".html")).write_text(page, encoding="utf8")
        links[uid] = stem + ".html"

    def link(u, start=0):
        return f'<li><a href="{links[u]}#{start}">{html.escape(u)}</a> · {"成功" if attempts[u]["success"] else "失败"} · {attempts[u]["policy_steps"]} 步</li>'

    (dest / "disagreements.html").write_text(
        "<meta charset=utf-8><h1>事后选择的分歧片段</h1><p>按连续 50 步窗口的平均两两标签分歧排序。</p><ol>"
        + "".join(link(u, t) for _, u, t in divergent)
        + '</ol><a href="index.html">总表</a>',
        encoding="utf8",
    )
    coverage = Counter((attempts[u]["task_id"], attempts[u]["success"]) for u in fixed)
    missing = [
        f"任务 {t} {'成功' if s else '失败'} {coverage[t,s]}/2"
        for t in range(10)
        for s in (True, False)
        if coverage[t, s] < 2
    ]
    rows = []
    for n, m in summary["models"].items():
        vals = m["all"]["macro"]
        cells = [n, str(m["selection"]["updates"])] + [
            f'{vals[k]["mean"]:.4f}' for k in ("ce", "mae", "clock_mae", "positive")
        ]
        rows.append("<tr>" + "".join("<td>" + v + "</td>" for v in cells) + "</tr>")
    index = "<meta charset=utf-8><style>body{font:16px system-ui;max-width:1100px;margin:3em auto}td,th{padding:.45em;text-align:left}tr:nth-child(even){background:#eee}</style>"
    index += (
        "<h1>R1 critic 独立轨迹评审</h1><p>每项先按轨迹平均再取任务宏平均。仅在同一监督定义内比较 CE 和绝对误差。</p><table><tr>"
        + "".join("<th>" + s + "</th>" for s in ["模型", "EMA 步数", "CE", "MAE", "时间对照 MAE", "positive 比例"])
        + "</tr>"
        + "".join(rows)
        + "</table>"
    )
    index += (
        '<p><a href="study.json">逐任务、分层、配对差值与 95% 区间（JSON）</a> · <a href="disagreements.html">分歧最大的片段</a></p><h2>按任务和结果预先取样</h2><p>'
        + html.escape("；".join(missing) or "每个任务各两条成功和失败。")
        + "</p><ol>"
        + "".join(link(u) for u in fixed)
        + "</ol>"
    )
    (dest / "index.html").write_text(index, encoding="utf8")
    write_json(
        dest / "gallery.json", dict(fixed=fixed, disagreements=divergent, links=links, coverage_shortfalls=missing)
    )


VIEW = """<!doctype html><meta charset=utf-8><title>轨迹评审</title><style>body{max-width:1150px;margin:1.2em auto;font:15px system-ui}video{width:520px;max-width:95%}canvas{width:100%;border:1px solid #ccc}label{margin-right:12px}pre{white-space:pre-wrap}</style>
<a href="index.html">总表</a><h2 id="title"></h2><video id="v" controls muted preload="metadata"></video><p id="info"></p><input id="slide" type="range" min="0" style="width:95%"><p id="models"></p>
<select id="field"><option value="value">预测 value 与目标（虚线）</option><option value="advantage">advantage 与阈值（虚线）</option><option value="window_reward">50 步奖励和</option><option value="value_delta">bootstrap value 变化</option><option value="positive">adv_ind · 六行标签</option></select>
<canvas id="plot" width="1100" height="210"></canvas><pre id="details"></pre>
<script>
const P=__PAYLOAD__, names=Object.keys(P.data), colors=['#2166ac','#d65c13','#168052','#874da2','#b91e48','#5c632a'];
const el=id=>document.getElementById(id), v=el('v'), slider=el('slide'), ctx=el('plot').getContext('2d');
el('title').textContent=P.prompt+' · '+(P.success?'成功':'失败');v.src=P.video;slider.max=P.n-1;
names.forEach((n,i)=>{const l=document.createElement('label'),b=document.createElement('input');b.type='checkbox';b.checked=true;b.onchange=draw;l.style.color=colors[i];l.append(b,document.createTextNode(n));el('models').append(l)});
function draw(){const t=+slider.value,k=el('field').value; const ids=names.map((n,i)=>i).filter(i=>el('models').children[i].firstChild.checked);
const X=i=>45+i*1045/Math.max(1,P.n-1);let lo=0,hi=0;const lines=[];
ids.forEach(i=>{const d=P.data[names[i]];lines.push([i,d[k],false]);if(k==='value')lines.push([i,d.target,true]);if(k==='advantage')lines.push([i,Array(P.n).fill(d.threshold),true])});
lines.forEach(([,a])=>{lo=Math.min(lo,...a);hi=Math.max(hi,...a)});const Y=y=>180-170*(y-lo)/(hi-lo||1);
ctx.clearRect(0,0,1100,210);ctx.fillStyle='#555';ctx.fillText(hi.toFixed(2),1,12);ctx.fillText(lo.toFixed(2),1,180);
P.decisions.forEach(([s,e],j)=>{if(j%2===0){ctx.fillStyle='#f2f3f5';ctx.fillRect(X(s),0,X(Math.min(e,P.n-1))-X(s),184)}});
lines.forEach(([i,a,dash])=>{ctx.strokeStyle=colors[i];ctx.fillStyle=colors[i];ctx.setLineDash(dash?[3,4]:[]);if(k==='positive'){a.forEach((z,j)=>{if(z)ctx.fillRect(X(j),12+i*28,Math.max(1,1045/P.n),14)});return}ctx.beginPath();a.forEach((z,j)=>j?ctx.lineTo(X(j),Y(z)):ctx.moveTo(X(j),Y(z)));ctx.stroke()});
ctx.setLineDash([]);ctx.strokeStyle='#222';ctx.beginPath();ctx.moveTo(X(t),0);ctx.lineTo(X(t),188);ctx.stroke();
const seg=P.decisions.find(([s,e])=>s<=t && t<e);el('info').textContent=P.uid+' · 动作 '+t+' / '+(P.n-1)+' · 执行区间 '+JSON.stringify(seg);
el('details').textContent=ids.map(i=>{const d=P.data[names[i]];return names[i]+'   目标 '+d.target[t].toFixed(3)+'   V '+d.value[t].toFixed(3)+'   A = '+d.window_reward[t].toFixed(3)+' + '+d.value_delta[t].toFixed(3)+' = '+d.advantage[t].toFixed(3)+'   阈值 '+d.threshold.toFixed(3)+'   '+(d.positive[t]?'positive':'negative')}).join('\\n')}
v.onloadedmetadata=()=>{v.currentTime=Math.min(P.n-1,Math.max(0,Number(location.hash.slice(1))||0))/10;draw()};
v.ontimeupdate=()=>{slider.value=Math.min(P.n-1,Math.floor(v.currentTime*10+1e-5));draw()};slider.oninput=()=>{v.currentTime=+slider.value/10;draw()};el('field').onchange=draw;draw();
</script>"""
