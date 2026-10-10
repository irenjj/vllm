"use strict";
(async () => {
  const $ = (id) => document.getElementById(id);
  try {
    const response = await fetch("data.json");
    if (!response.ok) throw Error("Run prepare_demo.py first");
    const data = await response.json();
    let diag = data.diagnostics;
    const selected = new Set(diag.query_positions),
      cache = new Map();
    let row = new Map(),
      pending = false;
    const video = $("video"),
      canvas = $("heat"),
      ctx = canvas.getContext("2d");
    const layer = $("layer"),
      head = $("head");
    for (const name of data.layer_names) layer.add(new Option(name, name));
    layer.value = Object.keys(diag.layers)[0];
    function key(positions, name) {
      return JSON.stringify({
        positions: [...positions].sort((a, b) => a - b),
        layer_name: name,
      });
    }
    cache.set(key(diag.query_positions, layer.value), diag);
    function visible(t) {
      return (
        !t.visual &&
        !t.text.includes("<|") &&
        !t.text.includes("<think") &&
        !t.text.includes("</think") &&
        !/seconds>/.test(t.text) &&
        t.text.trim()
      );
    }
    function renderTokens() {
      $("prompt").replaceChildren();
      $("answer").replaceChildren();
      for (const t of data.tokens) {
        if (!$("special").checked && !visible(t)) continue;
        const b = document.createElement("button");
        b.className = "token";
        b.textContent = t.text.trim() ? t.text : JSON.stringify(t.text);
        b.title = `Position ${t.position}`;
        b.classList.toggle("active", selected.has(t.position));
        b.onclick = () => {
          selected.has(t.position)
            ? selected.delete(t.position)
            : selected.add(t.position);
          b.classList.toggle("active", selected.has(t.position));
          $("status").textContent =
            `已选 ${selected.size} 个 token，点击回放更新`;
        };
        $(t.position < data.prompt_length ? "prompt" : "answer").append(b);
      }
    }
    function setHeads() {
      const old = head.value;
      head.replaceChildren(new Option("返回的全部 head 平均", "mean"));
      for (const h of Object.values(diag.layers)[0].head_indices)
        head.add(new Option(`Head ${h}`, String(h)));
      if ([...head.options].some((o) => o.value === old)) head.value = old;
    }
    function calculate() {
      const a = Object.values(diag.layers)[0];
      row = new Map();
      diag.key_positions.forEach((position, k) => {
        let sum = 0,
          n = 0;
        a.weights.forEach((rows, h) => {
          if (head.value === "mean" || +head.value === a.head_indices[h])
            for (const q of rows) {
              sum += q[k];
              n++;
            }
        });
        row.set(position, n ? sum / n : 0);
      });
      $("textmap").replaceChildren();
      const textTokens = data.tokens.filter(visible);
      const max = Math.max(
        0,
        ...textTokens.map((t) => row.get(t.position) || 0),
      );
      for (const t of textTokens) {
        const e = document.createElement("span"),
          weight = row.get(t.position) || 0;
        e.textContent = t.text;
        e.title = `Position ${t.position}: ${(weight * 100).toFixed(5)}%`;
        e.style.background = `rgba(250,151,55,${max ? (weight / max) * 0.8 : 0})`;
        $("textmap").append(e);
      }
      draw();
    }
    function draw() {
      if (!video.videoWidth || !row.size) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const [nt, h, w] = data.grid,
        times = data.timestamps;
      let group = 0;
      for (let i = 1; i < nt; i++)
        if (
          Math.abs(times[i] - video.currentTime) <
          Math.abs(times[group] - video.currentTime)
        )
          group = i;
      const all = data.visual_positions.map((p) => row.get(p) || 0),
        max = Math.max(...all);
      let mass = 0;
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
          const value = all[group * h * w + y * w + x];
          mass += value;
          ctx.fillStyle = `rgba(255,130,30,${max ? (value / max) * +$("opacity").value : 0})`;
          ctx.fillRect(
            (x * canvas.width) / w,
            (y * canvas.height) / h,
            canvas.width / w,
            canvas.height / h,
          );
        }
      $("time").textContent = video.currentTime.toFixed(2) + " s";
      $("seek").value = video.currentTime;
      $("stats").textContent =
        `采样 ${times[group]} s · 当前时间组 ${(mass * 100).toFixed(2)}% · 整段视频 ${(all.reduce((a, b) => a + b, 0) * 100).toFixed(2)}% · ${diag.query_positions.length} 个 token`;
    }
    $("update").onclick = async () => {
      if (pending) return;
      if (!selected.size || selected.size > 128) {
        $("status").textContent = "请选择 1–128 个 token";
        return;
      }
      pending = true;
      $("update").disabled = true;
      $("status").textContent = "正在回放…";
      try {
        const id = key(selected, layer.value);
        if (cache.has(id)) diag = cache.get(id);
        else {
          const r = await fetch("/api/replay", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: id,
            }),
            result = await r.json();
          if (!r.ok || result.error)
            throw Error(result.error || `HTTP ${r.status}`);
          diag = result;
          cache.set(id, result);
        }
        setHeads();
        calculate();
        $("status").textContent =
          `完成：${diag.query_positions.length} 个 token`;
      } catch (e) {
        $("status").textContent = e.message;
      } finally {
        pending = false;
        $("update").disabled = false;
      }
    };
    $("special").onchange = renderTokens;
    layer.onchange = () => ($("status").textContent = "点击回放更新所选层");
    head.onchange = calculate;
    $("opacity").oninput = draw;
    video.onloadedmetadata = () => {
      $("seek").max = video.duration;
      draw();
    };
    video.ontimeupdate = draw;
    $("play").onclick = async () => {
      try {
        if (video.paused) await video.play();
        else video.pause();
      } catch (e) {
        $("status").textContent = e.message;
      }
    };
    $("seek").oninput = () => {
      video.currentTime = +$("seek").value;
      draw();
    };
    renderTokens();
    setHeads();
    calculate();
    $("status").textContent = "已载入真实采集结果";
  } catch (e) {
    $("status").textContent = e.message;
  }
})();
