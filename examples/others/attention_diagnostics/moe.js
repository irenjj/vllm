/* Replay diagnostics only; no inferred expert semantics or fabricated routes. */
"use strict";
const $ = (id) => document.getElementById(id);
const tasks = new Map();
let current = null;
function element(tag, text) {
  const e = document.createElement(tag);
  e.textContent = text;
  return e;
}
function indices(text, limit) {
  const result = new Set();
  for (const part of text.split(",")) {
    const match = part.trim().match(/^(\d+)(?:-(\d+))?$/);
    if (!match) throw Error("位置/层格式应为 0-47 或 3,7,19");
    const start = Number(match[1]),
      end = Number(match[2] ?? match[1]);
    if (end < start || end >= limit) throw Error("位置或层超出范围");
    for (let i = start; i <= end; i++) result.add(i);
  }
  return [...result].sort((a, b) => a - b);
}
function validateTask(t) {
  if (
    !t ||
    typeof t.id !== "string" ||
    !t.id ||
    typeof t.model !== "string" ||
    !t.model ||
    typeof t.name !== "string"
  )
    throw Error("任务需要 id、name、model（含版本）");
  const d = t.diagnostics;
  if (
    d?.capture_kind !== "moe" ||
    !Number.isInteger(d.sequence_length) ||
    d.sequence_length < 1 ||
    !Array.isArray(d.query_positions) ||
    !d.query_positions.length ||
    d.query_positions.length > 100000 ||
    new Set(d.query_positions).size !== d.query_positions.length ||
    d.query_positions.some(
      (p) => !Number.isInteger(p) || p < 0 || p >= d.sequence_length,
    )
  )
    throw Error("无效的 MoE token 位置");
  if (
    !d.layers ||
    !Object.keys(d.layers).length ||
    Object.keys(d.layers).length > 128
  )
    throw Error("缺少 MoE 层数据");
  for (const l of Object.values(d.layers)) {
    if (
      l.expert_id_space !== "logical" ||
      !Number.isInteger(l.num_experts) ||
      l.num_experts < 1 ||
      l.num_experts > 65536 ||
      l.expert_ids?.length !== d.query_positions.length ||
      l.routing_weights?.length !== d.query_positions.length
    )
      throw Error("无效的专家数据");
    l.expert_ids.forEach((ids, i) => {
      const weights = l.routing_weights[i];
      if (
        !Array.isArray(ids) ||
        !ids.length ||
        ids.length > l.num_experts ||
        new Set(ids).size !== ids.length ||
        !Array.isArray(weights) ||
        weights.length !== ids.length ||
        ids.some((x) => !Number.isInteger(x) || x < 0 || x >= l.num_experts) ||
        weights.some((x) => !Number.isFinite(x) || x < 0)
      )
        throw Error("无效的专家 ID 或路由权重");
    });
  }
  return t;
}
function addTasks(incoming) {
  const candidate = new Map(tasks);
  for (const t of incoming) candidate.set(validateTask(t).id, t);
  if (new Set([...candidate.values()].map((t) => t.model)).size > 1)
    throw Error("不能汇总不同模型版本，请先清空任务组");
  const sizes = new Map();
  for (const t of candidate.values())
    for (const [name, l] of Object.entries(t.diagnostics.layers)) {
      if (sizes.has(name) && sizes.get(name) !== l.num_experts)
        throw Error("同名层专家数量不一致");
      sizes.set(name, l.num_experts);
    }
  tasks.clear();
  for (const [id, t] of candidate) tasks.set(id, t);
  refresh();
}
function refresh() {
  $("tasks").replaceChildren();
  for (const t of tasks.values()) {
    const p = element(
      "p",
      `${t.name} · ${t.diagnostics.query_positions.length} token · ${Object.keys(t.diagnostics.layers).length} 层 `,
    );
    const b = element("button", "移除");
    b.onclick = () => {
      tasks.delete(t.id);
      refresh();
    };
    p.append(b);
    $("tasks").append(p);
  }
  const previous = $("taskFilter").value,
    layer = $("layer").value;
  $("taskFilter").replaceChildren(new Option("整个任务组", ""));
  for (const t of tasks.values()) $("taskFilter").add(new Option(t.name, t.id));
  if (tasks.has(previous)) $("taskFilter").value = previous;
  const names = [
    ...new Set(
      [...tasks.values()].flatMap((t) => Object.keys(t.diagnostics.layers)),
    ),
  ].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  $("layer").replaceChildren(...names.map((n) => new Option(n, n)));
  if (names.includes(layer)) $("layer").value = layer;
  render();
}
function render() {
  const name = $("layer").value,
    subset = [...tasks.values()].filter(
      (t) =>
        (!$("taskFilter").value || t.id === $("taskFilter").value) &&
        t.diagnostics.layers[name],
    );
  $("grid").replaceChildren();
  $("details").replaceChildren();
  const overview = document.createElement("table"),
    allTasks = [...tasks.values()].filter(
      (t) => !$("taskFilter").value || t.id === $("taskFilter").value,
    );
  for (const layer of [
    ...new Set(allTasks.flatMap((t) => Object.keys(t.diagnostics.layers))),
  ].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))) {
    const counts = new Map();
    let total = 0,
      experts = 0;
    for (const t of allTasks) {
      const l = t.diagnostics.layers[layer];
      if (!l) continue;
      experts = l.num_experts;
      total += l.expert_ids.length;
      for (const ids of l.expert_ids)
        for (const id of ids) counts.set(id, (counts.get(id) || 0) + 1);
    }
    const row = document.createElement("tr"),
      cell = document.createElement("td"),
      button = element("button", layer.replace("language_model.model.", ""));
    button.onclick = () => {
      $("layer").value = layer;
      render();
    };
    cell.append(button);
    row.append(
      cell,
      element("td", `${counts.size}/${experts} 专家 · ${total} token`),
      element(
        "td",
        [...counts]
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
          .map(([id, n]) => `E${id}: ${n} 次`)
          .join(" · "),
      ),
    );
    overview.append(row);
  }
  $("overview").replaceChildren(overview);
  if (!subset.length) {
    $("summary").textContent = "尚无数据";
    return;
  }
  const n = subset[0].diagnostics.layers[name].num_experts,
    counts = Array(n).fill(0),
    weights = Array(n).fill(0);
  let tokens = 0;
  for (const t of subset) {
    const l = t.diagnostics.layers[name];
    tokens += l.expert_ids.length;
    l.expert_ids.forEach((ids, i) =>
      ids.forEach((id, k) => {
        counts[id]++;
        weights[id] += l.routing_weights[i][k];
      }),
    );
  }
  const values = $("metric").value === "count" ? counts : weights,
    max = Math.max(...values);
  $("summary").textContent =
    `${subset.length} 个任务 · ${tokens} 个已采集 token · ${counts.filter((v) => v > 0).length}/${n} 个专家被选择 · 不跨层合并专家编号`;
  for (let id = 0; id < n; id++) {
    const b = element(
      "button",
      `E${id}\n${values[id].toFixed($("metric").value === "count" ? 0 : 2)}`,
    );
    b.style.background = `rgba(39,167,159,${max ? 0.12 + (0.8 * values[id]) / max : 0.12})`;
    b.title = `${counts[id]} 次 / ${tokens} token (${((counts[id] / tokens) * 100).toFixed(2)}%)；权重和 ${weights[id]}`;
    b.onclick = () => showExpert(id, subset, name);
    $("grid").append(b);
  }
}
function showExpert(id, subset, name) {
  const table = document.createElement("table");
  const header = document.createElement("tr");
  ["任务", "Token 绝对位置", "文本", "路由权重"].forEach((s) =>
    header.append(element("th", s)),
  );
  table.append(header);
  let shown = 0,
    total = 0;
  for (const t of subset) {
    const d = t.diagnostics,
      l = d.layers[name];
    l.expert_ids.forEach((ids, i) => {
      const k = ids.indexOf(id);
      if (k < 0) return;
      total++;
      if (shown++ >= 1000) return;
      const row = document.createElement("tr"),
        pos = d.query_positions[i],
        token = t.tokens?.find((x) => x.position === pos);
      [
        t.name,
        String(pos),
        token?.text ?? "—",
        l.routing_weights[i][k].toFixed(6),
      ].forEach((s) => row.append(element("td", s)));
      table.append(row);
    });
  }
  $("details").replaceChildren(
    element("h3", `Expert ${id} · ${total} 次选择（最多显示 1000 行）`),
    table,
  );
}
$("taskFilter").onchange = render;
$("layer").onchange = render;
$("metric").onchange = render;
$("clear").onclick = () => {
  tasks.clear();
  refresh();
};
$("import").onchange = async (e) => {
  try {
    const all = [];
    for (const file of e.target.files) {
      if (file.size > 50 * 1024 * 1024) throw Error("单文件不能超过 50 MiB");
      const obj = JSON.parse(await file.text());
      all.push(...(obj.tasks ?? [obj]));
    }
    addTasks(all);
    $("status").textContent = `导入成功，当前 ${tasks.size} 个任务`;
  } catch (err) {
    $("status").textContent = err.message;
  } finally {
    e.target.value = "";
  }
};
$("export").onclick = () => {
  const blob = new Blob(
      [
        JSON.stringify({
          schema: "vllm-moe-diagnostics-v1",
          tasks: [...tasks.values()],
        }),
      ],
      { type: "application/json" },
    ),
    url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = "moe-task-group.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("run").onclick = async () => {
  $("run").disabled = true;
  try {
    if (!current) {
      const r = await fetch("data.json");
      if (!r.ok) throw Error("当前示例 data.json 不可用，可导入其他任务结果");
      current = await r.json();
    }
    const model = current.model_revision ?? $("model").value.trim();
    if (!model) throw Error("请填写模型权重版本，任务组按此版本校验");
    const scope = $("scope").value,
      n = current.tokens.length;
    const positions =
      scope === "custom"
        ? indices($("positions").value, n)
        : current.tokens
            .filter(
              (t) =>
                scope === "all" ||
                (scope === "prompt"
                  ? t.position < current.prompt_length
                  : t.position >= current.prompt_length),
            )
            .map((t) => t.position);
    if (!positions.length) throw Error("没有选择 token");
    const available = current.moe_layer_names || [];
    const layers = indices($("layers").value, available.length).map(
      (i) => available[i],
    );
    let combined = null;
    for (let start = 0; start < positions.length; start += 128) {
      $("status").textContent =
        `正在回放 ${start + 1}–${Math.min(start + 128, positions.length)} / ${positions.length} token…`;
      const r = await fetch("/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kind: "moe",
          positions: positions.slice(start, start + 128),
          layer_names: layers,
        }),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw Error(d.error || `HTTP ${r.status}`);
      if (
        JSON.stringify(d.query_positions) !==
          JSON.stringify(positions.slice(start, start + 128)) ||
        d.sequence_length !== n ||
        JSON.stringify(Object.keys(d.layers).sort()) !==
          JSON.stringify([...layers].sort())
      )
        throw Error("回放响应与所选 token/层不一致");
      validateTask({
        id: "check",
        name: "check",
        model: "check",
        diagnostics: d,
      });
      if (!combined) combined = d;
      else {
        if (
          d.sequence_length !== combined.sequence_length ||
          Object.keys(d.layers).join() !== Object.keys(combined.layers).join()
        )
          throw Error("回放批次不一致");
        combined.query_positions.push(...d.query_positions);
        for (const [name, l] of Object.entries(d.layers)) {
          const dst = combined.layers[name];
          if (l.num_experts !== dst.num_experts)
            throw Error("模型在采集期间改变");
          dst.expert_ids.push(...l.expert_ids);
          dst.routing_weights.push(...l.routing_weights);
        }
      }
    }
    addTasks([
      {
        id: crypto.randomUUID(),
        name: $("task").value || "未命名任务",
        model,
        tokens: current.tokens,
        diagnostics: combined,
      },
    ]);
    $("status").textContent =
      "采集完成。分批结果来自多次回放，不是原始生成时的路由录像。";
  } catch (err) {
    $("status").textContent = `采集失败，未加入任务组：${err.message}`;
  } finally {
    $("run").disabled = false;
  }
};
refresh();

$("sample").onclick = async () => {
  try {
    const r = await fetch("moe-task-group.json");
    if (!r.ok) throw Error("尚无实测任务组文件，请采集或导入");
    const d = await r.json();
    addTasks(d.tasks);
    $("status").textContent = "已载入真实 GPU 回放结果，可继续导入其他任务";
  } catch (e) {
    $("status").textContent = e.message;
  }
};
fetch("moe-task-group.json")
  .then((r) => (r.ok ? r.json() : null))
  .then((d) => {
    if (d) {
      addTasks(d.tasks);
      $("status").textContent = "已载入真实 GPU 回放结果，可继续导入其他任务";
    }
  })
  .catch((e) => {
    $("status").textContent = e.message;
  });

fetch("data.json")
  .then((r) => (r.ok ? r.json() : null))
  .then((data) => {
    if (!data) return;
    current = data;
    if (data.moe_layer_names?.length) {
      $("layers").value = `0-${data.moe_layer_names.length - 1}`;
      $("layers").title = data.moe_layer_names
        .map((name, i) => `${i}: ${name}`)
        .join("\n");
    }
    if (data.model_revision) {
      $("model").value = data.model_revision;
      $("model").title = data.model_revision;
      $("model").readOnly = true;
    }
  })
  .catch(() => {});
