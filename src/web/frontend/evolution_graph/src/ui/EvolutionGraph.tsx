import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import G6 from "@antv/g6";
import { Streamlit, withStreamlitConnection, ComponentProps } from "streamlit-component-lib";

type AnyDict = Record<string, any>;

type EvoDelta = {
  enter?: string[];
  exit?: string[];
};

type EvoTimelineV1 = {
  schema?: string;
  base_graph?: {
    nodes?: AnyDict[];
    edges?: AnyDict[];
  };
  frames?: string[];
  deltas?: EvoDelta[];
};

type EvoPayload = {
  nodes?: AnyDict[];
  edges?: AnyDict[];
  frames?: string[];
  display_mode?: string;
  timeline?: EvoTimelineV1;
};

function parseTime(s: string): number | null {
  const v = String(s || "").trim();
  if (!v) return null;
  const t = Date.parse(v);
  return Number.isFinite(t) ? t : null;
}

function clampInt(v: number, lo: number, hi: number): number {
  const x = Math.trunc(v);
  return Math.max(lo, Math.min(hi, x));
}

function toEvCount(n: AnyDict): number {
  const ev = n?.evidence;
  if (Array.isArray(ev)) return ev.length;
  const c = Number(n?.evidence_count);
  return Number.isFinite(c) ? c : 0;
}

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const h = String(hex || "").trim();
  const m = /^#?([0-9a-fA-F]{6})$/.exec(h);
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function rgba(hex: string, a: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return `rgba(255,255,255,${a})`;
  return `rgba(${rgb.r},${rgb.g},${rgb.b},${a})`;
}

const COLORS = {
  dark: {
    WEAK: "#BDC3C7",
    MED: "#F39C12",
    STRONG: "#C0392B",
    ENTER: "#2ECC71",
    EXIT: "#C0392B",
    REL_KIND_STATE: "#60A5FA",
    REL_KIND_EVENT: "#F59E0B",
    NODE_FILL: "#4FA6D8",
    NODE_STROKE: "rgba(255,255,255,0.18)",
    REL_STROKE_ACTIVE: "rgba(255,255,255,0.42)",
    REL_STROKE_INACTIVE: "rgba(255,255,255,0.14)",
    TOOLTIP_BG: "rgba(10, 14, 28, 0.92)",
    TOOLTIP_COLOR: "rgba(255,255,255,0.92)",
  },
  light: {
    WEAK: "#9CA3AF",
    MED: "#D97706",
    STRONG: "#DC2626",
    ENTER: "#16A34A",
    EXIT: "#DC2626",
    REL_KIND_STATE: "#2563EB",
    REL_KIND_EVENT: "#B45309",
    NODE_FILL: "#3B82F6",
    NODE_STROKE: "rgba(0,0,0,0.18)",
    REL_STROKE_ACTIVE: "rgba(0,0,0,0.42)",
    REL_STROKE_INACTIVE: "rgba(0,0,0,0.14)",
    TOOLTIP_BG: "rgba(255, 255, 255, 0.95)",
    TOOLTIP_COLOR: "#1F2937",
  },
};

function EvolutionGraphBase(props: ComponentProps) {
  const payload = (props.args?.payload || {}) as EvoPayload;
  const timeline = payload.timeline && typeof payload.timeline === "object" ? payload.timeline : undefined;
  const baseGraph = timeline?.base_graph;

  const nodesRaw = Array.isArray(baseGraph?.nodes) ? baseGraph.nodes : Array.isArray(payload.nodes) ? payload.nodes : [];
  const edgesRaw = Array.isArray(baseGraph?.edges) ? baseGraph.edges : Array.isArray(payload.edges) ? payload.edges : [];
  const frames = Array.isArray(timeline?.frames) ? timeline.frames : Array.isArray(payload.frames) ? payload.frames : [];

  const initialIdx = clampInt(Number(props.args?.initialFrameIdx ?? 0), 0, Math.max(0, frames.length - 1));
  const [frameIdx, setFrameIdx] = useState<number>(initialIdx);
  const [playing, setPlaying] = useState<boolean>(false);
  const [speedMs, setSpeedMs] = useState<number>(Number(props.args?.speedMs ?? 420));
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoverNodeId, setHoverNodeId] = useState<string | null>(null);

  const [displayMode, setDisplayMode] = useState<string>(String(props.args?.displayMode || payload.display_mode || "当前激活"));

  const [theme, setTheme] = useState<"dark" | "light">((props.theme as any)?.base === "light" ? "light" : "dark");
  const C = COLORS[theme];

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    // Sync with Streamlit theme
    if (props.theme) {
       const nextTheme = (props.theme as any).base === "light" ? "light" : "dark";
       setTheme(nextTheme);
    }
  }, [props.theme]);

  const height = Number(props.args?.height ?? 720);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<any>(null);
  const renderedRef = useRef<boolean>(false);
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);
  const frameIdxRef = useRef<number>(initialIdx);
  const sliderValRef = useRef<number>(initialIdx);
  const prevFrameIdxRef = useRef<number>(-1);
  const activeRelIdsRef = useRef<Set<string>>(new Set());
  const entityCountsRef = useRef<Map<string, number>>(new Map());
  const nodeItemByIdRef = useRef<Map<string, any>>(new Map());
  const edgeItemsByRelRef = useRef<Map<string, any[]>>(new Map());
  const prevEnterRef = useRef<Set<string>>(new Set());
  const prevExitSoonRef = useRef<Set<string>>(new Set());
  const displayModeRef = useRef<string>(displayMode);

  const [autoFocus, setAutoFocus] = useState<boolean>(false);

  const nodeById = useMemo(() => {
    const m = new Map<string, AnyDict>();
    for (const n of nodesRaw) {
      const id = String(n?.id || "").trim();
      if (id) m.set(id, n);
    }
    return m;
  }, [nodesRaw]);

  const relIntervals = useMemo(() => {
    const m = new Map<string, { start: number; end: number | null }>();
    for (const [id, n] of nodeById.entries()) {
      const t = String(n?.type || "").trim();
      if (t !== "relation_state") continue;
      const start = parseTime(String(n?.interval_start || n?.valid_from || n?.time || ""));
      if (start == null) continue;
      const endRaw = String(n?.interval_end || n?.valid_to || "");
      const end = endRaw ? parseTime(endRaw) : null;
      if (end != null && end < start) m.set(id, { start, end: null });
      else m.set(id, { start, end });
    }
    return m;
  }, [nodeById]);

  const relPairs = useMemo(() => {
    const sMap = new Map<string, string>();
    const oMap = new Map<string, string>();
    for (const e of edgesRaw) {
      const u = String(e?.from || "").trim();
      const v = String(e?.to || "").trim();
      if (!u || !v) continue;
      const nu = nodeById.get(u);
      const nv = nodeById.get(v);
      if (nu?.type === "relation_state" && nv?.type === "entity") oMap.set(u, v);
      if (nv?.type === "relation_state" && nu?.type === "entity") sMap.set(v, u);
    }
    const out = new Map<string, { s: string; o: string }>();
    for (const [rid, s] of sMap.entries()) {
      const o = oMap.get(rid);
      if (s && o) out.set(rid, { s, o });
    }
    return out;
  }, [edgesRaw, nodeById]);

  const frameTimes = useMemo(() => {
    const out: number[] = [];
    for (const s of frames) {
      const t = parseTime(s);
      if (t != null) out.push(t);
    }
    return out;
  }, [frames]);

  const effectiveDeltas = useMemo(() => {
    const raw = Array.isArray(timeline?.deltas) ? timeline?.deltas : null;
    if (raw && raw.length === frameTimes.length) {
      const out: EvoDelta[] = [];
      for (const d of raw) {
        const enter = Array.isArray((d as any)?.enter) ? (d as any).enter.map((x: any) => String(x)) : undefined;
        const exit = Array.isArray((d as any)?.exit) ? (d as any).exit.map((x: any) => String(x)) : undefined;
        out.push({ enter, exit });
      }
      return out;
    }

    if (!frameTimes.length || !relIntervals.size) return null;

    const out: EvoDelta[] = Array.from({ length: frameTimes.length }, () => ({}));
    const enterAt: string[][] = Array.from({ length: frameTimes.length }, () => []);
    const exitAt: string[][] = Array.from({ length: frameTimes.length }, () => []);

    for (const [rid, itv] of relIntervals.entries()) {
      const start = itv.start;
      const end = itv.end;

      if (displayMode === "累积至当前") {
        let enterIdx = -1;
        if (start <= frameTimes[0]) enterIdx = 0;
        else {
          for (let i = 1; i < frameTimes.length; i++) {
            if (frameTimes[i - 1] < start && start <= frameTimes[i]) {
              enterIdx = i;
              break;
            }
          }
        }
        if (enterIdx >= 0) enterAt[enterIdx].push(rid);
        continue;
      }

      if (end != null && end < frameTimes[0]) continue;

      let enterIdx = -1;
      if (start <= frameTimes[0] && (end == null || frameTimes[0] <= end)) enterIdx = 0;
      else {
        for (let i = 1; i < frameTimes.length; i++) {
          if (frameTimes[i - 1] < start && start <= frameTimes[i]) {
            enterIdx = i;
            break;
          }
        }
      }
      if (enterIdx >= 0) enterAt[enterIdx].push(rid);

      if (end == null) continue;
      for (let i = 0; i < frameTimes.length; i++) {
        if (frameTimes[i] > end && (i === 0 || frameTimes[i - 1] <= end)) {
          exitAt[i].push(rid);
          break;
        }
      }
    }

    for (let i = 0; i < frameTimes.length; i++) {
      if (enterAt[i].length) out[i].enter = enterAt[i];
      if (exitAt[i].length) out[i].exit = exitAt[i];
    }

    return out;
  }, [timeline?.deltas, frameTimes, relIntervals, displayMode]);

  useEffect(() => {
    const hi = Math.max(0, frameTimes.length - 1);
    setFrameIdx((cur: number) => clampInt(cur, 0, hi));
    sliderValRef.current = clampInt(sliderValRef.current, 0, hi);
  }, [frameTimes.length]);

  const timeNow = useMemo(() => {
    if (!frameTimes.length) return null;
    const idx = clampInt(frameIdx, 0, frameTimes.length - 1);
    return frameTimes[idx];
  }, [frameTimes, frameIdx]);

  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  useEffect(() => {
    frameIdxRef.current = frameIdx;
  }, [frameIdx]);

  useEffect(() => {
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      if (graphRef.current) {
        try {
          graphRef.current.destroy();
        } catch {}
        graphRef.current = null;
      }
      renderedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const w = Math.max(300, el.clientWidth || 900);
    const h = Math.max(320, Math.trunc(height) || 720);

    if (!graphRef.current || graphRef.current?.get?.("destroyed")) {
      const graph = new G6.Graph({
        container: el,
        width: w,
        height: h,
        layout: { type: "preset" },
        modes: { default: ["drag-canvas", "zoom-canvas"] },
        defaultNode: {
          type: "circle",
          size: 20,
          style: { fill: C.NODE_FILL, stroke: C.NODE_STROKE, lineWidth: 1 },
          labelCfg: { style: { fill: theme === "dark" ? "rgba(255,255,255,0.9)" : "rgba(0,0,0,0.9)", fontSize: 12 } },
        },
        defaultEdge: {
          style: {
            stroke: rgba(C.WEAK, 0.35),
            lineWidth: 1,
            endArrow: { path: G6.Arrow.triangle(6, 8, 2), d: 2, fill: rgba(C.WEAK, 0.4) },
          },
        },
        plugins: [
          new G6.Tooltip({
            offsetX: 10,
            offsetY: 10,
            itemTypes: ["node", "edge"],
            getContent: (e: any) => {
              const div = document.createElement("div");
              div.style.padding = "10px 12px";
              div.style.background = C.TOOLTIP_BG;
              div.style.border = theme === "dark" ? "1px solid rgba(255,255,255,0.12)" : "1px solid rgba(0,0,0,0.12)";
              div.style.borderRadius = "10px";
              div.style.boxShadow = "0 12px 40px rgba(0,0,0,0.55)";
              div.style.color = C.TOOLTIP_COLOR;
              div.style.maxWidth = "420px";
              
              const model = e?.item?.getModel?.() || {};
              const title = String(model?._ml_title || model?.title || model?.label || model?.id || "");
              div.innerHTML = title;
              return div;
            },
          }),
        ],
      });

      graphRef.current = graph;
      renderedRef.current = false;
      graph.on("node:click", (evt: any) => {
        const model = evt?.item?.getModel?.() || {};
        const id = String(model?.id || "");
        if (id) {
            setSelectedNodeId((prev) => (prev === id ? null : id));
            Streamlit.setComponentValue({ selected: id, frame_idx: frameIdxRef.current });
        }
      });
      graph.on("canvas:click", () => {
        setSelectedNodeId(null);
        Streamlit.setComponentValue({ selected: null, frame_idx: frameIdxRef.current });
      });
      graph.on("node:mouseenter", (evt: any) => {
        const model = evt?.item?.getModel?.() || {};
        const id = String(model?.id || "");
        if (id) setHoverNodeId(id);
      });
      graph.on("node:mouseleave", () => {
        setHoverNodeId(null);
      });
    }

    const graph = graphRef.current;
    const data = {
      nodes: nodesRaw.map((n) => ({
        id: String(n?.id || ""),
        label: String(n?.label || n?.id || ""),
        type: String(n?.type || ""),
        x: Number(n?.x ?? 0),
        y: Number(n?.y ?? 0),
        _ml_title: n?._ml_title,
        predicate: n?.predicate,
        evidence: n?.evidence,
        interval_start: n?.interval_start,
        interval_end: n?.interval_end,
        relation_kind: n?.relation_kind,
      })),
      edges: edgesRaw.map((e, idx) => ({
        id: String(e?.id || `e_${idx}`),
        source: String(e?.from || ""),
        target: String(e?.to || ""),
        type: String(e?.type || ""),
        _ml_title: e?._ml_title,
        title: e?.title,
      })),
    };

    graph.changeSize(w, h);
    if (!renderedRef.current) {
      graph.data(data);
      graph.render();
      graph.fitView(30);
      renderedRef.current = true;
    } else {
      graph.changeData(data);
    }

    const nodeItems = new Map<string, any>();
    for (const node of graph.getNodes()) {
      const m = node.getModel();
      const id = String(m?.id || "").trim();
      if (id) nodeItems.set(id, node);
    }
    nodeItemByIdRef.current = nodeItems;

    const edgeItemsByRel = new Map<string, any[]>();
    for (const edge of graph.getEdges()) {
      const m = edge.getModel();
      const s = String(m?.source || "");
      const t = String(m?.target || "");
      const rid = relIntervals.has(s) ? s : relIntervals.has(t) ? t : "";
      if (!rid) continue;
      const arr = edgeItemsByRel.get(rid) || [];
      arr.push(edge);
      edgeItemsByRel.set(rid, arr);
    }
    edgeItemsByRelRef.current = edgeItemsByRel;

    prevFrameIdxRef.current = -1;
    activeRelIdsRef.current = new Set();
    entityCountsRef.current = new Map();
    prevEnterRef.current = new Set();
    prevExitSoonRef.current = new Set();

    const ro = new ResizeObserver((entries) => {
      for (const it of entries) {
        const cw = Math.max(300, Math.trunc(it.contentRect.width));
        const ch = Math.max(320, Math.trunc(height));
        graph.changeSize(cw, ch);
      }
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
    };
  }, [nodesRaw, edgesRaw, height, theme]);

  useEffect(() => {
    const graph = graphRef.current;
    if (!graph || timeNow == null) return;

    const frameIdxSafe = clampInt(frameIdx, 0, Math.max(0, frameTimes.length - 1));
    const useDeltas = Boolean(effectiveDeltas && effectiveDeltas.length === frameTimes.length);
    const prevIdx = prevFrameIdxRef.current;
    const isSequential = prevIdx >= 0 && frameIdxSafe === prevIdx + 1;
    const isLoopToZero = prevIdx === frameTimes.length - 1 && frameIdxSafe === 0;
    const displayModeChanged = displayModeRef.current !== displayMode;
    displayModeRef.current = displayMode;
    const tweenMs = Math.max(120, Math.min(420, Math.trunc(Number(speedMs) * 0.65)));

    const computeActiveSet = (t: number) => {
      const out = new Set<string>();
      for (const [rid, itv] of relIntervals.entries()) {
        if (displayMode === "累积至当前") {
          if (itv.start <= t) out.add(rid);
        } else {
          if (itv.start <= t && (itv.end == null || t <= itv.end)) out.add(rid);
        }
      }
      return out;
    };

    const rebuildEntityCounts = (active: Set<string>) => {
      const m = new Map<string, number>();
      for (const rid of active) {
        const p = relPairs.get(rid);
        if (!p) continue;
        m.set(p.s, (m.get(p.s) || 0) + 1);
        m.set(p.o, (m.get(p.o) || 0) + 1);
      }
      return m;
    };

    const animateKeyShape = (item: any, attrs: AnyDict) => {
      const shape = item?.getKeyShape?.();
      if (!shape?.animate) return;
      try {
        if (shape.stopAnimate) shape.stopAnimate();
      } catch {}
      try {
        shape.animate(attrs, { duration: tweenMs, easing: "easeCubic" });
      } catch {}
    };

    const getShapeNum = (item: any, key: string, fallback: number) => {
      try {
        const v = Number(item?.getKeyShape?.()?.attr?.(key));
        return Number.isFinite(v) ? v : fallback;
      } catch {
        return fallback;
      }
    };

    const updateRelNode = (rid: string, isActive: boolean, animate: boolean) => {
      const item = nodeItemByIdRef.current.get(rid);
      if (!item) return;
      const n = nodeById.get(rid) || null;
      const kind = String((n as any)?.relation_kind || "").trim().toLowerCase();
      const isEvent = kind === "event";
      const opacity = isActive ? 1 : 0.12;
      const fromOpacity = getShapeNum(item, "opacity", opacity);
      const fromLineWidth = getShapeNum(item, "lineWidth", isActive ? 1.5 : 1);
      
      const toSize = isActive ? 24 : 20; // Slight pulse size
      const fromSize = getShapeNum(item, "r", 10) * 2;

      item.update({
        size: toSize,
        zIndex: isEvent ? 20 : 10,
        style: {
          fill: isEvent ? rgba(C.REL_KIND_EVENT, theme === "dark" ? 0.10 : 0.08) : rgba(C.REL_KIND_STATE, theme === "dark" ? 0.08 : 0.06),
          stroke: isEvent
            ? rgba(C.REL_KIND_EVENT, isActive ? 0.85 : 0.28)
            : rgba(C.REL_KIND_STATE, isActive ? (theme === "dark" ? 0.78 : 0.70) : (theme === "dark" ? 0.26 : 0.22)),
          lineWidth: isEvent ? (isActive ? 2 : 1.25) : (isActive ? 1.6 : 1),
          opacity,
        },
      });
      if (animate) {
        try {
          item?.getKeyShape?.()?.attr?.("opacity", fromOpacity);
          item?.getKeyShape?.()?.attr?.("lineWidth", fromLineWidth);
          item?.getKeyShape?.()?.attr?.("r", fromSize / 2);
        } catch {}
        animateKeyShape(item, { opacity, lineWidth: isActive ? 1.5 : 1, r: toSize / 2 });
      }
    };

    const updateEntityNode = (eid: string, isVisible: boolean, animate: boolean) => {
      const item = nodeItemByIdRef.current.get(eid);
      if (!item) return;
      const opacity = isVisible ? 1 : 0.12;
      const fromOpacity = getShapeNum(item, "opacity", opacity);
      item.update({
        style: {
          fill: isVisible ? C.NODE_FILL : (theme === "dark" ? "rgba(79,166,216,0.18)" : "rgba(59,130,246,0.18)"),
          stroke: C.NODE_STROKE,
          opacity,
        },
        labelCfg: { style: { opacity: Math.max(0.15, opacity) } },
      });
      if (animate) {
        try {
          item?.getKeyShape?.()?.attr?.("opacity", fromOpacity);
        } catch {}
        animateKeyShape(item, { opacity });
      }
    };

    const updateRelEdges = (
      rid: string,
      isActive: boolean,
      relEnter: Set<string>,
      relExitSoon: Set<string>,
      animate: boolean,
    ) => {
      const edges = edgeItemsByRelRef.current.get(rid);
      if (!edges?.length) return;
      const n = nodeById.get(rid) || null;
      const kind = String((n as any)?.relation_kind || "").trim().toLowerCase();
      const isEvent = kind === "event";
      const cnt = n ? toEvCount(n) : 0;
      const baseColor = cnt >= 5 ? C.STRONG : cnt >= 2 ? C.MED : C.WEAK;
      const color = relEnter.has(rid) ? C.ENTER : relExitSoon.has(rid) ? C.EXIT : baseColor;
      const alpha = isActive ? Math.min(0.92, 0.22 + 0.14 * Math.min(5, Math.max(0, cnt))) : 0.06;
      const width = isActive ? 1 + Math.min(5, Math.floor(Math.sqrt(cnt + 1))) : 1;
      const dashed = Boolean(relExitSoon.has(rid));
      
      // Simulate "growth" via lineDashOffset animation if entering
      const isEntering = relEnter.has(rid);
      const baseDash = isEvent ? [2, 2] : undefined;
      const lineDash = dashed ? [6, 4] : isEntering ? [10, 10] : baseDash;

      for (const edge of edges) {
        const fromOpacity = animate ? getShapeNum(edge, "opacity", isActive ? 1 : 0.12) : 0;
        const fromLineWidth = animate ? getShapeNum(edge, "lineWidth", width) : 0;
        
        edge.update({
          style: {
            stroke: rgba(color, alpha),
            lineWidth: width,
            lineDash,
            endArrow: { path: G6.Arrow.triangle(6, 8, 2), d: 2, fill: rgba(color, Math.min(0.9, alpha + 0.08)) },
            opacity: isActive ? 1 : 0.12,
          },
        });
        
        if (animate) {
          try {
            edge?.getKeyShape?.()?.attr?.("opacity", fromOpacity);
            edge?.getKeyShape?.()?.attr?.("lineWidth", fromLineWidth);
          } catch {}
          const targetAttrs: any = { opacity: isActive ? 1 : 0.12, lineWidth: width };
          
          // Simple dash flow animation if entering
          if (isEntering) {
             try {
                const shape = edge.getKeyShape();
                shape.stopAnimate();
                shape.animate(
                  (ratio: number) => {
                    return { lineDashOffset: -20 * ratio };
                  },
                  { repeat: true, duration: 1000 }
                );
             } catch {}
          } else if (!dashed) {
             try { edge.getKeyShape().attr("lineDash", undefined); } catch {}
          }

          animateKeyShape(edge, targetAttrs);
        }
      }
    };

    const computeEnterExitSoon = () => {
      if (useDeltas && effectiveDeltas) {
        const cur = effectiveDeltas[frameIdxSafe] || {};
        const nxt = frameIdxSafe + 1 < effectiveDeltas.length ? effectiveDeltas[frameIdxSafe + 1] || {} : {};
        if (displayMode === "累积至当前") {
          return {
            enter: new Set<string>(Array.isArray(cur.enter) ? cur.enter.map(String) : []),
            exitSoon: new Set<string>(),
          };
        }
        return {
          enter: new Set<string>(Array.isArray(cur.enter) ? cur.enter.map(String) : []),
          exitSoon: new Set<string>(Array.isArray(nxt.exit) ? nxt.exit.map(String) : []),
        };
      }

      const prevT = frameIdxSafe > 0 ? frameTimes[frameIdxSafe - 1] : null;
      const nextT = frameIdxSafe + 1 < frameTimes.length ? frameTimes[frameIdxSafe + 1] : null;
      const enter = new Set<string>();
      const exitSoon = new Set<string>();
      if (prevT != null) {
        for (const [rid, itv] of relIntervals.entries()) {
          if (prevT < itv.start && itv.start <= timeNow) enter.add(rid);
        }
      }
      if (nextT != null) {
        for (const [rid, itv] of relIntervals.entries()) {
          if (itv.end != null && timeNow < itv.end && itv.end <= nextT) exitSoon.add(rid);
        }
      }
      return { enter, exitSoon };
    };

    const { enter: relEnter, exitSoon: relExitSoon } = computeEnterExitSoon();

    // If selected or hovered, force full update to apply focus logic (simpler than delta patching focus)
    const doFullUpdate = !useDeltas || displayModeChanged || !isSequential || isLoopToZero || prevIdx < 0 || selectedNodeId != null || hoverNodeId != null;

    if (doFullUpdate) {
      const active = computeActiveSet(timeNow);
      activeRelIdsRef.current = active;
      const counts = rebuildEntityCounts(active);
      entityCountsRef.current = counts;

      // Auto-focus camera logic
      if (autoFocus && active.size > 0 && graph) {
          let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
          let hasPoint = false;
          for (const rid of active) {
              const item = graph.findById(rid);
              if (item) {
                  const bbox = item.getBBox();
                  if (bbox) {
                      minX = Math.min(minX, bbox.minX);
                      minY = Math.min(minY, bbox.minY);
                      maxX = Math.max(maxX, bbox.maxX);
                      maxY = Math.max(maxY, bbox.maxY);
                      hasPoint = true;
                  }
              }
          }
          if (hasPoint) {
              const cx = (minX + maxX) / 2;
              const cy = (minY + maxY) / 2;
              graph.moveTo(cx, cy, true); // smooth move
          }
      }

      const visibleEntities = new Set<string>();
      for (const rid of active) {
        const p = relPairs.get(rid);
        if (p?.s) visibleEntities.add(p.s);
        if (p?.o) visibleEntities.add(p.o);
      }

      // Compute focus set if a node is selected or hovered
      let focusNodes: Set<string> | null = null;
      const targetId = hoverNodeId || selectedNodeId;
      if (targetId) {
        focusNodes = new Set<string>();
        focusNodes.add(targetId);
        // Add 1-hop neighbors from active relations
        for (const rid of active) {
           const p = relPairs.get(rid);
           if (!p) continue;
           if (p.s === targetId) { focusNodes.add(p.o); focusNodes.add(rid); }
           if (p.o === targetId) { focusNodes.add(p.s); focusNodes.add(rid); }
        }
      }

      for (const node of graph.getNodes()) {
        const m = node.getModel();
        const id = String(m?.id || "");
        const t = String(m?.type || "");
        const isActiveRel = t === "relation_state" && active.has(id);
        const isEntity = t === "entity";
        const isVisibleEntity = isEntity && visibleEntities.has(id);
        const kind = t === "relation_state" ? String((nodeById.get(id) as any)?.relation_kind || "").trim().toLowerCase() : "";
        const isEvent = kind === "event";
        
        let isFocused = true;
        if (focusNodes) {
            isFocused = focusNodes.has(id);
        }

        const opacity = (isActiveRel || isVisibleEntity) && isFocused ? 1 : 0.12;

        if (t === "relation_state") {
          node.update({
            zIndex: isEvent ? 20 : 10,
            style: {
              fill: isEvent ? rgba(C.REL_KIND_EVENT, theme === "dark" ? 0.10 : 0.08) : rgba(C.REL_KIND_STATE, theme === "dark" ? 0.08 : 0.06),
              stroke: isEvent
                ? rgba(C.REL_KIND_EVENT, isActiveRel ? 0.85 : 0.28)
                : rgba(C.REL_KIND_STATE, isActiveRel ? (theme === "dark" ? 0.78 : 0.70) : (theme === "dark" ? 0.26 : 0.22)),
              lineWidth: isEvent ? (isActiveRel ? 2 : 1.25) : (isActiveRel ? 1.6 : 1),
              opacity,
            },
          });
        } else {
          node.update({
            style: {
              fill: isVisibleEntity ? C.NODE_FILL : (theme === "dark" ? "rgba(79,166,216,0.18)" : "rgba(59,130,246,0.18)"),
              stroke: C.NODE_STROKE,
              opacity,
            },
            labelCfg: { style: { opacity: Math.max(0.15, opacity) } },
          });
        }
      }

      for (const edge of graph.getEdges()) {
        const m = edge.getModel();
        const s = String(m?.source || "");
        const t = String(m?.target || "");
        const rid = relIntervals.has(s) ? s : relIntervals.has(t) ? t : "";
        const isActive = rid ? active.has(rid) : false;
        
        let isFocused = true;
        if (focusNodes) {
             // Edge is focused if its relation node is focused
             isFocused = rid ? focusNodes.has(rid) : false;
        }

        const n = rid ? nodeById.get(rid) : null;
        const kind = String((n as any)?.relation_kind || "").trim().toLowerCase();
        const isEvent = kind === "event";
        const cnt = n ? toEvCount(n) : 0;

        const baseColor = cnt >= 5 ? C.STRONG : cnt >= 2 ? C.MED : C.WEAK;
        const color = rid && relEnter.has(rid) ? C.ENTER : rid && relExitSoon.has(rid) ? C.EXIT : baseColor;
        const alpha = isActive && isFocused ? Math.min(0.92, 0.22 + 0.14 * Math.min(5, Math.max(0, cnt))) : 0.06;
        const width = isActive ? 1 + Math.min(5, Math.floor(Math.sqrt(cnt + 1))) : 1;
        const dashed = Boolean(rid && relExitSoon.has(rid));
        const baseDash = isEvent ? [2, 2] : undefined;

        edge.update({
          style: {
            stroke: rgba(color, alpha),
            lineWidth: width,
            lineDash: dashed ? [6, 4] : baseDash,
            endArrow: { path: G6.Arrow.triangle(6, 8, 2), d: 2, fill: rgba(color, Math.min(0.9, alpha + 0.08)) },
            opacity: isActive && isFocused ? 1 : 0.12,
          },
        });
      }

      prevEnterRef.current = relEnter;
      prevExitSoonRef.current = relExitSoon;
      prevFrameIdxRef.current = frameIdxSafe;
      graph.paint();
      return;
    }

    const delta = effectiveDeltas?.[frameIdxSafe] || {};
    const active = activeRelIdsRef.current;
    const counts = entityCountsRef.current;

    const addEntity = (eid: string) => {
      counts.set(eid, (counts.get(eid) || 0) + 1);
    };
    const removeEntity = (eid: string) => {
      const v = (counts.get(eid) || 0) - 1;
      if (v > 0) counts.set(eid, v);
      else counts.delete(eid);
    };

    const entered = Array.isArray(delta.enter) ? delta.enter : [];
    const exited = displayMode === "累积至当前" ? [] : Array.isArray(delta.exit) ? delta.exit : [];

    for (const rid of entered) {
      if (active.has(rid)) continue;
      active.add(rid);
      const p = relPairs.get(rid);
      if (p?.s) addEntity(p.s);
      if (p?.o) addEntity(p.o);
    }
    for (const rid of exited) {
      if (!active.has(rid)) continue;
      active.delete(rid);
      const p = relPairs.get(rid);
      if (p?.s) removeEntity(p.s);
      if (p?.o) removeEntity(p.o);
    }

    const affectedRids = new Set<string>();
    for (const rid of entered) affectedRids.add(rid);
    for (const rid of exited) affectedRids.add(rid);
    for (const rid of prevEnterRef.current) affectedRids.add(rid);
    for (const rid of prevExitSoonRef.current) affectedRids.add(rid);
    for (const rid of relEnter) affectedRids.add(rid);
    for (const rid of relExitSoon) affectedRids.add(rid);

    const affectedEntities = new Set<string>();
    for (const rid of affectedRids) {
      const p = relPairs.get(rid);
      if (p?.s) affectedEntities.add(p.s);
      if (p?.o) affectedEntities.add(p.o);
    }

    for (const rid of affectedRids) {
      updateRelNode(rid, active.has(rid), true);
      updateRelEdges(rid, active.has(rid), relEnter, relExitSoon, true);
    }
    for (const eid of affectedEntities) {
      updateEntityNode(eid, (counts.get(eid) || 0) > 0, true);
    }

    prevEnterRef.current = relEnter;
    prevExitSoonRef.current = relExitSoon;
    prevFrameIdxRef.current = frameIdxSafe;
    graph.paint();
  }, [frameIdx, timeNow, displayMode, relIntervals, frameTimes, nodeById, relPairs, effectiveDeltas, selectedNodeId, hoverNodeId, autoFocus, theme]);

  useEffect(() => {
    if (!playing) return;
    if (!frameTimes.length) return;

    const tick = (ts: number) => {
      if (!playing) return;
      const last = lastTickRef.current || 0;
      if (ts - last >= Math.max(50, Math.trunc(speedMs))) {
        lastTickRef.current = ts;
        setFrameIdx((cur: number) => {
          const nxt = cur + 1;
          return nxt >= frameTimes.length ? 0 : nxt;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [playing, speedMs, frameTimes.length]);

  useEffect(() => {
    Streamlit.setFrameHeight(height + 4);
  }, [height]);

  const curText = useMemo(() => {
    if (!frames.length || timeNow == null) return "No frames";
    const d = new Date(timeNow);
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }, [frames.length, timeNow]);

  return (
    <div className="wrap">
      <div className="toolbar">
        <div
          className="btn"
          onClick={() => {
            const v = 0;
            sliderValRef.current = v;
            setFrameIdx(v);
            if (!playing) Streamlit.setComponentValue({ frame_idx: v });
          }}
          title="跳到第一帧"
        >
          ⏮
        </div>
        <div
          className="btn"
          onClick={() => {
            const v = Math.max(0, clampInt(frameIdxRef.current, 0, Math.max(0, frameTimes.length - 1)) - 1);
            sliderValRef.current = v;
            setFrameIdx(v);
            if (!playing) Streamlit.setComponentValue({ frame_idx: v });
          }}
          title="上一帧"
        >
          ◀
        </div>
        <div className="btn" onClick={() => setPlaying((v: boolean) => !v)} title={playing ? "暂停" : "播放"}>
          {playing ? "⏸" : "▶"}
        </div>
        <div
          className="btn"
          onClick={() => {
            const hi = Math.max(0, frameTimes.length - 1);
            const v = Math.min(hi, clampInt(frameIdxRef.current, 0, hi) + 1);
            sliderValRef.current = v;
            setFrameIdx(v);
            if (!playing) Streamlit.setComponentValue({ frame_idx: v });
          }}
          title="下一帧"
        >
          ▶
        </div>
        <input
          className="range"
          type="range"
          aria-label="时间轴"
          title="时间轴"
          min={0}
          max={Math.max(0, frames.length - 1)}
          value={clampInt(frameIdx, 0, Math.max(0, frames.length - 1))}
          onChange={(e: ChangeEvent<HTMLInputElement>) => {
            const v = Number(e.target.value);
            sliderValRef.current = v;
            setFrameIdx(v);
          }}
          onMouseUp={() => {
            if (!playing) {
              const hi = Math.max(0, frameTimes.length - 1);
              Streamlit.setComponentValue({ frame_idx: clampInt(sliderValRef.current, 0, hi) });
            }
          }}
          onTouchEnd={() => {
            if (!playing) {
              const hi = Math.max(0, frameTimes.length - 1);
              Streamlit.setComponentValue({ frame_idx: clampInt(sliderValRef.current, 0, hi) });
            }
          }}
        />
        <select
          className="btn"
          value={displayMode}
          onChange={(e: ChangeEvent<HTMLSelectElement>) => setDisplayMode(e.target.value)}
          aria-label="显示模式"
          title="显示模式"
        >
          <option value="当前激活">当前激活</option>
          <option value="累积至当前">累积至当前</option>
        </select>
        <div 
           className="btn" 
           onClick={() => setAutoFocus(v => !v)}
           title={autoFocus ? "关闭自动运镜" : "开启自动运镜"}
           style={{ opacity: autoFocus ? 1 : 0.5 }}
        >
           🎥
        </div>
        <select
          className="btn"
          value={speedMs}
          onChange={(e: ChangeEvent<HTMLSelectElement>) => setSpeedMs(Number(e.target.value))}
          aria-label="播放速度"
          title="播放速度"
        >
          <option value={250}>0.25s</option>
          <option value={420}>0.42s</option>
          <option value={650}>0.65s</option>
          <option value={900}>0.9s</option>
        </select>
        <div className="meta">{curText}</div>
      </div>
      <div className="canvas">
        <div className="canvas-inner" ref={containerRef} />
      </div>
    </div>
  );
}

export default withStreamlitConnection(EvolutionGraphBase);
