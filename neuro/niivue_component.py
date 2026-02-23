"""
NiiVue viewer component for Streamlit.

Generates a self-contained HTML page with NiiVue (loaded from CDN) that
renders NIfTI volumes in an orthographic multi-plane view.  Embedded via
``st.components.v1.html()`` (srcdoc iframe).

NIfTI files are served via Streamlit's built-in static file server
(see ``file_server.py``).  Files are symlinked with a ``.bin`` extension
to prevent Tornado from transparently decompressing ``.nii.gz``.  The
JS resolves the app origin at runtime from ``document.referrer`` so URLs
work through SSH tunnels.
"""

# Custom colormaps registered with NiiVue via addColormap().
# Each entry: {R, G, B, A, I} arrays (values 0-255).
# I = intensity control points; NiiVue interpolates between them.
CUSTOM_COLORMAPS = {
    # Nilearn-style cold_hot: cyan→blue→[transparent black]→red→yellow
    # Alpha=0 at center so zero-valued voxels are transparent in overlays.
    'cold_hot': {
        'R': [0,   0,   0,   0,   0,   0,   255, 255],
        'G': [255, 255, 0,   0,   0,   0,   0,   255],
        'B': [255, 255, 255, 0,   0,   0,   0,   0],
        'A': [0,   255, 255, 0,   255, 255, 255, 255],
        'I': [0,   1,   64,  124, 128, 131, 192, 255],
    },
}

# CSS linear-gradient approximations of NiiVue colormaps.
# Used to render preview strips in the layer stack UI.
COLORMAP_CSS = {
    'gray':           'linear-gradient(to right, #000, #fff)',
    'red':            'linear-gradient(to right, #000, #f00)',
    'green':          'linear-gradient(to right, #000, #0f0)',
    'blue':           'linear-gradient(to right, #000, #00f)',
    'warm':           'linear-gradient(to right, #000, #a00, #f80, #ff0, #fff)',
    'electric_blue':  'linear-gradient(to right, #000, #00f, #0ff)',
    'cool':           'linear-gradient(to right, #00f, #0ff)',
    'plasma':         'linear-gradient(to right, #0d0887, #7e03a8, #cc4778, #f89540, #f0f921)',
    'viridis':        'linear-gradient(to right, #440154, #31688e, #35b779, #fde725)',
    'inferno':        'linear-gradient(to right, #000004, #420a68, #932667, #dd513a, #fca50a, #fcffa4)',
    'winter':         'linear-gradient(to right, #00f, #0ff, #0f0)',
    'hot':            'linear-gradient(to right, #000, #f00, #ff0, #fff)',
    'freesurfer':     'linear-gradient(to right, #000, #0af, #0f0, #ff0, #f00, #f0f)',
    'thermal':        'linear-gradient(to right, #000, #40f, #f0f, #f80, #ff0, #fff)',
    'blue2red':       'linear-gradient(to right, #00f, #fff, #f00)',
    'cold_hot':       'linear-gradient(to right, #0ff, #00f, #000, #f00, #ff0)',
}


def colormap_css(name, invert=False):
    """Return a CSS gradient string for *name*, optionally reversed."""
    css = COLORMAP_CSS.get(name, COLORMAP_CSS['gray'])
    if invert:
        css = css.replace('to right', 'to left')
    return css


def build_niivue_html(volumes, height=700, viewer_id='default'):
    """
    Build a self-contained HTML string that renders NIfTI volumes with NiiVue.

    Parameters
    ----------
    volumes : list of dict
        Each dict must contain:
            - ``path`` (str): relative URL path to the NIfTI file
              (e.g. '/_app/static/nifti/abc123.bin').
            - ``name`` (str): filename hint (e.g. 'T1w.nii.gz') so NiiVue
              knows the format.
            - ``colormap`` (str): NiiVue colormap name (e.g. 'gray', 'red').
            - ``opacity`` (float): Opacity 0-1.
        The first volume is treated as the background.
    height : int, optional
        Canvas height in pixels.  Default 700.
    viewer_id : str, optional
        Unique ID for this viewer instance.  Used to persist crosshair
        position in localStorage so it survives overlay changes.

    Returns
    -------
    str
        Complete HTML document string.
    """
    if not volumes:
        return '<html><body style="background:#0E1117;color:#888;">No images selected.</body></html>'

    # Build a JS array of {path, name, colormap, opacity, cal_min, cal_max} objects.
    volume_entries = []
    for v in volumes:
        cal_min = v.get('cal_min', '')
        cal_max = v.get('cal_max', '')
        entry = (
            '{'
            f' path: "{v["path"]}",'
            f' name: "{v["name"]}",'
            f' colormap: "{v["colormap"]}",'
            f' opacity: {v["opacity"]}'
        )
        if cal_min != '':
            entry += f', cal_min: {cal_min}'
        if cal_max != '':
            entry += f', cal_max: {cal_max}'
        if v.get('colormap_invert'):
            entry += ', colormapInvert: true'
        entry += ' }'
        volume_entries.append(entry)
    volume_list_js = ',\n          '.join(volume_entries)

    # Build JS for registering custom colormaps
    import json
    custom_cmap_js_lines = []
    for name, cmap in CUSTOM_COLORMAPS.items():
        custom_cmap_js_lines.append(
            f'nv.addColormap("{name}", {json.dumps(cmap)});'
        )
    custom_cmap_js = '\n        '.join(custom_cmap_js_lines)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0E1117; overflow: hidden; }}
    #viewer-wrap {{
      position: relative;
      width: 100%;
      height: {height}px;
    }}
    #gl {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    #status {{
      position: absolute;
      top: 10px;
      left: 10px;
      color: #ccc;
      font-family: monospace;
      font-size: 12px;
      max-width: 95%;
      word-wrap: break-word;
      white-space: pre-wrap;
      z-index: 10;
      background: rgba(0,0,0,0.7);
      padding: 8px;
      border-radius: 4px;
    }}
    #crosshair-info {{
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      color: #ddd;
      font-family: monospace;
      font-size: 11px;
      background: rgba(0,0,0,0.75);
      padding: 4px 10px;
      z-index: 10;
      pointer-events: none;
    }}
  </style>
</head>
<body>
  <div id="viewer-wrap">
    <div id="status">Starting diagnostics...</div>
    <canvas id="gl" width="800" height="{height}"></canvas>
    <div id="crosshair-info">Click on image to see crosshair info</div>
  </div>

  <script>
    const VIEWER_ID = "{viewer_id}";
    const XHAIR_KEY = "pqc_xhair_" + VIEWER_ID;
    const statusEl = document.getElementById("status");
    function log(msg) {{
      statusEl.textContent += "\\n" + msg;
      console.log("[niivue-diag]", msg);
    }}

    (async function() {{
      try {{
        // 1. Resolve the Streamlit app origin.
        let appOrigin = "";
        try {{ appOrigin = new URL(document.referrer).origin; }} catch(e) {{}}
        if (!appOrigin) appOrigin = window.location.origin;
        if (!appOrigin || appOrigin === "null") appOrigin = "http://localhost:8501";
        log("App origin: " + appOrigin);

        // 2. Check WebGL2
        const testCanvas = document.createElement("canvas");
        const glTest = testCanvas.getContext("webgl2");
        if (!glTest) {{
          log("FAIL: WebGL2 not supported by this browser");
          return;
        }}
        log("OK: WebGL2 supported");

        // 3. Build volume list with absolute URLs
        const rawVolumes = [
            {volume_list_js}
        ];
        const volumes = rawVolumes.map(v => {{
          const vol = {{
            url: appOrigin + v.path,
            name: v.name,
            colormap: v.colormap,
            opacity: v.opacity,
          }};
          if (v.cal_min !== undefined) vol.cal_min = v.cal_min;
          if (v.cal_max !== undefined) vol.cal_max = v.cal_max;
          if (v.colormapInvert) vol.colormapInvert = true;
          return vol;
        }});

        // 4. Test that the static file server can reach the files
        for (const vi of volumes) {{
          try {{
            const resp = await fetch(vi.url, {{ method: "HEAD" }});
            log("OK: " + vi.name + " -> " + resp.status + " (" + (resp.headers.get("content-length") || "?") + " bytes)");
            if (!resp.ok) {{
              log("FAIL: " + vi.name + " returned HTTP " + resp.status);
            }}
          }} catch (fetchErr) {{
            log("FAIL: " + vi.name + " fetch error -> " + fetchErr.message);
          }}
        }}

        // 5. Import NiiVue from CDN
        log("Importing NiiVue from CDN...");
        let Niivue;
        try {{
          const mod = await import("https://unpkg.com/@niivue/niivue@0.67.0/dist/index.js");
          Niivue = mod.Niivue;
          log("OK: NiiVue imported");
        }} catch (importErr) {{
          log("FAIL: CDN import error -> " + importErr.message);
          return;
        }}

        // 6. Create NiiVue instance and attach to canvas
        log("Loading volumes...");
        const canvas = document.getElementById("gl");
        const infoEl = document.getElementById("crosshair-info");

        const nv = new Niivue({{
          backColor: [0.055, 0.067, 0.09, 1],
        }});

        // 7. Set crosshair callback BEFORE loading volumes
        nv.onLocationChange = function(data) {{
          try {{
            if (!data) return;
            let parts = [];
            if (data.vox) {{
              const vx = data.vox;
              parts.push("Voxel: [" + vx[0] + ", " + vx[1] + ", " + vx[2] + "]");
            }}
            if (data.mm) {{
              const m = data.mm;
              parts.push("mm: [" + m[0].toFixed(1) + ", " + m[1].toFixed(1) + ", " + m[2].toFixed(1) + "]");
            }}
            // Extract voxel values from data.values
            if (data.values && data.values.length > 0) {{
              for (let i = 0; i < data.values.length; i++) {{
                const vname = (nv.volumes[i] && nv.volumes[i].name) || ("Vol " + i);
                const vlabel = vname.replace(".nii.gz", "");
                let raw = data.values[i];
                // NiiVue may return objects — extract the numeric value
                if (raw !== null && typeof raw === "object") {{
                  // Try common property names for the numeric value
                  if (typeof raw.value === "number") raw = raw.value;
                  else if (typeof raw.val === "number") raw = raw.val;
                  else {{
                    // Grab the first numeric property
                    const nums = Object.values(raw).filter(v => typeof v === "number");
                    if (nums.length > 0) raw = nums[0];
                    else raw = JSON.stringify(raw);
                  }}
                }}
                if (typeof raw === "number") {{
                  parts.push(vlabel + ": " + raw.toFixed(4));
                }} else {{
                  parts.push(vlabel + ": " + String(raw));
                }}
              }}
            }}
            if (parts.length > 0) {{
              infoEl.textContent = parts.join("    ");
            }}
            // Persist crosshair mm position so it survives overlay changes
            try {{
              if (data.mm) {{
                localStorage.setItem(XHAIR_KEY, JSON.stringify({{
                  mm: [data.mm[0], data.mm[1], data.mm[2]]
                }}));
              }}
            }} catch(e) {{
              console.warn("[crosshair] localStorage save failed:", e);
            }}
          }} catch(e) {{
            console.error("[crosshair]", e);
          }}
        }};

        await nv.attachToCanvas(canvas);

        // 8. Register custom colormaps before loading volumes
        {custom_cmap_js}

        await nv.loadVolumes(volumes);

        // 9. Restore crosshair position from previous session (mm coords)
        try {{
          const raw = localStorage.getItem(XHAIR_KEY);
          if (raw) {{
            const saved = JSON.parse(raw);
            if (saved.mm && nv.volumes.length > 0) {{
              // Convert mm world coords → fractional [0..1] via NiiVue
              if (typeof nv.mm2frac === "function") {{
                const frac = nv.mm2frac(saved.mm);
                if (frac) {{
                  nv.scene.crosshairPos = frac;
                  nv.drawScene();
                  console.log("[crosshair] restored via mm2frac:", saved.mm);
                }}
              }} else {{
                console.warn("[crosshair] nv.mm2frac not available");
              }}
            }}
          }}
        }} catch(e) {{
          console.warn("[crosshair] restore failed:", e);
        }}

        // Success — hide diagnostics
        statusEl.style.display = "none";

      }} catch (err) {{
        log("UNCAUGHT ERROR: " + err.message);
        console.error("[niivue-diag] uncaught:", err);
      }}
    }})();
  </script>
</body>
</html>"""
    return html
