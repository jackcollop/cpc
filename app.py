r"""Dash app: browse CPC condition / progress GeoTIFFs and plot a single week
or the change between two weeks, interactively in Plotly.

Concept taken from tif.ipynb / brz/tif.py:
  - single-band float32 GeoTIFFs, nodata = -9999
  - files named like  cottonCond26w28.tif  (…w<week>.tif)
  - US state + county borders (lon/lat GeoJSON) reprojected onto the raster CRS
  - single week  -> sequential RdYlGn (red = poor, green = good)
  - change (B-A) -> diverging RdBu centred on 0

Data layout (all under one root):
  <root>/<commodity>/cpc<commodity>2026.zip
      condition/<commodity>Cond26w<week>.tif
      progress/<commodity>Prog26w<week>.tif

  e.g.  cpc2026/cotton/cpccotton2026.zip  ->  condition/cottonCond26w28.tif

Run:  python app.py   then open http://127.0.0.1:8050
"""

import os
import re
import glob
import json
import zipfile
from functools import lru_cache

import numpy as np
import rasterio
from pyproj import Transformer
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, no_update

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# All paths are resolved relative to this script so nothing is machine-specific.
HERE = os.path.dirname(os.path.abspath(__file__))
# Root that holds one subfolder per commodity, each containing a .zip.
DEFAULT_BASE = os.path.join(HERE, "cpc2026/cpc2026")
STATES_GEOJSON = os.path.join(HERE, "us_states.geojson")
COUNTIES_GEOJSON = os.path.join(HERE, "counties-fips.json")

NODATA = -9999.0
MAX_AXIS = 700          # downsample so neither raster axis exceeds this many cells
WEEK_RE = re.compile(r"w(\d+)\.tif$", re.IGNORECASE)

STATE_FIPS = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa", "20": "Kansas",
    "21": "Kentucky", "22": "Louisiana", "23": "Maine", "24": "Maryland",
    "25": "Massachusetts", "26": "Michigan", "27": "Minnesota", "28": "Mississippi",
    "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma",
    "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina",
    "46": "South Dakota", "47": "Tennessee", "48": "Texas", "49": "Utah",
    "50": "Vermont", "51": "Virginia", "53": "Washington", "54": "West Virginia",
    "55": "Wisconsin", "56": "Wyoming", "72": "Puerto Rico",
}
NAME_TO_FIPS = {v: k for k, v in STATE_FIPS.items()}
DATASET_LABELS = {"condition": "Condition", "progress": "Progress"}


# --------------------------------------------------------------------------- #
# Data discovery (commodity -> zip -> dataset -> week)
# --------------------------------------------------------------------------- #
def list_commodities(base):
    """{commodity_name -> zip_path} for each subfolder of base holding a .zip."""
    out = {}
    if not base or not os.path.isdir(base):
        return out
    for entry in sorted(os.listdir(base)):
        sub = os.path.join(base, entry)
        if os.path.isdir(sub):
            zips = glob.glob(os.path.join(sub, "*.zip"))
            if zips:
                out[entry] = zips[0]
    return out


@lru_cache(maxsize=32)
def _zip_tifs(zip_path):
    """List of .tif member paths inside the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        return [n for n in zf.namelist() if n.lower().endswith(".tif")]


def list_datasets(zip_path):
    """Top-level folders inside the zip that contain tifs (condition, progress)."""
    ds = set()
    for n in _zip_tifs(zip_path):
        parts = n.split("/")
        if len(parts) > 1:
            ds.add(parts[0])
    return sorted(ds)


def list_weeks(zip_path, dataset):
    """{week:int -> inner tif path} for one dataset folder inside the zip."""
    weeks = {}
    prefix = dataset + "/"
    for n in _zip_tifs(zip_path):
        if n.startswith(prefix):
            m = WEEK_RE.search(os.path.basename(n))
            if m:
                weeks[int(m.group(1))] = n
    return dict(sorted(weeks.items()))


# --------------------------------------------------------------------------- #
# Raster + borders (cached — reading zips / reprojecting 3k counties is slow)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def load_raster(zip_path, inner):
    """(masked float32 array, bounds, crs_wkt) for a GeoTIFF stored in a zip."""
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(inner)
    with rasterio.MemoryFile(data) as mf, mf.open() as ds:
        arr = ds.read(1).astype(np.float32)
        bounds = tuple(ds.bounds)            # (left, bottom, right, top)
        crs_wkt = ds.crs.to_wkt()
    return np.ma.masked_equal(arr, NODATA), bounds, crs_wkt


@lru_cache(maxsize=64)
def border_coords(geojson_path, crs_wkt, state_fips):
    """Reproject GeoJSON polygon outlines into the raster CRS.

    Returns (xs, ys, text, bbox): xs/ys are flat lists with None gaps between
    rings, text is the per-vertex hover label, bbox is (xmin,ymin,xmax,ymax) of
    the drawn features. Cached per (file, crs, state filter) so redraw is fast."""
    to_raster = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)
    with open(geojson_path, encoding="utf-8") as fh:
        gj = json.load(fh)

    xs, ys, text = [], [], []
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for feat in gj["features"]:
        props = feat["properties"]
        # county file has a STATE fips; state file has a name -> map to fips
        feat_fips = props.get("STATE") or NAME_TO_FIPS.get(props.get("name", ""))
        if state_fips and feat_fips != state_fips:
            continue
        name = props.get("NAME") or props.get("name", "")
        lsad = props.get("LSAD", "")
        label = f"{name} {lsad}".strip()
        geom = feat["geometry"]
        rings = geom["coordinates"] if geom["type"] == "Polygon" else \
            [ring for poly in geom["coordinates"] for ring in poly]
        for ring in rings:
            lon, lat = zip(*[(p[0], p[1]) for p in ring])
            x, y = to_raster.transform(lon, lat)
            xs.extend([*x, None])
            ys.extend([*y, None])
            text.extend([label] * len(x) + [None])
            xmin, xmax = min(xmin, min(x)), max(xmax, max(x))
            ymin, ymax = min(ymin, min(y)), max(ymax, max(y))
    bbox = None if xmin == np.inf else (xmin, ymin, xmax, ymax)
    return xs, ys, text, bbox


def border_trace(geojson_path, crs_wkt, state_fips=None, lw=0.5,
                 label=False, name="borders"):
    xs, ys, text, bbox = border_coords(geojson_path, crs_wkt, state_fips or None)
    hover = (dict(text=text, hovertemplate="%{text}<extra></extra>")
             if label else dict(hoverinfo="skip"))
    trace = go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color="black", width=lw),
        showlegend=False, name=name, **hover,
    )
    return trace, bbox


def downsample(z, x, y):
    """Stride the raster down so neither axis exceeds MAX_AXIS cells."""
    ry = max(1, int(np.ceil(z.shape[0] / MAX_AXIS)))
    rx = max(1, int(np.ceil(z.shape[1] / MAX_AXIS)))
    return z[::ry, ::rx], x[::rx], y[::ry]


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(zip_path, dataset, week, compare_week, mode, region, show_counties):
    weeks = list_weeks(zip_path, dataset)
    if week not in weeks:
        return go.Figure().update_layout(
            title="Select a week", plot_bgcolor="white", height=720)

    commodity = os.path.basename(os.path.dirname(zip_path)).title()
    dlabel = DATASET_LABELS.get(dataset, dataset.title())

    b, bounds, crs_wkt = load_raster(zip_path, weeks[week])
    left, bottom, right, top = bounds

    is_change = mode == "change" and compare_week in weeks and compare_week != week
    if is_change:
        a, _, _ = load_raster(zip_path, weeks[compare_week])
        z_masked = np.ma.masked_where(a.mask | b.mask, b - a)
        valid = z_masked.compressed()
        lim = float(np.percentile(np.abs(valid), 98)) if valid.size else 1.0
        lim = lim or 1.0
        colorscale, zmid, zmin, zmax = "RdBu", 0, -lim, lim
        cbar_title = f"Δ {dlabel.lower()}<br>(w{week} − w{compare_week})"
        title = f"{commodity} {dlabel}: Δ  w{compare_week} → w{week}"
        htmpl = "x=%{x:.0f}<br>y=%{y:.0f}<br>Δ=%{z:.2f}<extra></extra>"
    else:
        z_masked = b
        valid = z_masked.compressed()
        zmin = float(np.percentile(valid, 2)) if valid.size else 0.0
        zmax = float(np.percentile(valid, 98)) if valid.size else 1.0
        colorscale, zmid, cbar_title = "RdYlGn", None, f"{dlabel.lower()} index"
        title = f"{commodity} {dlabel}: week {week}"
        htmpl = "x=%{x:.0f}<br>y=%{y:.0f}<br>value=%{z:.2f}<extra></extra>"

    rows, cols = z_masked.shape
    x_coords = np.linspace(left, right, cols)
    y_coords = np.linspace(top, bottom, rows)   # row 0 = north edge
    z = z_masked.filled(np.nan)
    z, x_coords, y_coords = downsample(z, x_coords, y_coords)

    heat = go.Heatmap(
        z=z, x=x_coords, y=y_coords,
        colorscale=colorscale, zmid=zmid, zmin=zmin, zmax=zmax,
        colorbar=dict(title=cbar_title), hovertemplate=htmpl,
    )

    state_fips = NAME_TO_FIPS.get(region) if region and region != "CONUS" else None
    data = [heat]
    if show_counties:
        c_trace, _ = border_trace(COUNTIES_GEOJSON, crs_wkt, state_fips,
                                  lw=0.25, label=True, name="counties")
        data.append(c_trace)
    s_trace, s_bbox = border_trace(STATES_GEOJSON, crs_wkt, state_fips,
                                   lw=0.6, name="states")
    data.append(s_trace)

    # Zoom: to the selected state's bbox, else the raster extent.
    if state_fips:
        bbox = border_coords(COUNTIES_GEOJSON, crs_wkt, state_fips)[3] or s_bbox
        x_range, y_range = [bbox[0], bbox[2]], [bbox[1], bbox[3]]
    else:
        x_range, y_range = [left, right], [bottom, top]

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=30, r=30, t=60, b=30),
    )
    fig.update_xaxes(range=x_range, showgrid=False, zeroline=False)
    fig.update_yaxes(range=y_range, showgrid=False, zeroline=False,
                     scaleanchor="x", scaleratio=1)
    return fig


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
app = Dash(__name__)
app.title = "CPC condition / progress viewer"

_ctrl = {"marginBottom": "14px"}
_label = {"fontWeight": 600, "fontSize": "13px", "display": "block",
          "marginBottom": "4px"}

app.layout = html.Div(style={"display": "flex", "fontFamily": "system-ui, sans-serif",
                             "height": "100vh"}, children=[
    # ---- control panel ---------------------------------------------------- #
    html.Div(style={"width": "300px", "padding": "18px", "background": "#f5f6f8",
                    "borderRight": "1px solid #ddd", "overflowY": "auto"}, children=[
        html.H2("CPC viewer", style={"marginTop": 0, "fontSize": "20px"}),

        html.Div(style=_ctrl, children=[
            html.Label("Data root", style=_label),
            dcc.Input(id="base-dir", type="text", value=DEFAULT_BASE,
                      debounce=True, style={"width": "100%"}),
        ]),
        html.Div(style=_ctrl, children=[
            html.Label("Commodity", style=_label),
            dcc.Dropdown(id="commodity", clearable=False),
        ]),
        html.Div(style=_ctrl, children=[
            html.Label("Dataset", style=_label),
            dcc.Dropdown(id="dataset", clearable=False),
        ]),
        html.Div(style=_ctrl, children=[
            html.Label("Mode", style=_label),
            dcc.RadioItems(
                id="mode",
                options=[{"label": " Single week", "value": "single"},
                         {"label": " Change vs another week", "value": "change"}],
                value="single", labelStyle={"display": "block"}),
        ]),
        html.Div(style=_ctrl, children=[
            html.Label("Week", style=_label),
            dcc.Dropdown(id="week", clearable=False),
        ]),
        html.Div(id="compare-wrap", style=_ctrl, children=[
            html.Label("Compare to week", style=_label),
            dcc.Dropdown(id="compare-week", clearable=False),
        ]),
        html.Div(style=_ctrl, children=[
            html.Label("Region", style=_label),
            dcc.Dropdown(
                id="region", clearable=False, value="CONUS",
                options=[{"label": "CONUS (all)", "value": "CONUS"}] +
                        [{"label": n, "value": n} for n in sorted(NAME_TO_FIPS)]),
        ]),
        html.Div(style=_ctrl, children=[
            dcc.Checklist(
                id="show-counties",
                options=[{"label": " County borders (hoverable)", "value": "yes"}],
                value=[]),
        ]),
        html.P("Change mode: B − A on cells valid in both weeks, diverging RdBu "
               "centred on 0. Single week: RdYlGn, scaled 2–98th pct.",
               style={"fontSize": "11px", "color": "#666"}),
    ]),

    # ---- graph ------------------------------------------------------------ #
    html.Div(style={"flex": 1, "padding": "10px"}, children=[
        dcc.Loading(dcc.Graph(id="graph", style={"height": "95vh"}),
                    type="default"),
    ]),
])


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@app.callback(
    Output("commodity", "options"),
    Output("commodity", "value"),
    Input("base-dir", "value"),
)
def _commodities(base):
    coms = list_commodities(base)
    if not coms:
        return [], None
    opts = [{"label": k.title(), "value": v} for k, v in coms.items()]
    return opts, opts[0]["value"]


@app.callback(
    Output("dataset", "options"),
    Output("dataset", "value"),
    Input("commodity", "value"),
)
def _datasets(zip_path):
    if not zip_path:
        return [], None
    ds = list_datasets(zip_path)
    opts = [{"label": DATASET_LABELS.get(d, d.title()), "value": d} for d in ds]
    default = "condition" if "condition" in ds else (ds[0] if ds else None)
    return opts, default


@app.callback(
    Output("week", "options"),
    Output("week", "value"),
    Output("compare-week", "options"),
    Output("compare-week", "value"),
    Input("commodity", "value"),
    Input("dataset", "value"),
)
def _weeks(zip_path, dataset):
    if not zip_path or not dataset:
        return [], None, [], None
    weeks = list_weeks(zip_path, dataset)
    if not weeks:
        return [], None, [], None
    opts = [{"label": f"week {w}", "value": w} for w in weeks]
    ws = list(weeks)
    latest = ws[-1]
    prev = ws[-2] if len(ws) > 1 else ws[-1]
    return opts, latest, opts, prev


@app.callback(
    Output("compare-wrap", "style"),
    Input("mode", "value"),
)
def _toggle_compare(mode):
    style = dict(_ctrl)
    style["display"] = "block" if mode == "change" else "none"
    return style


@app.callback(
    Output("graph", "figure"),
    Input("commodity", "value"),
    Input("dataset", "value"),
    Input("week", "value"),
    Input("compare-week", "value"),
    Input("mode", "value"),
    Input("region", "value"),
    Input("show-counties", "value"),
)
def _plot(zip_path, dataset, week, compare_week, mode, region, show_counties):
    if not zip_path or not dataset or week is None:
        return no_update
    return build_figure(zip_path, dataset, week, compare_week, mode, region,
                        bool(show_counties))


if __name__ == "__main__":
    app.run(debug=True)
