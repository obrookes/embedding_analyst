import os
import gc
import sys
import panel as pn
import numpy as np
import pandas as pd
from json import load
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CustomJS
from bokeh.layouts import layout
import matplotlib.pyplot as plt

src_csv = sys.argv[1]
df = pd.read_csv(src_csv)

# Reduced components
x1 = df.x1.values
x2 = df.x2.values

labels = df.labels.values
logits = df.logits.values

misclassifications = [1.0 if x != y else 0.0 for x, y in zip(labels, logits)]
confidences = df.confidences.values

data = df.filenames.values
prefix = "/home/dl18206/Desktop/phd/data/panaf/acp/videos/all/"
data = [os.path.join(prefix, x.split("/")[-1].split(".")[0] + ".mp4") for x in data]

colors_fill = {
    "0.0": "rgb(0, 0, 255)",  # unlabeled
    "1.0": "rgb(255, 0, 0)",  # prev_chosen
}

if sys.argv[2] == "labels":
    fill_by = labels
elif sys.argv[2] == "errors":
    fill_by = misclassifications

source = ColumnDataSource(
    data=dict(
        x=x1,
        y=x2,
        color=[colors_fill[f"{x}"] for x in fill_by],
    )
)

embed = pn.Row(sizing_mode="stretch_both")

image = pn.layout.Row()
plot = figure(title=f"Coloured by {sys.argv[2]}", width=750, height=750)
plot.circle(
    "x",
    "y",
    size=5,
    fill_color="color",
    line_color="color",
    source=source,
    legend_label="videos",
)
plots = pn.pane.Bokeh(plot)

embed.objects = [pn.layout.Row(plots, pn.Column(image))]

s1 = ColumnDataSource(data=dict(idx=[0]))

callback = CustomJS(
    args=dict(s1=s1),
    code="""
                        var current_idx = s1.data.idx[0];
                        if (cb_data['index'].indices.length > 0){
                            if(cb_data['index'].indices[0] != current_idx){
                            
                            var dict = {
                                idx: cb_data['index'].indices,
                                };


                            s1.data = dict;
                            }
                        }
                        """,
)

hover_tool_plot = HoverTool(callback=callback)


def decode_idx(value):
    labels = ["no_camera_reaction", "camera_reaction"]
    return labels[int(value)]


def get_values(attr, old, new):
    idx = s1.data["idx"][0]
    img = data[idx]
    err = misclassifications[idx]
    label = labels[idx]
    pred = logits[idx]
    conf = confidences[idx]

    image.objects = [
        pn.layout.Column(
            pn.pane.Video(
                img,
                width=640,
                loop=True,
                autoplay=True,
            ),
            pn.pane.Str(f"Video: {img.split('/')[-1]}"),
            pn.pane.Str(f"Error: {True if err==1.0 else False}"),
            pn.pane.Str(f"Label: {decode_idx(label)}"),
            pn.pane.Str(f"Prediction: {decode_idx(pred)} ({round(conf*100, 2)}%)"),
        )
    ]


s1.on_change("data", get_values)
plot.add_tools(hover_tool_plot)

template = pn.template.FastGridTemplate(
    site="Panel", title="App", prevent_collision=True
)

template.main[0:12, 1:12] = embed
template.servable()

# panel serve embed_view_reduced.py --show --autoreload --num-threads 4 --args data/r50_cns_binary_val.csv errors
