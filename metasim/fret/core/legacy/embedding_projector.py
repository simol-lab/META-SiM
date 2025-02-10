"""Library for the embedding projector."""
import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sciplotlib import style as spstyle
from smfret.dataset import FRETTraceSet

def trim(x, precision=4):
    """Trims a float to a given precision."""
    if int(x) == x:
        return x
    else:
        return float(np.format_float_positional(x, precision=4, unique=False, fractional=False, trim='k'))


def generate_sprite_plot(log_dir, trace_set):
    """Generates the .png file for sprite image of traces."""
    image_files = []
    with plt.style.context(spstyle.get_style('nature-reviews')):
        for idx in tqdm(range(trace_set.size)):
            fig = plt.figure(figsize=(2, 0.7))
            ax = fig.add_axes([0.2, 0.17, 0.68, 0.7])
            ax.plot(trace_set.traces[idx].acceptor, linewidth=0.1)
            ax.plot(trace_set.traces[idx].donor, linewidth=0.1)
            ax.set_xticks([])
            ax.set_yticks([])
            image_file = f'/tmp/trace_{idx}.png'
            image_files.append(image_file)
            fig.savefig(image_file, dpi=200, pad_inches=0, bbox_inches='tight', transparent=True)
            plt.close()

    
    images = []
    for f in image_files:
        e = Image.open(f)
        m = e.copy()
        e.close()
        images.append(m)
    width, height = images[0].size
    
    cols = int(np.sqrt(len(images))) + 1
    rows = cols
    
    total_width = cols * width
    total_height = rows * height
    
    new_im = Image.new('RGBA', (total_width, total_height))
    
    x_offset = 0
    for idx, im in enumerate(images):
      col = idx % cols
      row = idx // cols
      new_im.paste(im, (col * width, row * height))
    
    new_im.save(os.path.join(log_dir, 'trace_sprite.png'))


def generate_projector_data(log_dir, embedding, trace_set, plot_label, sprite=False):
    """Generates tensor and metadata for the embedding projector."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if sprite:
        generate_sprite_plot(log_dir, trace_set);
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
      f.write('name\tground truth\tdonor\tacceptor\n')
      for index in range(embedding.shape[0]):
        donor = np.array([trim(x) for x in trace_set.traces[index].donor])
        acceptor = np.array([trim(x) for x in trace_set.traces[index].acceptor])
        f.write(f"trace {index}\t{plot_label[index]}\t{json.dumps(donor.tolist())}\t{json.dumps(acceptor.tolist())}\n")
    with open(log_dir + 'tensor.bytes', 'wb') as f:
        f.write(np.array(tf.cast(embedding, tf.float32)).tobytes())
    np.savetxt(log_dir + 'tensor.tsv', embedding, delimiter='\t')


def generate_tensor_and_metadata(trace_sets, start_frame, output_dir, encoder, max_frame=2000, max_traces=4000):
    """Generates the files for embedding projector."""

    embeddings = []
    labels = []
    colors = []
    color_count = 1
    image_files = []
    traces = []
    
    count = 0
    for trace_set in trace_sets:
        trim_size = len(trace_set.time) // 1000 * 1000
        trace_set.trim(trim_size)
        trace_set.trim(max_frame, start_frame=start_frame)
        trace_set.broadcast_data_to_traces()
        print(trace_set.size)
        if count + trace_set.size > max_traces:
            break
        else:
            count += trace_set.size
                
        with tf.device('/CPU:0'):
            embedding = encoder.predict(trace_set.to_tensor())
            embeddings.append(embedding)
            labels.append(np.max(trace_set.label, axis=-1))
            traces.extend(trace_set.traces)
    
    trace_set = FRETTraceSet()
    trace_set.traces = traces
    trace_set.size = len(traces)
    embedding = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    generate_projector_data(output_dir, embedding, trace_set, labels, sprite=False)