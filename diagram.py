import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_layer(ax, layer_name, size, position, color='lightblue'):
    rect = Rectangle(position, size[0], size[1], edgecolor='black', facecolor=color, lw=2)
    ax.add_patch(rect)
    ax.text(position[0] + size[0] / 2, position[1] + size[1] / 2, layer_name, 
            horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')

fig, ax = plt.subplots(figsize=(12, 6))

# Define layer positions and sizes
layers = [
    ("LSTM (64)", (2, 1), (0, 5)),
    ("Dropout (0.3)", (3, 1), (3, 5)),
    ("LSTM (45)", (2, 1), (7, 5)),
    ("Dropout (0.2)", (3, 1), (10, 5)),
    ("Dense (1)", (2, 1), (14, 5)),
    ("Compile", (2, 1), (17, 5), 'lightgreen'),
    ("EarlyStopping", (3, 1), (19, 5), 'lightcoral'),
    # ("ModelCheckpoint", (3, 1), (19, 4), 'lightcoral'),
    # ("ReduceLROnPlateau", (4, 1), (19, 2), 'lightcoral')
]

# Draw layers
for layer in layers:
    draw_layer(ax, *layer)

# Set limits and hide axes
ax.set_xlim(0, 22)
ax.set_ylim(1, 8)
ax.axis('off')

plt.tight_layout()
plt.show()
