import squarify # pip install squarify (algorithm for treemap)
import matplotlib.pyplot as plt
import pandas as pd



# set a higher resolution
plt.rcParams['figure.dpi'] = 300

df = pd.read_csv("https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/simple-treemap.csv")

print(df)

fig, ax = plt.subplots(figsize=(10,10))
ax.set_axis_off()

# add treemap
squarify.plot(
   sizes=df["value"],
   label=df["name"],
   ax=ax
)

# display plot
plt.savefig('Treestuff.jpg')
