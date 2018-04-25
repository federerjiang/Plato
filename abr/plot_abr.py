import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
# import matplotlib
# matplotlib.use('Agg')

# raw_data = {'first_name': ['full', 'partial', 'mm', 'Plato'],
#         'quality': [4, 24, 31, 2],
#         'smooth': [25, 94, 57, 62],
#         'cv': [25, 94, 57, 62],
#         'rebuf': [5, 43, 23, 23]}
raw_data = {'first_name': ['quality', 'smooth', 'cv', 'rebuf'],
        'full': [4, 24, 31, 2],
        'partial': [4, 24, 31, 2],
        'mm': [25, 94, 57, 62],
        'Plato': [30, 100, 60, 75]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'full', 'partial', 'mm', 'Plato'])
print(df)
# print(df['first_name'][2]
pos = list((range(len(df['first_name']))))
width = 0.15
print(pos)

fig, ax = plt.subplots(figsize=(10, 5))

plt.bar(pos,
        df['full'],
        width,
        alpha=0.5,
        color='blue',
        label='full')

plt.bar([p + width for p in pos],
        df['partial'],
        width,
        alpha=0.5,
        color='green',
        label='partial')

plt.bar([p + width*2 for p in pos],
        df['mm'],
        width,
        alpha=0.5,
        color='yellow',
        label='mm')

plt.bar([p + width*3 for p in pos],
        df['Plato'],
        width,
        alpha=0.5,
        color='red',
        label='Plato')

ax.set_ylabel('Score')
# ax.set_title('Test ')
ax.set_xticks([p + 1.5 * width for p in pos])
ax.set_xticklabels(df['first_name'])
plt.xlim(min(pos)-width, max(pos)+width*5)
# plt.ylim([0, max(df['full'], df['partial'], df['mm'], df['Plato'])])
plt.ylim([0, 100])
# plt.legend(['full', 'partial', 'mm', 'Plato'], loc='upper left')
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
plt.grid()
# plt.show()
fig.savefig('abr-all' + '.eps', format='eps', dpi=1000)



