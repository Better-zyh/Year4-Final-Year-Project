import numpy as np
import matplotlib.pyplot as plt
# Training Time Comparison
fig, ax = plt.subplots()
ax.set_ylabel('Time')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.rcParams['axes.unicode_minus'] = False  
barWidth = 0.2
# values of y-axis
bars1 = [44.4729,23.7673,24.3078,30.8493]
bars2 = [24.5032,21.1196,23.7143,23.1124]
bars3 = [38.4417,39.3136,39.4897,39.0817]
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
plt.bar(r1, bars1, color='r', width=barWidth, edgecolor='white', label='LSTM')
plt.bar(r2, bars2, color='y', width=barWidth, edgecolor='white', label='GRU')
plt.bar(r3, bars3, color='b', width=barWidth, edgecolor='white', label='Bi-LSTM')
# label of x-axis
plt.xticks([r + barWidth for r in range(len(bars1))], ['Round 1','Round 2','Round 3','Average'])
plt.title("Traning time of different models")
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.show()

# Mean Square Error Comparison
fig, ax = plt.subplots()
ax.set_ylabel('MSE')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.rcParams['axes.unicode_minus'] = False 
barWidth = 0.2
# values of y-axis
bars1 = [4.3340,7.6449,4.4109,5.4633]
bars2 = [4.6110,5.9706,3.6978,4.7598]
bars3 = [4.6208,5.2370,5.7812,5.2130]
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
plt.bar(r1, bars1, color='r', width=barWidth, edgecolor='white', label='LSTM')
plt.bar(r2, bars2, color='y', width=barWidth, edgecolor='white', label='GRU')
plt.bar(r3, bars3, color='b', width=barWidth, edgecolor='white', label='Bi-LSTM')
# label of x-axis
plt.xticks([r + barWidth for r in range(len(bars1))], ['Round 1','Round 2','Round 3','Average'])
plt.title("Mean Square Error of Different Models")
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.show()

