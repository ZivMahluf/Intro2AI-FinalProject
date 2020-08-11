import joblib
import matplotlib.pyplot as plt


# read python dict back from the file
mydict2 = joblib.load('losses.pkl', 'r')
x0 = [x[0] for x in mydict2]
x1 = [x[1] for x in mydict2]
x2 = [x[2] for x in mydict2]
plt.subplot(111)
plt.plot(x0)
plt.show()
plt.subplot(211)
plt.plot(x1)
plt.show()
plt.subplot(311)
plt.plot(x2)
plt.show()
# for x in mydict2:
#     # print(x)
#     plt.plot(x)


