"""
Make the Iris dataset tutorial-sized.

"""
import csv

with open('iris.csv') as f:
    rows = [r for r in csv.reader(f) if r]

with open('iris.txt', 'w') as f:
    f.write("Sepal length (cm)"
            " Petal length (cm)"
            " Versicolor (0) or Virginica (1)")
    for r in rows:
        if 'versicolor' in r[-1]:
            label = 0
        elif 'virginica' in r[-1]:
            label = 1
        else:
            continue

        f.write("{:17} {:17} {:31}\n".format(float(r[0]), float(r[2]), label))
